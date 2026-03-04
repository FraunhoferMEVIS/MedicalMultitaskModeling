from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pydantic import Field
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from mmm.data_loading.MultilabelClassificationDataset import MultilabelClassificationDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.logging.batch_visualization import visualize_batch
from mmm.logging.type_ext import StepMetricDict
from mmm.logging.wandb_ext import build_wandb_image, remove_wandb_special_chars
from mmm.mmm_types.GroupUsage import GroupingStrategy, GroupUsage
from mmm.mtl_modules.shared_blocks.Grouper import Grouper
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.settings import mtl_settings
from mmm.utils import flatten_list_of_dicts

from .MTLTask import MTLTask


class MultilabelClassificationTask(MTLTask):
    """
    Multilabel task. Takes a single image as input and can predict multiple binary classes.
    If you have multiple multi-class problems, you either need to convert that problem
    into a binary problem for each class or multiple multi-class classification tasks.

    Loss weights are optional.
    If a loss weight is zero this class will not contribute to the loss or metrics.

    For AUC computation, classes completely missing are ignored.
    """

    class Config(MTLTask.Config):
        encoder_key: str = "encoder"
        squeezer_key: str = "squeezer"
        grouper_key: GroupUsage = Field(
            default=GroupUsage(grouper_key="grouper"),
            description="If the key is set, assumes a grouper to exist in the shared modules.",
        )
        confidence_threshold: float = 0.5
        dropout: float = 0.2
        metrics: List[Literal["acc", "f1", "auc", "confmat"]] = ["acc", "auc", "confmat"]

    def __init__(
        self,
        hidden_dim: int,
        args: Config,
        cohort: TrainValCohort[MultilabelClassificationDataset],
    ):
        super().__init__(args, cohort)
        self.args: MultilabelClassificationTask.Config
        self.hidden_dim: int = hidden_dim
        self.class_names: List[str] = cohort.datasets[0].class_names

        self.flatten = nn.Flatten(1)
        self.task_modules = nn.ModuleDict(
            {
                "classification_head": nn.Sequential(
                    nn.Dropout(p=self.args.dropout),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, len(self.class_names)),
                )
            }
        )
        self._grouper_meta = None

    def prepare_batch(self, batch: Dict[str, Any]) -> Any:
        batch["image"] = batch["image"].to(self.torch_device)
        batch["class_labels"] = batch["class_labels"].to(self.torch_device)
        if "loss_weights" in batch:
            batch["loss_weights"] = batch["loss_weights"].to(self.torch_device)
        return batch

    def forward(self, inputs, shared_blocks: Dict[str, SharedBlock]):
        # Enable image, supercase_indices for backward compatibility
        x, supercase_indexes, contexts = inputs if len(inputs) == 3 else (inputs[0], inputs[1], None)

        if MultilabelClassificationDataset.batch_is_compressed(x):
            hidden_vector = x
        else:
            pyr = shared_blocks[self.args.encoder_key](x)
            _, squeezed = shared_blocks["squeezer"](pyr)
            hidden_vector = self.flatten(squeezed)

        if self.args.grouper_key.grouper_key:
            if hasattr(self.args, "positions") and self.args.positions is not None:
                positions = [c[self.args.positions[0]] for c in contexts]
            else:
                positions = None
            hidden_vector, self._grouper_meta = shared_blocks[self.args.grouper_key.grouper_key](
                hidden_vector, supercase_indexes, self.args.grouper_key, positions=positions
            )

        out = self.task_modules["classification_head"](hidden_vector)
        return out

    def training_step(self, batch: Dict[str, Any], shared_blocks: SharedModules) -> Tuple[torch.Tensor, Dict]:
        x = batch["image"]
        y = batch["class_labels"]
        w = batch["loss_weights"] if "loss_weights" in batch else None

        metas = batch["meta"] if "meta" in batch else [{} for _ in range(len(x))]
        group_ids = [x.get("group_id") for x in metas]
        contexts = [x.get("context") for x in metas]
        supercase_indices = Grouper.extract_ids_from_batch(group_ids, for_task_name=self.get_name()).to(
            self.torch_device
        )
        if self.args.grouper_key.grouper_key:
            grouper: Grouper = shared_blocks.module.shared_modules[self.args.grouper_key.grouper_key]  # type: ignore
            y, target_indices = grouper.group_targets(y, supercase_indices, self.args.grouper_key, return_indices=True)
            assert self.args.grouper_key.grouping is GroupingStrategy.full, "Implement other grouping strategies."
            # metas = [metas[i] for i in target_indices]
            # group_ids, contexts = [group_ids[i] for i in target_indices], [contexts[i] for i in target_indices]
            if w is not None:
                w = w[target_indices]

        y_hat = shared_blocks.forward((x, supercase_indices, contexts), self.forward)

        batch_loss = F.binary_cross_entropy_with_logits(y_hat, y, weight=w) * 1.44269  # log2(exp(1))
        if w is not None:
            batch_loss = batch_loss / w.mean()

        with torch.no_grad():
            step_results: StepMetricDict = {  # type: ignore (.numpy() does not correctly indicate numpy array)
                "targets": y.cpu().numpy(),
                "logits": y_hat.detach().cpu().float().numpy(),
                "preds": torch.sigmoid(y_hat).detach().cpu().float().numpy(),
                "weights": (w.detach().cpu().numpy() if w is not None else np.ones(y.shape)),
            }
            step_results = self.add_meta_info(step_results, metas)
            one_index_per_supercase = [
                (supercase_indices == s_index).nonzero()[0].item() for s_index in supercase_indices.unique()
            ]
            # if self.args.grouper_key.grouper_key: # and self.args.grouper_key.grouping is GroupingStrategy.single:
            # For each supercase,
            # step_results = {k: v[one_index_per_supercase] for k, v in step_results.items()}

            self.add_step_result(batch_loss.item(), step_results)

            live_vis = {}
            if self.ask_for_visualization():
                batch_info = {}
                # Biggest batchsize is the maximum number of occurrences of a supercase_index
                if (
                    self._grouper_meta is not None
                    and (max_batchsize := torch.bincount(supercase_indices).max().item())
                    < mtl_settings.max_classes_detailed_logging
                ):
                    logging.info(f"Visualizing grouper attention weights for batch with {max_batchsize=}")
                    batch_info.update(
                        {
                            "group_id_to_index": {g: supercase_indices[i].item() for i, g in enumerate(group_ids)},
                            "graphs": {
                                "attn_avg": self._grouper_meta["attn_weights"].mean(dim=1),  # average heads
                                "lastweights_avg": self._grouper_meta["lastweights"].mean(dim=1),  # average heads
                            },
                        }
                    )
                    for head_idx in range(self._grouper_meta["attn_weights"].shape[1]):
                        batch_info["graphs"][f"attn_head_{head_idx}"] = self._grouper_meta["attn_weights"][
                            :, head_idx, ...
                        ]
                        batch_info["graphs"][f"lastweights_head_{head_idx}"] = self._grouper_meta["lastweights"][
                            :, head_idx, ...
                        ]

                log = visualize_batch(
                    x.detach().cpu(),
                    metas,
                    captions=[
                        "<br>".join(
                            [
                                f"{cls_name}({target}, w:{weight})>{pred:.2f}"
                                for cls_name, target, pred, weight in zip(
                                    self.class_names,
                                    step_results["targets"][vis_index],
                                    step_results["preds"][vis_index],
                                    step_results["weights"][vis_index],
                                )
                            ]
                        )
                        for vis_index in range(x.shape[0])
                    ],
                    batch_info=batch_info,
                )
                log.upload()
                live_vis["preds"] = log.build_instruction()

        return batch_loss, live_vis

    def add_meta_info(self, step_results, metas) -> dict:
        return step_results

    def log_epoch_metrics(self, return_metrics=False) -> Tuple[Dict[str, Any], str]:
        metric_computers = {"acc": accuracy_score, "f1": f1_score}
        metrics = flatten_list_of_dicts(self._step_metrics)

        _, print_str = super().log_epoch_metrics()
        log_dict = {}

        preds_per_class = [
            metrics["preds"][:, i] > self.args.confidence_threshold for i, _ in enumerate(self.class_names)
        ]
        weights_per_class = [metrics["weights"][:, i] for i, _ in enumerate(self.class_names)]

        for metric_literal in self.args.metrics:
            try:
                if metric_literal in ["acc", "f1"]:
                    metrics_per_class = {
                        f"{self.class_names[i]}_{metric_literal}": metric_computers[metric_literal](
                            y_true=metrics["targets"][:, i],
                            y_pred=preds_per_class[i],
                            sample_weight=weights_per_class[i],
                        )
                        for i, _ in enumerate(self.class_names)
                    }
                elif metric_literal == "auc":
                    metrics_per_class = {
                        f"{self.class_names[i]}_{metric_literal}": roc_auc_score(
                            y_true=metrics["targets"][:, i],
                            y_score=metrics["logits"][:, i],
                            sample_weight=weights_per_class[i],
                        )
                        for i, _ in enumerate(self.class_names)
                        if len(np.unique(metrics["targets"][:, i])) > 1
                    }
                    metrics_per_class = {k: v for k, v in metrics_per_class.items() if not np.isnan(v)}
                elif metric_literal in ["confmat"]:
                    pass  # does not fit in metrics_per_class
                else:
                    raise Exception(f"Unknown metric {metric_literal}")
                log_dict[f"{metric_literal}_mean"] = np.mean(list(metrics_per_class.values()))  # type: ignore
                print_str = f"{print_str} - {metric_literal}_mean: {log_dict[f'{metric_literal}_mean']}"
                for class_name, v in metrics_per_class.items():
                    log_dict[f"{remove_wandb_special_chars(class_name)}"] = v
            except Exception as e:
                logging.error(f"Error in metric {metric_literal}: {e}")

            if metric_literal == "confmat":
                for class_index, class_name in enumerate(self.class_names):
                    log_dict[f"confmat_{remove_wandb_special_chars(class_name)}"] = wandb.plot.confusion_matrix(
                        preds=(metrics["preds"][:, class_index] > 0.5).astype(int),
                        y_true=metrics["targets"][:, class_index],
                        class_names=["no", "yes"],
                        title=f"{self._prefix} {class_name}",
                    )
        return (log_dict, print_str, metrics) if return_metrics else (log_dict, print_str)

    def needs_shared_blocks(self):
        res = [
            self.args.encoder_key,
            self.args.squeezer_key,
        ]

        if self.args.grouper_key.grouper_key:
            res.append(self.args.grouper_key.grouper_key)

        return res


class BCESurvivalTask(MultilabelClassificationTask):
    """
    Interprets the class labels as binary survival labels for each time step.

    Computes a C-index.

    Inference is done by using the last positive class label as the predicted time.
    """

    class Config(MultilabelClassificationTask.Config):
        metrics: List[Literal["acc", "f1", "auc", "confmat"]] = ["acc", "auc"]
        dropout: float = 0.5

    @staticmethod
    def clfdataset_from_regdataset(d: dict[str, Any], bins: torch.FloatTensor) -> dict[str, Any]:
        """
        The regression target (key: target) is assumed to be a float. The event indicator is in meta.event.
        """
        assert len(bins.shape) == 1 and isinstance(bins, torch.FloatTensor), "Bins should be a 1D tensor"

        d["meta"]["regression_target"] = d["target"].item() if isinstance(d["target"], torch.Tensor) else d["target"]

        if "event" in d["meta"]:
            if d["meta"]["event"] == 1:
                # If there is an event, the class labels are set to 1 for all bins after the event
                d["class_labels"] = (bins >= d["target"]).float()
                d["loss_weights"] = torch.ones_like(d["class_labels"])
            else:
                # Prepare the loss weights such that the values after censoring are ignored
                d["class_labels"] = torch.zeros_like(bins)
                d["loss_weights"] = (~(bins >= d["target"]).float().bool()).float()
                # At least the first loss weight should be 1
                d["loss_weights"][0] = 1
            assert d["loss_weights"].sum() > 0, "No loss weights set"

        return d

    def add_meta_info(self, step_results, metas) -> Dict:
        super_metrics = super().add_meta_info(step_results, metas)
        super_metrics["timetoevent"] = torch.tensor([m["regression_target"] for m in metas])
        super_metrics["event"] = torch.tensor([m["event"] for m in metas])
        return super_metrics

    def log_epoch_metrics(self) -> Tuple[Dict[str, Any] | str]:
        log_dict, print_str, metrics = super().log_epoch_metrics(return_metrics=True)
        # Only keep mean values in log_dict
        log_dict = {k: v for k, v in log_dict.items() if k.endswith("_mean")}

        risk_scores = metrics["preds"].sum(axis=1)
        risk_table = wandb.Table(
            columns=["event", "risk", "timetoevent"],
            data=[[metrics["event"][i], risk_scores[i], metrics["timetoevent"][i]] for i in range(len(risk_scores))],
        )
        log_dict.update(
            {
                "num_events": (num_events := metrics["event"].sum()),
                "risk_table": risk_table,
            }
        )

        # Compute C-index by using the sum of predictions as the predicted risk
        if num_events > 0:
            from lifelines.utils import concordance_index

            # Also use lifelines package:
            log_dict["c-index"] = concordance_index(
                # event_times=metrics["targets"].squeeze(),
                event_times=metrics["timetoevent"],
                # predicted_scores=risk.squeeze(),
                predicted_scores=risk_scores * -1,
                # event_observed=metrics["event"].squeeze().astype(bool),
                event_observed=metrics["event"].astype(bool),
            )
            # try:
            #     from sksurv.metrics import concordance_index_censored

            #     c_index, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
            #         metrics["event"].astype(bool), metrics["timetoevent"], risk_scores
            #     )
            #     assert c_index == log_dict["c-index"]
            # logs = {
            #         "c-index": c_index,
            #         "concordant": concordant,
            #         "discordant": discordant,
            #         "tied_risk": tied_risk,
            #         "tied_time": tied_time,
            #     }
            # )
            # except ImportError:
            #     pass

        return log_dict, print_str
