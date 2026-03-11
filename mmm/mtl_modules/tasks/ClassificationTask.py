from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import wandb
from PIL.Image import Image as PIL_Image
from pydantic import Field
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch.utils.data import Dataset
from typing_extensions import Annotated

from mmm.data_loading.ClassificationDataset import ClassificationDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.logging.batch_visualization import visualize_batch
from mmm.logging.type_ext import StepMetricDict
from mmm.mmm_types.GroupUsage import GroupUsage
from mmm.mtl_modules.shared_blocks.Grouper import Grouper
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.neural import CrossEntropyLossConfig, LossConfigs
from mmm.settings import mtl_settings
from mmm.utils import flatten_list_of_dicts

from .MTLTask import MTLTask

CLF_METRICS = Literal["confusion matrix", "accuracy", "top5accuracy", "auc", "f1", "kappa"]


class ClassificationTask(MTLTask):
    """
    Special case of a MTLTask which deals with classification directly from the backbone's output $$Z$$.

    It requires cohorts holding classification datasets.

    Labels are encoded by their index in `class_names`.
    In consequence, the order needs to correspond to the labels!!
    """

    class Config(MTLTask.Config):
        encoder_key: str = "encoder"
        squeezer_key: str = "squeezer"
        grouper_key: GroupUsage = Field(
            default=GroupUsage(grouper_key=""),
            description="If the key is set, assumes a grouper to exist in the shared modules.",
        )
        loss_fn: Annotated[LossConfigs, Field(discriminator="loss_type")] = CrossEntropyLossConfig()
        dropout: float = 0.2
        metrics: Optional[List[CLF_METRICS]] = Field(
            default=None,
            description="If none, the task will decide which metrics make sense",
        )
        head: Literal["pretraining", "pretraining_gelu"] = "pretraining"

    def __init__(
        self,
        hidden_dim: int,
        args: Config,
        cohort: TrainValCohort[ClassificationDataset],
    ):
        super().__init__(args, cohort)
        self.args: ClassificationTask.Config  # Make sure IDE knows about the task specific fields
        self.class_names = cohort.datasets[0].get_classes_for_visualization()
        self.hidden_dim = hidden_dim

        self.task_modules = self._create_pretraining_head()
        self.task_modules.update(self._build_context_modules(self.hidden_dim))

        self._grouper_meta = None
        self.flatten = nn.Flatten(1)
        self.criterion: nn.Module = self.args.loss_fn.build_instance()

    def _create_pretraining_head(self):
        out_dim = len(self.class_names)
        new_dict = nn.ModuleDict(
            {
                "classification_head": nn.Sequential(
                    nn.Dropout(p=self.args.dropout),
                    nn.ReLU() if self.args.head == "pretraining" else nn.GELU(),
                    nn.Linear(self.hidden_dim, out_dim),
                )
            }
        )
        return new_dict

    def prepare_batch(self, batch: Dict[str, Any]) -> Any:
        batch["image"] = batch["image"].to(self.torch_device)
        batch["class"] = batch["class"].to(self.torch_device)
        return batch

    def forward(self, inputs, shared_blocks: Dict[str, SharedBlock]):
        # Enable image, supercase_indices for backward compatibility
        x, supercase_indexes, contexts = inputs if len(inputs) == 3 else (inputs[0], inputs[1], None)

        if ClassificationDataset.batch_is_compressed(x):
            hidden_vector = x
        else:
            pyr = shared_blocks[self.args.encoder_key](x)
            _, hidden_vector = shared_blocks[self.args.squeezer_key](pyr)
            hidden_vector = self.flatten(hidden_vector)

        if (
            hasattr(self.args, "token_contexts") and len(self.args.token_contexts) > 0
        ):  # Only if there is at least one context which should be used
            for ctx in self.args.token_contexts:
                ctx_item = [c[ctx.index_in_context] for c in contexts]
                hidden_vector = self.task_modules[f"{ctx.index_in_context}"](hidden_vector, ctx_item)

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

    def training_step(self, batch: dict[str, Any], shared_blocks: SharedModules):
        x: torch.Tensor = batch["image"]
        y: torch.Tensor = batch["class"]
        metas = batch.get("meta", [{} for _ in range(batch["image"].shape[0])])

        if not True in [bool(t.numel()) for t in x]:  # skip if batch is empty
            logging.info("encountered batch without valid training examples, skipping batch")
            return None

        # If a grouper is used, extract supercase_indices
        group_ids = [x.get("group_id") for x in metas]
        contexts = [x.get("context") for x in metas]
        supercase_indices = Grouper.extract_ids_from_batch(group_ids, for_task_name=self.get_name()).to(
            self.torch_device
        )
        if self.args.grouper_key.grouper_key:
            # A batch with ids ["id1", "s3", "s3"] would become [0, 1, 1]
            grouper: Grouper = shared_blocks.module.shared_modules[self.args.grouper_key.grouper_key]  # type: ignore

            # the targets need to be grouped as well, currently y is a (B,) tensor with class indices
            # For each unique supercase index, we need to find the corresponding class index
            y = grouper.group_targets(y, supercase_indices, self.args.grouper_key)

        y_hat = shared_blocks.forward((x, supercase_indices, contexts), self.forward)
        batch_loss = self.criterion(y_hat, y) / np.log(len(self.class_names))

        step_results: StepMetricDict = {  # type: ignore (.numpy() does not correctly indicate numpy array)
            "targets": y.cpu().numpy(),
            "logits": y_hat.detach().cpu().float().numpy(),
            "preds": torch.argmax(y_hat.detach().cpu().float(), dim=1).numpy(),
        }
        self.add_step_result(batch_loss.item(), step_results)

        realtime_log = {}
        if self.ask_for_visualization():
            batch_info = {}
            if self._grouper_meta is not None:
                batch_info.update(
                    {
                        "group_id_to_index": {g: supercase_indices[i].item() for i, g in enumerate(group_ids)},
                        "grouper_meta": self._grouper_meta,
                        "graphs": {
                            "attn_avg": self._grouper_meta["attn_weights"].mean(dim=1),  # average heads
                            "lastweights_avg": self._grouper_meta["lastweights"].mean(dim=1),  # average heads
                        },
                    }
                )
            log = visualize_batch(
                x.detach().cpu(),
                metas,
                captions=[
                    f"""target = {step_results["targets"][i]} ({self.class_names[step_results["targets"][i].item()]}),
<br>pred = {step_results["preds"][i]} ({self.class_names[step_results["preds"][i].item()]}),
<br>logits = {step_results["logits"][i]}"""
                    for i in range(x.shape[0])
                ],
                batch_info=batch_info,
            )
            log.upload()
            realtime_log["preds"] = log.build_instruction()

        return batch_loss, realtime_log

    def _get_short_class_names(self, max_length=10):
        if True in [len(c) > max_length for c in self.class_names]:
            return [f"{i};{c[:max_length]}" for i, c in enumerate(self.class_names)]
        else:
            return self.class_names

    @staticmethod
    def compute_metrics(y_true, y_pred, y_score: np.ndarray | None, selected_metrics, plot_info: dict | None = None):
        log_dict, print_str = {}, ""
        if "accuracy" in selected_metrics:
            log_dict["acc"] = accuracy_score(y_true=y_true, y_pred=y_pred)
            log_dict["acc_balanced"] = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
            print_str = f"{print_str} - acc: {log_dict['acc']}"

        if "auc" in selected_metrics and y_score is not None:
            try:
                log_dict["auc"] = roc_auc_score(
                    y_true, y_score[:, 1] if y_score.shape[1] == 2 else y_score, multi_class="ovr"
                )

                print_str = f"{print_str} - auc: {log_dict['auc']}"
            except ValueError as e:
                logging.warning(f"Computing auc failed with {e}")

        if "confusion matrix" in selected_metrics:
            log_dict["confmat"] = wandb.plot.confusion_matrix(
                preds=y_pred,  # type: ignore
                y_true=y_true,  # type: ignore
                class_names=plot_info.get("classnames", [f"C{i}" for i in range(len(np.unique(y_true)))]),
                title=plot_info.get("confmat_title", "Confusion Matrix"),
            )

        if "f1" in selected_metrics:
            log_dict["f1"] = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
            log_dict["f1_weighted"] = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")

        return log_dict, print_str

    def log_epoch_metrics(self) -> Tuple[Dict[str, Any], str]:
        metrics = flatten_list_of_dicts(self._step_metrics)
        if self.args.metrics is None:
            if len(self.class_names) <= mtl_settings.max_classes_detailed_logging:
                selected_metrics = [
                    "confusion matrix",
                    "accuracy",
                    "auc",
                    "f1",
                    "kappa",
                ]
            else:
                selected_metrics = ["accuracy", "top5accuracy"]
        else:
            selected_metrics = self.args.metrics

        _, print_str = super().log_epoch_metrics()

        scores = nn.Softmax(dim=1)(torch.from_numpy(metrics["logits"]).float())
        log_dict, standard_metrics_printstr = self.compute_metrics(
            metrics["targets"],
            metrics["preds"],
            scores,
            selected_metrics,
            plot_info={
                "classnames": self._get_short_class_names(),
                "confmat_title": f"{self._prefix}_{self.get_name()}",
            },
        )
        print_str = f"{print_str} {standard_metrics_printstr}"

        if "top5accuracy" in selected_metrics:
            classes_in_loop = np.unique(metrics["targets"])
            classes_in_loop.sort()
            log_dict["top5acc"] = top_k_accuracy_score(
                y_true=metrics["targets"],
                y_score=metrics["logits"][:, classes_in_loop],
                k=5,
                labels=classes_in_loop,
            )
            print_str = f"{print_str} - top5acc: {log_dict['top5acc']}"

        if "kappa" in selected_metrics:
            log_dict["kappa_linear"] = cohen_kappa_score(y1=metrics["targets"], y2=metrics["preds"], weights="linear")
            log_dict["kappa_quadratic"] = cohen_kappa_score(
                y1=metrics["targets"], y2=metrics["preds"], weights="quadratic"
            )

        return log_dict, print_str

    def needs_shared_blocks(self):
        res = [
            self.args.encoder_key,
            self.args.squeezer_key,
        ]

        if self.args.grouper_key.grouper_key:
            res.append(self.args.grouper_key.grouper_key)

        return res
