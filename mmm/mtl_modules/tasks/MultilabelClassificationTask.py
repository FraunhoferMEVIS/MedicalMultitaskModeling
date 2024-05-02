from __future__ import annotations
from typing import Any, List, Dict, Mapping, Tuple, Optional, Type, Literal
import random
from pydantic import Field

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from mmm.logging.type_ext import StepMetricDict

from .MTLTask import MTLTask
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.data_loading.MultilabelClassificationDataset import (
    MultilabelClassificationDataset,
)
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.logging.wandb_ext import remove_wandb_special_chars
from mmm.neural.pooling import GlobalPooling, GlobalPoolingConfig

from mmm.utils import flatten_list_of_dicts
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


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
        confidence_threshold: float = 0.5
        dropout: float = 0.2
        metrics: List[Literal["acc", "f1", "auc"]] = ["acc", "auc"]

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
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, len(self.class_names)),
                )
            }
        )

    def prepare_batch(self, batch: Dict[str, Any]) -> Any:
        batch["image"] = batch["image"].to(self.torch_device)
        batch["class_labels"] = batch["class_labels"].to(self.torch_device)
        if "loss_weights" in batch:
            batch["loss_weights"] = batch["loss_weights"].to(self.torch_device)
        return batch

    def forward(self, x, shared_blocks: Dict[str, SharedBlock]):
        # assert shared_blocks[self.args.encoder_key].training == self.training,\
        #     f"Encoder is in different state than {self.get_name()}"
        pyr = shared_blocks[self.args.encoder_key](x)
        _, squeezed = shared_blocks["squeezer"](pyr)
        hidden_vector = self.flatten(squeezed)
        out = self.task_modules["classification_head"](hidden_vector)
        return out

    def training_step(self, batch: Dict[str, Any], shared_blocks: SharedModules) -> Tuple[torch.Tensor, Dict]:
        x = batch["image"]
        y = batch["class_labels"]
        w = batch["loss_weights"] if "loss_weights" in batch else None

        y_hat = shared_blocks.forward(x, self.forward)

        batch_loss = F.binary_cross_entropy_with_logits(y_hat, y, weight=w) * 1.44269  # log2(exp(1))
        if w is not None:
            batch_loss = batch_loss / w.mean()

        with torch.no_grad():
            step_results: StepMetricDict = {  # type: ignore (.numpy() does not correctly indicate numpy array)
                "targets": y.cpu().numpy(),
                "logits": y_hat.detach().cpu().numpy(),
                "preds": torch.sigmoid(y_hat).detach().cpu().numpy(),
                "weights": (w.detach().cpu().numpy() if w is not None else np.ones(y.shape)),
            }
            self.add_step_result(batch_loss.item(), step_results)

            live_vis = self._visualize_preds(
                x.detach().cpu(),
                step_results,
                batch["meta"] if "meta" in batch else None,
            )

        return batch_loss, live_vis

    def _visualize_preds(self, training_ims, step_metrics: Dict, metainfo: Optional[List[Dict]]) -> Dict[str, Any]:
        vis_n = min(self._takeout_vis_budget(), training_ims.size(0))

        if vis_n <= 0:
            return {}

        preds = []
        for rand_index in random.sample(list(range(training_ims.size(0))), vis_n):
            predstrs = [
                f"{cls_name}({target})>{pred:.2f}"
                for cls_name, target, pred in zip(
                    self.class_names,
                    step_metrics["targets"][rand_index],
                    step_metrics["preds"][rand_index],
                )
            ]
            preds.append(wandb.Image(training_ims[rand_index], caption="\n".join(predstrs)))
        return {"preds": preds} if preds else {}

    def _get_short_class_names(self, max_length=10):
        if True in [len(c) > max_length for c in self.class_names]:
            return [f"{i};{c[:max_length]}" for i, c in enumerate(self.class_names)]
        else:
            return self.class_names

    def log_epoch_metrics(self) -> Tuple[Dict[str, Any], str]:
        metric_computers = {"acc": accuracy_score, "f1": f1_score}
        metrics = flatten_list_of_dicts(self._step_metrics)

        _, print_str = super().log_epoch_metrics()
        log_dict = {}

        preds_per_class = [
            metrics["preds"][:, i] > self.args.confidence_threshold for i, _ in enumerate(self.class_names)
        ]
        weights_per_class = [metrics["weights"][:, i] for i, _ in enumerate(self.class_names)]

        for metric_literal in self.args.metrics:
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
            else:
                raise Exception(f"Unknown metric {metric_literal}")
            log_dict[f"{metric_literal}_mean"] = np.mean(list(metrics_per_class.values()))  # type: ignore
            print_str = f"{print_str} - {metric_literal}_mean: {log_dict[f'{metric_literal}_mean']}"
            for class_name, v in metrics_per_class.items():
                log_dict[f"{remove_wandb_special_chars(class_name)}"] = v

        return log_dict, print_str
