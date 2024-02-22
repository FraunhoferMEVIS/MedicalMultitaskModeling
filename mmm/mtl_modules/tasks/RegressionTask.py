from __future__ import annotations
import json
from typing import Any, List, Dict, Tuple, Optional, Literal
from typing_extensions import Annotated
import random
import logging

import numpy as np
import torch.nn as nn
from pydantic import Field

from mmm.logging.type_ext import StepMetricDict
from mmm.logging.wandb_ext import build_wandb_image

from .MTLTask import MTLTask
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.data_loading.RegressionDataset import RegressionDataset
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.neural.pooling import GlobalPooling, GlobalPoolingConfig
from mmm.neural import LossConfigs, MSELossConfig
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules

from mmm.utils import flatten_list_of_dicts
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score


class RegressionTask(MTLTask):
    """
    Special case of a MTLTask which deals with regression directly from the backbone's output $$Z$$.

    It requires cohorts holding regression datasets.
    """

    class Config(MTLTask.Config):
        encoder_key: str = "encoder"
        squeezer_key: str = "squeezer"
        loss_fn: Annotated[LossConfigs, Field(discriminator="loss_type")] = MSELossConfig()
        dropout: float = 0.2
        global_pooling: GlobalPoolingConfig = Field(
            default=GlobalPoolingConfig(pooling_type=GlobalPooling.AveragePooling),
            description="Adaptive pooling to reduce the dimensions of the last tensor before the latent space z.",
        )
        metrics: Optional[List[Literal["mae", "mse", "rmse", "max_error", "r2_score"]]] = Field(
            default=None,
            description="If none, the task will decide which metrics make sense",
        )
        head: Literal["pretraining", "smart"] = "pretraining"

    def __init__(self, hidden_dim: int, args: Config, cohort: TrainValCohort[RegressionDataset]):
        super().__init__(args, cohort)
        self.args: RegressionTask.Config  # Make sure IDE knows about the task specific fields
        self.hidden_dim = hidden_dim

        if self.args.head == "pretraining":
            self.task_modules = self._create_pretraining_head()
        else:
            self.task_modules = self._create_smart_head()
        self.pooling: nn.Module = self.args.global_pooling.build_instance()
        self.flatten = nn.Flatten(1)
        self.criterion: nn.Module = self.args.loss_fn.build_instance()

    def _create_pretraining_head(self):
        head = nn.Sequential(nn.Dropout(p=self.args.dropout), nn.ReLU(), nn.Linear(self.hidden_dim, 1))

        return nn.ModuleDict({"regression_head": head})

    def _create_smart_head(self) -> nn.ModuleDict:
        head = nn.Sequential(
            nn.Dropout(p=self.args.dropout),
            nn.ReLU(),
            nn.Linear(max(4, self.hidden_dim), max(4, self.hidden_dim // 2)),
            nn.Dropout(p=self.args.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        return nn.ModuleDict({"regression_head": head})

    def prepare_batch(self, batch: Dict[str, Any]) -> Any:
        batch["image"] = batch["image"].to(self.torch_device)
        batch["target"] = batch["target"].unsqueeze(-1).to(self.torch_device)
        return batch

    def forward(self, inputs, shared_blocks: Dict[str, SharedBlock]):
        x, _ = inputs
        pyr = shared_blocks[self.args.encoder_key](x)
        squeezed = shared_blocks["squeezer"](pyr)
        hidden_vector = self.flatten(self.pooling(squeezed))

        return self.task_modules["regression_head"](hidden_vector)

    def training_step(self, batch: Dict[str, Any], shared_blocks: SharedModules):
        x = batch["image"]
        y = batch["target"]

        # skip if batch is empty
        tensornums = [bool(t.numel()) for t in x]
        if not True in tensornums:
            logging.info("encountered batch without valid training examples, skipping batch")
            return None

        y_hat = shared_blocks.forward((x, None), self.forward)

        batch_loss = self.criterion(y_hat, y)

        step_results: StepMetricDict = {  # type: ignore (.numpy() does not correctly indicate numpy array)
            "targets": y.cpu().numpy(),
            "preds": y_hat.detach().cpu().numpy(),
        }
        self.add_step_result(batch_loss.item(), step_results)

        live_vis = self._visualize_preds(
            x.detach().cpu(),
            step_results,
            (batch["meta"] if "meta" in batch else [{} for _ in range(batch["image"].shape[0])]),
        )

        return batch_loss, live_vis

    def _visualize_preds(self, training_ims, step_metrics: Dict, metas: List[Dict]) -> Dict[str, Any]:
        vis_n = min(self._takeout_vis_budget(), training_ims.size(0))

        if vis_n <= 0:
            return {}

        preds = []
        for rand_index in random.sample(list(range(training_ims.size(0))), vis_n):
            metastr = json.dumps(metas[rand_index], default=lambda o: str(o))
            description = f"{np.min(training_ims[rand_index].numpy()):.3f}, {np.max(training_ims[rand_index].numpy()):.3f}; {training_ims[rand_index].size()}"
            preds.append(
                build_wandb_image(
                    training_ims[rand_index],
                    f"{description}\nTrue: {step_metrics['targets'][rand_index]}\nPred: {step_metrics['preds'][rand_index]}\n{metastr}",
                )
            )
        return {"preds": preds} if preds else {}

    def log_epoch_metrics(self) -> Tuple[Dict[str, Any], str]:
        metrics = flatten_list_of_dicts(self._step_metrics)
        if self.args.metrics is None:
            selected_metrics = ["mae", "rmse", "r2_score", "max_error"]
        else:
            selected_metrics = self.args.metrics

        _, print_str = super().log_epoch_metrics()
        log_dict: Dict[str, Any] = {}

        if "mae" in selected_metrics:
            log_dict["mae"] = mean_absolute_error(metrics["targets"], metrics["preds"])

        if "mse" in selected_metrics:
            log_dict["mse"] = mean_squared_error(metrics["targets"], metrics["preds"])

        if "rmse" in selected_metrics:
            log_dict["rmse"] = np.sqrt(mean_squared_error(metrics["targets"], metrics["preds"]))

        if "r2_score" in selected_metrics:
            log_dict["r2_score"] = r2_score(metrics["targets"], metrics["preds"])

        if "max_error" in selected_metrics:
            log_dict["max_error"] = max_error(metrics["targets"], metrics["preds"])

        return log_dict, print_str
