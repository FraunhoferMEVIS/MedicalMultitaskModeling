from __future__ import annotations

import json
import logging
import random
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch.nn as nn
from pydantic import Field
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from typing_extensions import Annotated

from mmm.data_loading.RegressionDataset import RegressionDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.logging.type_ext import StepMetricDict
from mmm.logging.wandb_ext import build_wandb_image, build_wandb_image_for_clf
from mmm.mmm_types.GroupUsage import GroupUsage
from mmm.mtl_modules.shared_blocks.Grouper import Grouper, make_grid_for_supercase
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.neural.losses import MSELossConfig, RMSELossConfig, SmoothL1Loss
from mmm.utils import flatten_list_of_dicts

from .MTLTask import MTLTask


class RegressionTask(MTLTask):
    """
    Special case of a MTLTask which deals with regression directly from the backbone's output $$Z$$.

    It requires cohorts holding regression datasets.
    """

    class Config(MTLTask.Config):
        encoder_key: str = "encoder"
        squeezer_key: str = "squeezer"
        grouper_key: GroupUsage = Field(
            default=GroupUsage(grouper_key=""),
            description="If the key is set, assumes a grouper to exist in the shared modules.",
        )
        loss_fn: Annotated[
            MSELossConfig | SmoothL1Loss.Config, Field(discriminator="loss_type")
        ] = SmoothL1Loss.Config()
        dropout: float = 0.2
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
        self.flatten = nn.Flatten(1)
        self.criterion: nn.Module = self.args.loss_fn.build_instance()
        self._grouper_meta = None

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
        batch["target"] = batch["target"].unsqueeze(-1).to(self.torch_device).float()
        return batch

    def forward(self, inputs, shared_blocks: Dict[str, SharedBlock]):
        x, supercase_indexes = inputs
        pyr = shared_blocks[self.args.encoder_key](x)
        _, squeezed = shared_blocks[self.args.squeezer_key](pyr)
        hidden_vector = self.flatten(squeezed)

        if self.args.grouper_key.grouper_key:
            hidden_vector, self._grouper_meta = shared_blocks[self.args.grouper_key.grouper_key](
                hidden_vector, supercase_indexes
            )

        return self.task_modules["regression_head"](hidden_vector)

    def training_step(self, batch: Dict[str, Any], shared_blocks: SharedModules):
        x = batch["image"]
        y = batch["target"]

        # skip if batch is empty
        tensornums = [bool(t.numel()) for t in x]
        if not True in tensornums:
            logging.info("encountered batch without valid training examples, skipping batch")
            return None

        if self.args.grouper_key.grouper_key:
            # A batch with ids ["id1", "s3", "s3"] would become [0, 1, 1]
            grouper: Grouper = shared_blocks.module.shared_modules[self.args.grouper_key.grouper_key]
            supercase_indices = grouper.extract_ids_from_batch(
                [x["group_id"] for x in batch["meta"]], for_task_name=self.get_name()
            ).to(self.torch_device)

            y = grouper.group_targets(y, supercase_indices, self.args.grouper_key)
        else:
            supercase_indices = None

        y_hat = shared_blocks.forward((x, supercase_indices), self.forward)

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
            supercase_indices=supercase_indices,
        )

        return batch_loss, live_vis

    def _visualize_preds(
        self, training_ims, step_metrics: Dict, metas: List[Dict], supercase_indices
    ) -> Dict[str, Any]:
        vis_n = min(self.ask_for_visualization(), training_ims.size(0))

        if vis_n <= 0:
            return {}

        if supercase_indices is not None:
            # Select one of the groups for visualization
            group_index = random.choice(list(set(supercase_indices.cpu().numpy())))
            grid_img, weight_str, vis_indices, _ = make_grid_for_supercase(
                training_ims, supercase_indices, group_index, self._grouper_meta
            )

            caption = f"""
Group {group_index} with {len(vis_indices)} subcases, group id: {metas[0]["group_id"]}
weights:
{weight_str}
preds:
{step_metrics["preds"][group_index]}
targets:
{step_metrics["targets"][group_index]}
{[metas[i] for i in vis_indices]=}
            """
            wandb_img = build_wandb_image(
                im=grid_img,
                caption=caption,
            )
            return {"preds": [wandb_img]}
        else:
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
