from __future__ import annotations
from typing import Any, List, Dict, Tuple
from typing_extensions import Annotated
import random
import logging

import wandb
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from pydantic import Field
from lifelines.utils import concordance_index

from mmm.logging.type_ext import StepMetricDict
from mmm.logging.wandb_ext import build_wandb_image

from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.data_loading.ClassificationDataset import ClassificationDataset
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.Grouper import Grouper
from mmm.neural.losses import SurvivalLossConfig
from mmm.neural import LossConfigs, SurvivalLossConfig
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.mtl_modules.shared_blocks.Grouper import Grouper

from mmm.utils import flatten_list_of_dicts


class SurvivalPredictionTask(MTLTask):
    """
    Task for survival prediction. Based on literature research, two ways of predicting the survival outcome are possible
    One is a regression task, where parameters for the Cox proportional hazards model are estimated
    The other one is a classification task, in which the survival time is binned into X bins.

    This class therefore heavily draws from the classification and Regression tasks.

    Requires case like:

    {
        "image": ... # following MMM image instance assumptions
        "class": \in Survival bins # binned survival time
        "target": \in Survival time # float
        "meta": {"censor": 0/1} # 0 if event happened, 1 if not
    }
    """

    class Config(MTLTask.Config):
        encoder_key: str = "encoder"
        squeezer_key: str = "squeezer"
        grouper_key: str = "grouper"
        loss_fn: Annotated[LossConfigs, Field(discriminator="loss_type")] = SurvivalLossConfig(
            loss_type="cox_reg", alpha=0.4
        )
        dropout: float = 0.375
        max_visualizations_per_full_train_loop: int = 3

    def __init__(
        self,
        hidden_dim: int,
        args: Config,
        cohort: TrainValCohort[ClassificationDataset],
    ):
        super().__init__(args, cohort)
        self.args: SurvivalPredictionTask.Config  # Make sure IDE knows about the task specific fields
        self.criterion: nn.Module = self.args.loss_fn.build_instance()
        self.hidden_dim = hidden_dim
        if self.criterion.continuous:
            self.task_modules = self._create_pretraining_head(out_dim=1)
        else:
            self.class_names = cohort.datasets[0].get_classes_for_visualization()
            self.task_modules = self._create_pretraining_head(out_dim=len(self.class_names))

        self.flatten = nn.Flatten(1)

    def _create_pretraining_head(self, out_dim: int):
        new_dict = nn.ModuleDict(
            {
                "prediction_head": nn.Sequential(
                    nn.Dropout(p=self.args.dropout),
                    nn.Linear(self.hidden_dim, out_dim),
                )
            }
        )
        return new_dict

    def prepare_batch(self, batch: Dict[str, Any]) -> Any:
        batch["image"] = batch["image"].to(self.torch_device)
        if "class" in batch:
            batch["class"] = batch["class"].to(self.torch_device)
        if "target" in batch:
            batch["target"] = batch["target"].to(self.torch_device)
        assert all(
            ["censor" in list(x.keys()) for x in batch["meta"]]
        ), "To predict survival you need to add 'censor' as key to the meta dict"

        return batch

    def forward(self, inputs, shared_blocks: Dict[str, SharedBlock]):
        x, supercase_indices = inputs
        pyr = shared_blocks[self.args.encoder_key](x)
        _, hidden_vector = shared_blocks[self.args.squeezer_key](pyr)
        hidden_vector = self.flatten(hidden_vector)

        if self.args.grouper_key in list(shared_blocks.keys()):
            hidden_vector, self._grouper_weights = shared_blocks[self.args.grouper_key](
                hidden_vector, supercase_indices
            )

        out = self.task_modules["prediction_head"](hidden_vector)
        return out

    def training_step(self, batch: Dict[str, Any], shared_blocks: SharedModules):
        x = batch["image"]  # bag of images
        y = batch["target"]  # bins of survival

        # skip if batch is empty
        tensornums = [bool(t.numel()) for t in x]
        if not True in tensornums:
            logging.info("encountered batch without valid training examples, skipping batch")
            return None

        # If a grouper is used, extract supercase_indices
        if self.args.grouper_key in list(shared_blocks.shared_modules.keys()):
            # A batch with ids ["id1", "s3", "s3"] would become [0, 1, 1]
            grouper: Grouper = dict(shared_blocks.shared_modules.items())[self.args.grouper_key]
            supercase_indices = grouper.extract_ids_from_batch([x["group_id"] for x in batch["meta"]]).to(
                self.torch_device
            )

            # the targets need to be grouped as well, currently y is a (B,) tensor with class indices
            # For each unique supercase index, we need to find the corresponding class index
            y = grouper.group_targets(y, supercase_indices)
            censor = grouper.group_targets(torch.Tensor([x["censor"] for x in batch["meta"]]), supercase_indices)
        else:
            supercase_indices = None
            censor = torch.Tensor([x["censor"] for x in batch["meta"]])
            self._grouper_weights = None

        y_hat = shared_blocks.forward((x, supercase_indices), self.forward)

        if self.criterion.continuous:
            step_results: StepMetricDict = {
                "preds": y_hat.detach().cpu().numpy(),
                "targets": y.cpu().numpy(),
                "censor": censor.cpu().numpy(),
                "logits": y_hat.detach().cpu().numpy(),
                "hazard": y_hat.detach().cpu().numpy(),
            }
        else:
            hazards = torch.sigmoid(y_hat.view(-1, len(self.class_names)).cpu().detach())
            S = torch.cumprod(1 - hazards, dim=1)
            S_padded = torch.cat([torch.ones_like(censor.view(-1, 1)), S], 1)

            step_results: StepMetricDict = {  # type: ignore (.numpy() does not correctly indicate numpy array)
                "targets": y.cpu().numpy(),
                "logits": y_hat.detach().cpu().numpy(),
                "preds": torch.argmax(hazards.detach().cpu(), dim=1).numpy(),
                "hazard": hazards.detach().cpu().numpy(),
                "survival": S.detach().cpu().numpy(),
                "padded_hazards": S_padded.detach().cpu().numpy(),
                "censor": censor.flatten().cpu().numpy(),
            }
        if sum(censor) > 1 and self.criterion.continuous:
            batch_loss = self.criterion(
                y_pred=y_hat, y_true=y.view(-1, 1), censor=censor.view(-1, 1).to(self.torch_device)
            )
        elif not self.criterion.continuous:
            batch_loss = self.criterion(
                y_pred=y_hat.view(-1, len(self.class_names)),
                y_true=y.view(-1, 1),
                censor=censor.view(-1, 1).to(self.torch_device),
            )
        else:
            return None

        self.add_step_result(batch_loss.item(), step_results)

        live_vis = self._visualize_preds(
            x.detach().cpu(),
            step_results,
            (batch["meta"] if "meta" in batch else [{} for _ in range(batch["image"].shape[0])]),
            supercase_indices=supercase_indices if supercase_indices is not None else torch.Tensor([1]),
        )
        return batch_loss, live_vis

    def _visualize_preds(
        self, training_ims, step_metrics: Dict, metas: List[Dict], supercase_indices
    ) -> Dict[str, Any]:
        vis_n = min(self._takeout_vis_budget(), training_ims.size(0))

        if vis_n <= 0:
            return {}
        # Assume Normal Images
        if training_ims[0].shape[0] == 3:
            # Select one of the groups for visualization
            group_index = random.choice(list(set(supercase_indices.cpu().numpy())))
            vis_indices = torch.where(supercase_indices == group_index)[0].cpu()
            if self._grouper_weights is not None:
                vis_cases_weights = self._grouper_weights[vis_indices]
                rows = int(np.sqrt(len(vis_indices)))
                # cols = training_ims[vis_indices].shape[0] // rows
                grid_img = make_grid(training_ims[vis_indices], nrow=rows)
                # Put the weights into rows
                weight_chunks = []
                for vis_case in vis_cases_weights:
                    weight_chunks.append(",".join([f"{single_w:.2f}" for single_w in vis_case]))
                weight_rows = [f"[{w}]" for w in weight_chunks]
                weight_rows[-1] = weight_rows[-1] + " " * (len(weight_rows[0]) - len(weight_rows[-1]))
                weight_str = "\n".join(weight_rows)
                caption = f"""
                Group {group_index} with {vis_indices} subcases, group id: {metas[vis_indices[0]]["group_id"]}
                weights:
                {weight_str}
                hazard: {step_metrics["hazard"][group_index]}
                target: {step_metrics["targets"][group_index]}
                event: {step_metrics['censor'][group_index]} 
                {[metas[i] for i in vis_indices]=}
            """
            else:
                rows = int(np.sqrt(len(vis_indices)))
                grid_img = make_grid(training_ims[vis_indices], nrow=rows)
                caption = f"""
                Group {group_index} with {len(vis_indices)} subcases, group id: {metas[0]["group_id"]}
                logits:
                {step_metrics["logits"][group_index]}   
                Predicted Hazard: {step_metrics["pred_hazard"][group_index]}
                {[metas[i] for i in vis_indices]=}
                """
            wandb_img = build_wandb_image(
                im=grid_img,
                caption=caption,
            )
        # Assume NICs
        else:
            idx = np.random.choice(len(training_ims))
            nic = training_ims[idx]
            meta = metas[idx]
            fmap_index = random.randint(0, nic.shape[0] - 1)
            wandb_img = wandb.Image(nic[fmap_index], caption=f'{step_metrics["targets"]=}\n {step_metrics["preds"]=}')

        return {"preds": [wandb_img]}

    def log_epoch_metrics(self) -> Tuple[Dict[str, Any], str]:
        metrics = flatten_list_of_dicts(self._step_metrics)
        if not self.criterion.continuous:
            selected_metrics = ["c-index-disc"]
        else:
            selected_metrics = ["c-index"]
        _, print_str = super().log_epoch_metrics()
        log_dict = {}

        if "c-index" in selected_metrics:
            try:
                log_dict["c-index"] = concordance_index(
                    metrics["targets"].squeeze(), -metrics["preds"].squeeze(), metrics["censor"].squeeze()
                )
            except ZeroDivisionError:
                logging.info("Had no uncensored datapoints in epoch. Will yield 0.5")
                log_dict["c-index"] = 0.5
        if "c-index-disc" in selected_metrics:
            risk = -torch.sum(torch.cumprod(torch.from_numpy(metrics["hazard"]), dim=1), dim=1)
            log_dict["c-index"] = concordance_index(
                metrics["targets"].squeeze(), risk.squeeze(), metrics["censor"].squeeze()
            )

        return log_dict, print_str
