from __future__ import annotations
import json
from ast import Not
from typing import Any, List, Dict, Mapping, Tuple, Optional, Type, Literal
from typing_extensions import Annotated
import random
from PIL.Image import Image as PIL_Image
import logging

import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pydantic import Field

from mmm.logging.type_ext import StepMetricDict
from mmm.logging.wandb_ext import build_wandb_image_for_clf

from .MTLTask import MTLTask
from mmm.settings import mtl_settings
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.data_loading.ClassificationDataset import ClassificationDataset
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.Grouper import Grouper, make_grid_for_supercase
from mmm.neural import LossConfigs, CrossEntropyLossConfig
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.neural.pooling import GlobalPooling, GlobalPoolingConfig
from mmm.mtl_modules.shared_blocks.Grouper import Grouper

from mmm.utils import flatten_list_of_dicts
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    top_k_accuracy_score,
)


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
        grouper_key: str = Field(
            default="",
            description="If set, assumes a grouper to exist in the shared modules.",
        )
        loss_fn: Annotated[LossConfigs, Field(discriminator="loss_type")] = CrossEntropyLossConfig()
        dropout: float = 0.2
        metrics: Optional[List[Literal["confusion matrix", "accuracy", "top5accuracy", "auc", "f1"]]] = Field(
            default=None,
            description="If none, the task will decide which metrics make sense",
        )
        head: Literal["pretraining", "smart"] = "pretraining"

    @classmethod
    def from_torchvision_style(
        cls: Type[ClassificationTask],
        hidden_dim: int,
        class_names: List[str],
        task_config: Config,
        cohort_config: TrainValCohort.Config,
        train_ds: Dataset[Tuple[PIL_Image, torch.Tensor]],
        val_ds: Dataset[Tuple[PIL_Image, torch.Tensor]],
    ) -> ClassificationTask:
        """
        Automatically builds a task from torchvision default datasets.

        Torchvision datasets provide their training examples using tuples (PIL_Image, class_index: Integer-torch-tensor)
        """
        cohort = TrainValCohort(
            cohort_config,
            ClassificationDataset.from_torchvision(train_ds, class_names=class_names),
            # If no validation set is specified, the trainer will use cross-validation from the training set
            ClassificationDataset.from_torchvision(val_ds, class_names=class_names),  # if val_ds is not None else None
        )
        return cls(hidden_dim, class_names, task_config, cohort)

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

        if self.args.head == "pretraining":
            self.task_modules = self._create_pretraining_head()
        else:
            self.task_modules = self._create_smart_head()

        self._grouper_weights = None
        self.flatten = nn.Flatten(1)
        self.criterion: nn.Module = self.args.loss_fn.build_instance()

    def _create_pretraining_head(self):
        out_dim = len(self.class_names)
        new_dict = nn.ModuleDict(
            {
                "classification_head": nn.Sequential(
                    nn.Dropout(p=self.args.dropout),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, out_dim),
                )
            }
        )
        return new_dict

    def _create_smart_head(self) -> nn.ModuleDict:
        out_dim = len(self.class_names)
        new_dict = nn.ModuleDict(
            {
                "classification_head": nn.Sequential(
                    nn.Dropout(p=self.args.dropout),
                    nn.ReLU(),
                    nn.Linear(
                        max(out_dim * 4, self.hidden_dim),
                        max(out_dim * 4, self.hidden_dim // 2),
                    ),
                    nn.Dropout(p=self.args.dropout),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, out_dim),
                )
            }
        )
        return new_dict

    def prepare_batch(self, batch: Dict[str, Any]) -> Any:
        batch["image"] = batch["image"].to(self.torch_device)
        batch["class"] = batch["class"].to(self.torch_device)
        return batch

    def forward(self, inputs, shared_blocks: Dict[str, SharedBlock]):
        x, supercase_indexes = inputs
        pyr = shared_blocks[self.args.encoder_key](x)
        _, hidden_vector = shared_blocks[self.args.squeezer_key](pyr)
        hidden_vector = self.flatten(hidden_vector)

        if self.args.grouper_key:
            hidden_vector, self._grouper_weights = shared_blocks[self.args.grouper_key](
                hidden_vector, supercase_indexes
            )

        out = self.task_modules["classification_head"](hidden_vector)
        return out

    def training_step(self, batch: Dict[str, Any], shared_blocks: SharedModules):
        x = batch["image"]
        y = batch["class"]

        # skip if batch is empty
        tensornums = [bool(t.numel()) for t in x]
        if not True in tensornums:
            logging.info("encountered batch without valid training examples, skipping batch")
            return None

        # If a grouper is used, extract supercase_indices
        if self.args.grouper_key:
            # A batch with ids ["id1", "s3", "s3"] would become [0, 1, 1]
            grouper: Grouper = shared_blocks.module.shared_modules[self.args.grouper_key]
            supercase_indices = grouper.extract_ids_from_batch([x["group_id"] for x in batch["meta"]]).to(
                self.torch_device
            )

            # batch sizes might be unified here for some settings of the Grouper.
            # x = grouper.unify_bagsizes(bag=x, supercase_indices=supercase_indices)

            # the targets need to be grouped as well, currently y is a (B,) tensor with class indices
            # For each unique supercase index, we need to find the corresponding class index
            y = grouper.group_targets(y, supercase_indices)
        else:
            supercase_indices = None

        y_hat = shared_blocks.forward((x, supercase_indices), self.forward)
        batch_loss = self.criterion(y_hat, y) / np.log(len(self.class_names))

        step_results: StepMetricDict = {  # type: ignore (.numpy() does not correctly indicate numpy array)
            "targets": y.cpu().numpy(),
            "logits": y_hat.detach().cpu().numpy(),
            "preds": torch.argmax(y_hat.detach().cpu(), dim=1).numpy(),
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
        self, training_ims, step_metrics: dict, metas: list[dict], supercase_indices
    ) -> Dict[str, Any]:
        vis_n = min(self._takeout_vis_budget(), training_ims.size(0))

        if vis_n <= 0:
            return {}

        if supercase_indices is not None:
            # Select one of the groups for visualization
            group_index = random.choice(list(set(supercase_indices.cpu().numpy())))
            grid_img, weight_str, vis_indices = make_grid_for_supercase(
                training_ims, supercase_indices, group_index, self._grouper_weights
            )

            caption = f"""
Group {group_index} with {len(vis_indices)} subcases, group id: {metas[0]["group_id"]}
weights:
{weight_str}
logits:
{step_metrics["logits"][group_index]}
{[metas[i] for i in vis_indices]=}
            """
            wandb_img, description, true_str, pred_str = build_wandb_image_for_clf(
                grid_img,
                step_metrics["targets"][group_index],
                step_metrics["preds"][group_index],
                self.class_names,
                caption_suffix=caption,
            )
            return {"preds": [wandb_img]}
        else:
            preds = []
            for rand_index in random.sample(list(range(training_ims.size(0))), vis_n):
                metastr = json.dumps(metas[rand_index], default=lambda o: str(o))
                wandb_img, description, true_str, pred_str = build_wandb_image_for_clf(
                    training_ims[rand_index],
                    step_metrics["targets"][rand_index],
                    step_metrics["preds"][rand_index],
                    self.class_names,
                    caption_suffix=metastr,
                )
                preds.append(wandb_img)
            return {"preds": preds} if preds else {}

    def _get_short_class_names(self, max_length=10):
        if True in [len(c) > max_length for c in self.class_names]:
            return [f"{i};{c[:max_length]}" for i, c in enumerate(self.class_names)]
        else:
            return self.class_names

    def log_epoch_metrics(self) -> Tuple[Dict[str, Any], str]:
        metrics = flatten_list_of_dicts(self._step_metrics)
        if self.args.metrics is None:
            if len(self.class_names) <= mtl_settings.max_classes_detailed_logging:
                selected_metrics = ["confusion matrix", "accuracy", "auc", "f1"]
            else:
                selected_metrics = ["accuracy", "top5accuracy"]
        else:
            selected_metrics = self.args.metrics

        _, print_str = super().log_epoch_metrics()
        log_dict = {}

        if "accuracy" in selected_metrics:
            log_dict["acc"] = accuracy_score(y_true=metrics["targets"], y_pred=metrics["preds"])
            print_str = f"{print_str} - acc: {log_dict['acc']}"

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

        if "auc" in selected_metrics:
            # Binary classification
            if metrics["logits"].shape[1] == 2:
                preds = nn.Softmax(dim=1)(torch.from_numpy(metrics["logits"]))[:, 1]
            else:
                preds = nn.Softmax(dim=1)(torch.from_numpy(metrics["logits"]))

            try:
                log_dict["auc"] = roc_auc_score(metrics["targets"], preds, multi_class="ovr")

                print_str = f"{print_str} - auc: {log_dict['auc']}"
            except ValueError as e:
                logging.warn(f"Computing auc failed with {e}")

        if "confusion matrix" in selected_metrics:
            log_dict["confmat"] = wandb.plot.confusion_matrix(
                preds=metrics["preds"],  # type: ignore
                y_true=metrics["targets"],  # type: ignore
                class_names=self._get_short_class_names(),
                title=f"{self._prefix}_{self.get_name()}",
            )

        if "f1" in selected_metrics:
            log_dict["f1"] = f1_score(y_true=metrics["targets"], y_pred=metrics["preds"], average="macro")

        return log_dict, print_str
