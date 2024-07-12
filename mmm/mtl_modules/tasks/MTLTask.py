from __future__ import annotations
from pydantic import Field
from typing import Any, List, Dict, Tuple, Optional
from abc import abstractmethod

import torch
import torch.nn as nn

from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.data_loading.MTLDataset import MTLDataset

from ..MTLModule import MTLModule
from mmm.mtl_modules.MTLModule import MTLModule
from mmm.event_selectors import EventSelector, RecurringEventSelector
from mmm.logging.type_ext import StepFeedbackDict, StepMetricDict
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules


class MTLTask(MTLModule):
    """
    Task which makes use of shared blocks during its forward pass.

    Guidelines when developing your own task types:

    Implementations need to extend/override abstractmethods such as `forward` and `training_step`.

    Giving feedback can happen after each step using the `training_step` implementation
    and after each loop/epoch/... by overwriting `log_epoch_metrics`.

    Loss is expected to be approx. 1 given random input. This makes sure your task type is not dominated nor dominates
    the framework's task types.
    """

    class Config(MTLModule.Config):
        vis_enabled_epochs: EventSelector = Field(
            default=RecurringEventSelector(starting_at=0, every_n=1),
            description="Controls the loops with enabled visualization.",
        )
        max_visualizations_per_full_train_loop: int = 2
        max_visualizations_per_full_val_loop: int = 4

    def __init__(self, args: Config, cohort: TrainValCohort[MTLDataset]) -> None:
        super().__init__(args)
        self.args: MTLTask.Config = args
        self.cohort: TrainValCohort[MTLDataset] = cohort
        self.task_modules: nn.ModuleDict

        self._step_losses: List[float] = []
        self._step_metrics: List[StepMetricDict] = []

        self._vis_number_buffer: float = 0.0

    def _takeout_vis_budget(self) -> int:
        """
        Returns the task's visualization budget at the current step.

        Relies on the module's `training: bool` property.

        Tries to induce the total number of steps from the dataloader's length if the dataset is map-style.
        """
        # Adjust the vis_enabled_epochs to enable visualizations for an epoch
        if not self.args.vis_enabled_epochs.is_event(self._epoch):
            return 0

        # current_step = len(self._step_losses)
        # dataloader_length = len(self.cohort.data_loaders[0 if self.training else 1])  # type: ignore
        if self.training:
            max_vis = self.args.max_visualizations_per_full_train_loop
        else:
            max_vis = self.args.max_visualizations_per_full_val_loop

        self._vis_number_buffer += 1
        return 1 if self._vis_number_buffer < max_vis else 0

    def prepare_batch(self, batch: Dict[str, Any]) -> Any:
        """
        Prepares the batch for training on the correct device.

        The trainer calls this funci

        By default, it will be attempted to copy the whole data structure to GPU
        """
        raise NotImplementedError

    def add_step_result(self, loss: float, step_metrics: StepMetricDict):
        self._step_losses.append(loss)
        self._step_metrics.append(step_metrics)

    def get_step_losses(self):
        return self._step_losses

    def is_currently_being_trained(self) -> bool:
        according_to_loss = len(self._step_losses) != 0
        according_to_metrics = len(self._step_metrics) != 0
        assert according_to_loss == according_to_metrics, f"Missing logging of metrics for MTLModule {self.get_name()}?"
        return according_to_loss

    def get_short_status(self) -> str:
        if self.is_currently_being_trained():
            avg_loss = sum(self.get_step_losses()) / len(self.get_step_losses())
            return f"{avg_loss:.3f}"
        else:
            return "-"

    @abstractmethod
    def forward(self, x: Any, shared_blocks: SharedModules):
        raise NotImplementedError

    @abstractmethod
    def training_step(
        self, batch: Dict[str, Any], shared_blocks: SharedModules
    ) -> Tuple[torch.Tensor, StepFeedbackDict]:
        """
        Returns the loss (single number as torch.FloatTensor) and a dictionary with immediate feedback for a human.

        Here, a task should store metrics for later aggregation using `self.add_step_result(...)`.
        """
        raise NotImplementedError()

    def reset_metric_buffers(self):
        self._step_losses = []
        self._step_metrics = []

    def prepare_epoch(self, epoch: int, prefix: str, training_mode: Optional[bool] = None):
        """
        Called before starting an epoch. Used to reset buffer and prepare dataloading for cross validation.
        """
        super().prepare_epoch(epoch, prefix, training_mode=training_mode)
        self.reset_metric_buffers()
        self._vis_number_buffer = 0

    def log_epoch_metrics(self) -> Tuple[Dict[str, Any], str]:
        """
        Called when the user of the task wants the task to log its metric buffers.

        Happens at the end of a loop and can be an expensive, infrequent operation.

        .. code-block::
            :caption: Example dictionary from the metrics buffer

            {
                "acc": 0.96,
                "auc": 0.76
            }

        The string is used for printing and should be one line.
        """
        return {}, f"{self.get_name()} - loss: {self.get_short_status()}"

    def __repr__(self) -> str:
        res = super().__repr__()
        res = f"{res}\nconfig={self.args.model_dump_json(indent=2)}\n{self.cohort}"
        return res

    def __repr_html__(self) -> str:
        return f"""{super().__repr_html__()}
        <br />
        {self.cohort.__repr_html__()}
        """
