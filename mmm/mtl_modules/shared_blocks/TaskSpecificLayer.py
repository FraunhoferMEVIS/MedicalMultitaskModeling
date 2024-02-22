"""
This module is part of the shared blocks, because it implements knowledge about an active task.
As a consequence, it is not a generally reusable PyTorch nn.Module.
"""

from copy import deepcopy

import torch.nn as nn


class TaskSpecificLayer(nn.Module):
    """
    Converts a layer into a task specific layer.

    A layer for a new task is copied from the `self.original_layer`.

    If you do not call `create_layer_for_task`, the original layer is used for your forward pass.
    In this case, an exception is raised.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.task_modules: nn.ModuleDict = nn.ModuleDict({})
        self.active_task = ""
        self.original_layer = layer

    def create_layer_for_task(self, task_id: str) -> None:
        if task_id not in self.task_modules:
            self.task_modules[task_id] = deepcopy(self.original_layer)
        else:
            raise Exception(f"Task {task_id} already known to this task-specific layer")

    def set_active_task(self, task_id: str):
        self.active_task = task_id

    def forward(self, *args, **kwargs):
        if self.active_task in self.task_modules:
            return self.task_modules[self.active_task].forward(*args, **kwargs)
        else:
            if self.active_task != "original":
                raise Exception(
                    f"Using forward pass of original layer in {self}. "
                    + f"Because of unknown active task {self.active_task}"
                    + f"To suppress this warning set activate task to 'original'"
                )
            return self.original_layer(*args, **kwargs)
