import torch.nn as nn

from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock

from .MTLTask import MTLTask


class TaskModule(nn.Module):
    """
    Wraps a task along with its shared MTLModule dependencies
    to create a single neural network for other libraries to understand.

    You should make sure the shared modules are using this task (set_active_task) before executing the forward pass.
    """

    def __init__(self, task: MTLTask, shared_modules: dict[str, SharedBlock]) -> None:
        super().__init__()
        self.task = task
        self.shared_modules = nn.ModuleDict(shared_modules)

    def forward(self, x):
        result = self.task.forward(x, self.shared_modules)
        return result
