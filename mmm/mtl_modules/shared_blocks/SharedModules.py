import torch.nn as nn

from .SharedBlock import SharedBlock


class SharedModules(nn.Module):
    """
    Intended to be wrapped by DDP to avoid a varying number of forward passes,
    leading to blocking.
    """

    def __init__(self, shared_blocks: dict[str, SharedBlock]):
        super().__init__()
        self.shared_modules = nn.ModuleDict(shared_blocks)

    @property
    def module(self):
        return self

    def forward(self, x, f):
        return f(x, self.shared_modules)
