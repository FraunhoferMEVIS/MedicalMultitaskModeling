import logging
from typing import Dict
import torch.nn as nn
import torch.distributed as dist

from .SharedBlock import SharedBlock


# class Dummy(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, shared_modules):
#         return x


class SharedModules(nn.Module):
    """
    Intended to be wrapped by DDP to avoid a varying number of forward passes,
    leading to blocking.
    """

    def __init__(self, shared_blocks: Dict[str, SharedBlock]):
        super().__init__()
        self.shared_modules = nn.ModuleDict(shared_blocks)

        # if not dist.is_initialized():
        #     # If we are not using DDP, make the shared modules available in the same way as DDP
        #     self.module = Dummy()
        #     self.module.shared_modules = self.shared_modules

    @property
    def module(self):
        return self

    def forward(self, x, f):
        # logging.info(f"!!!forward{f.__name__}!!!")
        return f(x, self.shared_modules)
