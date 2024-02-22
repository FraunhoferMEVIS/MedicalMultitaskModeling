"""
Wraps an nn.ModuleDict that was created using `trainer.save_blocks_native(...)`.
"""

from pathlib import Path
import torch
from io import BytesIO
from mmm.BaseModel import BaseModel

from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.tasks.MTLTask import MTLTask


class NativeBlocks:
    def __init__(self, p: Path | BytesIO, device_identifier: str = "cuda") -> None:
        self.modules_path, self.device = p, device_identifier
        self.torch_modules = torch.load(self.modules_path).to(device_identifier)

    def get_sharedblock_keys(self) -> list[str]:
        return [k for k, v in self.torch_modules.items() if isinstance(v, SharedBlock)]

    def get_task_keys(self) -> list[str]:
        return [k for k, v in self.torch_modules.items() if isinstance(v, MTLTask)]

    def get_device(self) -> str:
        return self.device

    def __getitem__(self, key):
        return self.torch_modules[key]
