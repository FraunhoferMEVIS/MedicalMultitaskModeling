from enum import Enum
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmm.BaseModel import BaseModel
from mmm.mtl_modules.shared_blocks.Grouper import AttentionPoolingReducer, TransformerReducer
from mmm.neural.TorchModule import TorchModule


class AttentionPoolingConfig(TorchModule):
    num_heads: int = 8
    type: Literal["abmil", "transformer"] = "abmil"

    def build_instance(self, dim) -> nn.Module:
        if self.type == "abmil":
            return AttentionPoolingReducer(
                embedding_dim=dim,
                num_heads=self.num_heads,
            )
        elif self.type == "transformer":
            return TransformerReducer(
                embedding_dim=dim,
                num_heads=self.num_heads,
            )


class CombinedPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.m = nn.AdaptiveMaxPool2d(dim)
        self.a = nn.AdaptiveAvgPool2d(dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.a(input) + self.m(input)


class GlobalPooling(str, Enum):
    AveragePooling = "average"
    MaxPooling = "max"
    Combined = "combined"


pooling_converter = {
    GlobalPooling.AveragePooling: nn.AdaptiveAvgPool2d,
    GlobalPooling.MaxPooling: nn.AdaptiveMaxPool2d,
    GlobalPooling.Combined: CombinedPooling,
}


class GlobalPoolingConfig(TorchModule):
    pooling_type: GlobalPooling

    def build_instance(self, *args, **kwargs) -> nn.Module:
        return pooling_converter[self.pooling_type]((1, 1))
