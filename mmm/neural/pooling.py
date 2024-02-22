from enum import Enum
from typing import Callable
import torch.nn as nn

from mmm.neural.TorchModule import TorchModule


class GlobalPooling(str, Enum):
    AveragePooling = "average"
    MaxPooling = "max"


pooling_converter = {
    GlobalPooling.AveragePooling: nn.AdaptiveAvgPool2d,
    GlobalPooling.MaxPooling: nn.AdaptiveMaxPool2d,
}


class GlobalPoolingConfig(TorchModule):
    pooling_type: GlobalPooling

    def build_instance(self, *args, **kwargs) -> nn.Module:
        return pooling_converter[self.pooling_type]((1, 1))
