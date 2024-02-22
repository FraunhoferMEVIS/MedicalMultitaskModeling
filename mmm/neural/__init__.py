from typing import Union
from .TorchModule import TorchModule
from .modules import *
from .losses import (
    CrossEntropyLossConfig,
    MSELossConfig,
    RMSELossConfig,
    FocalLossConfig,
    FocalLoss,
    NLLLossConfig,
)
from .activations import ActivationFunctionConfig, ActivationFn

LossConfigs = Union[
    CrossEntropyLossConfig,
    MSELossConfig,
    RMSELossConfig,
    FocalLossConfig,
    NLLLossConfig,
]
