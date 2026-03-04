from typing import Union

from .activations import ActivationFn, ActivationFunctionConfig
from .losses import (
    CrossEntropyLossConfig,
    FocalLoss,
    FocalLossConfig,
    KLDivLossConfig,
    MSELossConfig,
    NLLLossConfig,
    RMSELossConfig,
    SurvivalLossConfig,
)
from .modules import *
from .TorchModule import TorchModule

LossConfigs = Union[
    CrossEntropyLossConfig,
    MSELossConfig,
    RMSELossConfig,
    FocalLossConfig,
    NLLLossConfig,
    SurvivalLossConfig,
    KLDivLossConfig,
]
