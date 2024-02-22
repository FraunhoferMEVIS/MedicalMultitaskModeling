from enum import Enum
from typing import Callable
import torch.nn as nn

from mmm.neural.TorchModule import TorchModule


class ActivationFn(str, Enum):
    Identity = "identity"
    Sigmoid = "sigmoid"
    LogSigmoid = "logsigmoid"
    Tanh = "tanh"
    ReLU = "relu"
    LeakyReLU = "leakyrelu"
    GeLU = "gelu"


fn_converter = {
    ActivationFn.Identity: nn.Identity,
    ActivationFn.LogSigmoid: nn.LogSigmoid,
    ActivationFn.Sigmoid: nn.Sigmoid,
    ActivationFn.Tanh: nn.Tanh,
    ActivationFn.ReLU: nn.ReLU,
    ActivationFn.LeakyReLU: nn.LeakyReLU,
    ActivationFn.GeLU: nn.GELU,
}


class ActivationFunctionConfig(TorchModule):
    fn_type: ActivationFn

    def build_instance(self, *args, **kwargs) -> nn.Module:
        return fn_converter[self.fn_type]()
