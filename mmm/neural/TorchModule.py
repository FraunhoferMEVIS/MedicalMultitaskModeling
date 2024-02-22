from abc import abstractmethod
from mmm.BaseModel import BaseModel
import torch.nn as nn


class TorchModule(BaseModel):
    """
    Baseclass for pydantic models which can be directly instantiated into a torch.nn.Module
    """

    @abstractmethod
    def build_instance(self, *args, **kwargs) -> nn.Module:
        raise NotImplementedError
