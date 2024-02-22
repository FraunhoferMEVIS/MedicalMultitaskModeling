from functools import partial
from pydantic import Field
from typing import Dict, List, Literal, Optional
import torch
import torch.nn as nn
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
from torchvision.models.swin_transformer import (
    Swin_V2_T_Weights,
    Swin_V2_B_Weights,
    Swin_V2_S_Weights,
    SwinTransformer,
)
from ..TorchModule import TorchModule
from ..activations import ActivationFunctionConfig, ActivationFn
from mmm.torch_ext import infer_stride_channels_from_features

_models = {"tiny": swin_v2_t, "small": swin_v2_s, "base": swin_v2_b}

_weights = {
    "tiny": Swin_V2_T_Weights.DEFAULT,
    "small": Swin_V2_S_Weights.DEFAULT,
    "base": Swin_V2_B_Weights.DEFAULT,
}


class TorchVisionSwinformer(nn.Module):
    class Config(TorchModule):
        architecture: Literal["swinformer"] = "swinformer"
        pretrained: bool = False
        variant: Literal["tiny", "small", "base"] = "tiny"

        def build_instance(self, *args, **kwargs) -> nn.Module:
            return TorchVisionSwinformer(self)

    def __init__(self, args: Config) -> None:
        super().__init__()
        self.args = args
        # First load the full model to allow torchvision pretrained weights
        if args.pretrained:
            self.wrapped_model: SwinTransformer = _models[args.variant](weights=_weights[args.variant])  # type: ignore
        else:
            self.wrapped_model: SwinTransformer = _models[args.variant]()

        # We only need the features, not the head
        del self.wrapped_model.head

        with torch.no_grad():
            self.channels, self.strides = infer_stride_channels_from_features(
                self._feature_forward(torch.rand(1, 3, 224, 224))
            )
            # Currently strides are assumed to always be:
            assert self.strides == [1, 4, 8, 16, 32]

    def _feature_forward(self, x):
        feature_maps = [x]
        # self.wrapped_model.features are the layers which compute features in the swin transformer
        for l in self.wrapped_model.features:
            feature_maps.append(l(feature_maps[-1]))

        feature_maps = [x] + [feature_map.permute(0, 3, 1, 2) for feature_map in feature_maps[1:]]
        return feature_maps[::2]

    def get_strides(self):
        # Input and spatial sizes of the feature maps
        return self.strides

    def forward(self, x):
        pyr = self._feature_forward(x)

        return pyr

    def get_feature_pyramid_channels(self) -> List[int]:
        return self.channels
