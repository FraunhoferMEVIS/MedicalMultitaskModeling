"""
Wrappers for models imported from torchvision model zoo
"""

from typing import List, Dict, Literal, Tuple
from pydantic import Field

import torch.nn as nn

import torch
import torch.nn as nn
import torchvision.models as torch_models
from torchvision.models._utils import IntermediateLayerGetter
import logging

from ..TorchModule import TorchModule
from mmm.torch_ext import infer_stride_channels_from_features


class FeatureBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, return_layers: Dict[str, str]) -> None:
        super().__init__()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        raw_dict = self.body(x)

        return list(raw_dict.values())


layername_KB = {
    "densenet121": {
        "constr": lambda w: torch_models.densenet121(weights=w).features,
        "layernames": ["denseblock1", "denseblock2", "denseblock3", "denseblock4"],
        "weights": torch_models.DenseNet121_Weights.IMAGENET1K_V1,
    },
    "resnet18": {
        "constr": lambda w: torch_models.resnet18(weights=w),
        "layernames": ["layer1", "layer2", "layer3", "layer4"],
        "weights": torch_models.ResNet18_Weights.DEFAULT,
    },
    "resnet50": {
        "constr": lambda w: torch_models.resnet50(weights=w),
        "layernames": ["layer1", "layer2", "layer3", "layer4"],
        "weights": torch_models.ResNet50_Weights.DEFAULT,
    },
    "efficientnet_v2_s": {
        "constr": lambda w: torch_models.efficientnet_v2_s(weights=w).features,
        "layernames": ["2", "3", "5", "7"],  # hand picked for strides 4, 8, 16, 32
        "weights": torch_models.EfficientNet_V2_S_Weights.DEFAULT,
    },
}


class TorchVisionCNN(nn.Module):
    class Config(TorchModule):
        architecture: Literal["residualcnn"] = "residualcnn"
        variant: Literal[tuple(layername_KB.keys())] = "densenet121"  # type: ignore
        pretrained: bool = Field(default=True, description="Will load ImageNet1k weights from TorchVision")

        def build_instance(self, *args, **kwargs) -> nn.Module:
            return TorchVisionCNN(self)

    def __init__(self, args: Config) -> None:
        super().__init__()
        self.args = args

        kb = layername_KB[self.args.variant]

        weights = kb["weights"] if self.args.pretrained else None

        self.backbone = FeatureBackbone(
            # torch_models.densenet121(weights=weights).features,
            kb["constr"](weights),
            # return_layers={v: v for v in ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]},
            return_layers={v: v for v in kb["layernames"]},
        )

        with torch.no_grad():
            self.channels, self.strides = infer_stride_channels_from_features(self(torch.rand(1, 3, 224, 224)))

    def get_feature_pyramid_channels(self) -> List[int]:
        # return [3] + self.backbone.in_channels_list
        return self.channels

    def get_strides(self):
        # Input and spatial sizes of the feature maps
        return self.strides

    def forward(self, x):
        pyr = self.backbone(x)
        return [x] + pyr
