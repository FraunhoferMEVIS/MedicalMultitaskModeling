from typing import Literal, List, Optional
from pydantic import Field
import torch
import torch.nn as nn
import torchvision.models as torch_models
from torchvision.models.convnext import (
    ConvNeXt,
    ConvNeXt_Tiny_Weights,
    LayerNorm2d,
    ConvNeXt_Small_Weights,
)

from mmm.torch_ext import infer_stride_channels_from_features
from mmm.neural.TorchModule import TorchModule
from mmm.mtl_modules.shared_blocks import SharedBlock
from mmm.neural.activations import ActivationFunctionConfig, ActivationFn
from mmm.torch_ext import replace_childen_recursive


def replace_with_defaultnorm(old_norm: LayerNorm2d):
    return nn.BatchNorm2d(num_features=old_norm.normalized_shape[0], eps=old_norm.eps)


WEIGHTSKB = {
    "tiny": (torch_models.convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
    "small": (torch_models.convnext_small, ConvNeXt_Small_Weights.DEFAULT),
}


class TorchVisionConvnext(nn.Module):
    """
    For MTL, replace all layernorms with layernorms without learnable parameters!
    """

    class Config(TorchModule):
        architecture: Literal["convnext"] = "convnext"
        pretrained: bool = True
        # use_latent_layer: bool = False
        # latent_activation: ActivationFunctionConfig = Field(
        #     default=ActivationFunctionConfig(fn_type=ActivationFn.ReLU),
        #     description="Only relevant when used with `use_latent_layer`."
        # )
        variant: Literal["tiny", "small"] = "tiny"

        def build_instance(self, hidden_dim: int) -> nn.Module:
            # Avoid cyclical import by importing here
            return TorchVisionConvnext(self, hidden_dim)

    def __init__(self, args: Config, hidden_dim: int) -> None:
        super().__init__()
        self.args = args

        if self.args.pretrained:
            self.wrapped_model: ConvNeXt = WEIGHTSKB[self.args.variant][0](weights=WEIGHTSKB[self.args.variant][1])
        else:
            self.wrapped_model: ConvNeXt = WEIGHTSKB[self.args.variant][0]()

        # Sharedblocks will later replace the default norm with a good norm
        replace_childen_recursive(self.wrapped_model, LayerNorm2d, replace_with_defaultnorm)

        # Delete the last fc layer, it is not required
        self.wrapped_model.classifier = self.wrapped_model.classifier[:2]

        with torch.no_grad():
            self.channels, self.strides = infer_stride_channels_from_features(
                self._feature_forward(torch.rand(1, 3, 224, 224))
            )
        # assert hidden_dim == self.get_feature_pyramid_channels()[-1]

    def _feature_forward(self, x: torch.Tensor):
        """
        Given the original input, returns a list with the input and all feature maps
        """
        activations = [x]
        for layer in self.wrapped_model.features:
            activations.append(layer(activations[-1]))

        return activations[::2]

    def forward(self, x: torch.Tensor):
        features = self._feature_forward(x)
        # Apply avg pool
        z = self.wrapped_model.avgpool(features[-1])
        z = torch.flatten(z, 1)
        return features, z

    def get_strides(self):
        # Input and spatial sizes of the feature maps
        # return [1] + [4, 8, 16, 32]
        return self.strides

    def get_feature_pyramid_channels(self) -> List[int]:
        # [3, 96, 192, 384, 768]
        return self.channels
