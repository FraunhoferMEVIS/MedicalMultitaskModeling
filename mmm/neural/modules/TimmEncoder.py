from typing import List, Literal
from pydantic import Field
import torch
import timm
from timm.layers.norm import LayerNorm2d
import torch.nn as nn

from mmm.torch_ext import replace_childen_recursive
from ..TorchModule import TorchModule
from ..model_protocols import EncoderModel


def replace_timmlayernorm2d_with_defaultcnnnorm(old_norm: LayerNorm2d):
    return nn.BatchNorm2d(num_features=old_norm.normalized_shape[0], eps=old_norm.eps)


valid_variants = [
    "convnext_femto",
    "convnext_pico",
    "convnext_nano",
    "convnext_tiny",
    "convnext_small",
    "convnext_large",
    "convnext_xlarge_384_in22ft1k",
]


class TimmEncoder(nn.Module, EncoderModel):
    """
    Valid variants have pretrained weights, can yield feature maps and have the correct stride.
    """

    class Config(TorchModule):
        architecture: Literal["timm"] = "timm"
        variant: Literal[tuple(valid_variants)] = "convnext_nano"  # type: ignore
        pretrained: bool = Field(default=False, description="Loads ImageNet1k weights")
        drop_rate: float = 0.0
        num_channels: int = 3

        def build_instance(self, *args, **kwargs) -> nn.Module:
            # Cannot be imported earlier due to CyclicImportError
            from mmm.neural.modules.TimmEncoder import TimmEncoder

            return TimmEncoder(self)

    def __init__(self, args: Config) -> None:
        super().__init__()
        self.args = args
        self.timm_model = timm.create_model(
            args.variant,
            pretrained=args.pretrained,
            drop_rate=args.drop_rate,
            features_only=True,
            in_chans=args.num_channels,
            # out_indices=(2, 3, 4)
            # drop_path_rate=0.2
            # global_pool=args.pooling
        )

        # Timm uses a hack which makes our automatic replacements fail
        replace_childen_recursive(self.timm_model, LayerNorm2d, replace_timmlayernorm2d_with_defaultcnnnorm)

    def get_feature_pyramid_channels(self) -> List[int]:
        return [self.args.num_channels] + [d["num_chs"] for d in self.timm_model.feature_info]

    def get_strides(self) -> List[int]:
        return [1] + self.timm_model.feature_info.reduction()

    def forward(self, input: torch.Tensor):
        feature_pyramid = [input] + self.timm_model.forward(input)
        return feature_pyramid
