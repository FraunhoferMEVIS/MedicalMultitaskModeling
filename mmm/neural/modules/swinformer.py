from typing import List, Literal

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torchvision.models import swin_v2_b, swin_v2_s, swin_v2_t
from torchvision.models.swin_transformer import (
    PatchMergingV2,
    Swin_V2_B_Weights,
    Swin_V2_S_Weights,
    Swin_V2_T_Weights,
    SwinTransformer,
    SwinTransformerBlockV2,
)

from mmm.torch_ext import infer_stride_channels_from_features

from ..TorchModule import TorchModule


def build_large_swinv2(dropout: float = 0.0):
    """
    According to SwinTransformerv2 paper:

    - tiny: C=96, block = {2, 2, 6, 2}
    - small: C=96, block = {2, 2, 18, 2}
    - base: C=128, block = {2, 2, 18, 2}
    - large: C=192, block = {2, 2, 18, 2}
    - Huge: C=352, block = {2, 2, 18, 2}
    - Giant: C=512, block = {2, 2, 42, 4}

    But there is no information on the number of attention heads for the huge and giant model.
    For this reason, only the large model is implemented.

    For an input size 512x512 we get:
    Total params: 195,442,164
    Trainable params: 195,442,164
    Non-trainable params: 0
    Total mult-adds (M): 286.14
    =========================================================================================================
    Input size (MB): 3.15
    Forward/backward pass size (MB): 1481.64
    Params size (MB): 528.03
    Estimated Total Size (MB): 2012.82
    """
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=[16, 16],
        mlp_ratio=4.0,
        dropout=dropout,
        attention_dropout=dropout,
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
    )


_models = {"tiny": swin_v2_t, "small": swin_v2_s, "base": swin_v2_b, "large": build_large_swinv2}

_weights = {
    "tiny": Swin_V2_T_Weights.DEFAULT,
    "small": Swin_V2_S_Weights.DEFAULT,
    "base": Swin_V2_B_Weights.DEFAULT,
}


class TorchVisionSwinformer(nn.Module):
    class Config(TorchModule):
        architecture: Literal["swinformer"] = "swinformer"
        pretrained: bool = False
        variant: Literal["tiny", "small", "base", "large"] = "tiny"
        use_grad_checkpointing: bool = False

        def build_instance(self, *args, **kwargs) -> nn.Module:
            return TorchVisionSwinformer(self)

    def __init__(self, args: Config) -> None:
        super().__init__()
        self.args = args
        # First load the full model to allow torchvision pretrained weights
        if args.pretrained:
            assert args.variant in _weights, f"Pretrained weights not available for variant {args.variant}"
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
            if hasattr(self.args, "use_grad_checkpointing") and self.args.use_grad_checkpointing:
                feature_maps.append(cp.checkpoint(l, feature_maps[-1], use_reentrant=False))
            else:
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
