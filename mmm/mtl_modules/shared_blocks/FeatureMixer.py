from typing import Mapping, Tuple, Union

import torch
import torch.nn as nn

from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.Squeezer import Squeezer
from mmm.neural.modules.DenseFeatureMixer import DenseFeatureMixer
from mmm.neural.pooling import AttentionPoolingConfig, GlobalPoolingConfig

PoolingConfigs = Union[AttentionPoolingConfig, GlobalPoolingConfig]


class FeatureMixer(SharedBlock):
    """
    Takes a feature pyramid and squeezes it into a single feature map.
    """

    class Config(SharedBlock.Config):
        module_name: str = "mixer"

    def __init__(self, args: Config, for_encoder: PyramidEncoder, for_squeezer: Squeezer):
        super().__init__(args)
        self.args: FeatureMixer.Config

        channels = for_encoder.get_feature_pyramid_channels()[1:]
        channels[-1] = for_squeezer.get_hidden_dim()
        self.enc_mixers = nn.ModuleList(
            [
                DenseFeatureMixer(latent_dim=for_squeezer.get_hidden_dim(), pixel_feature_dim=channels[i])
                for i in range(len(channels))
            ]
        )

        self.make_mtl_compatible()

    def forward(
        self, semantic_features: torch.Tensor, feature_pyramid: list[torch.Tensor], group_indices: torch.Tensor
    ) -> list[torch.Tensor]:
        for i in range(1, len(feature_pyramid)):  # Start at second feature map, because first is the input
            feature_map = feature_pyramid[i]
            mixer = self.enc_mixers[i - 1]
            feature_pyramid[i] = mixer(semantic_features, feature_map, group_indices)
        return feature_pyramid
