from typing import Mapping, Tuple
from pydantic import Field

import torch
import torch.nn as nn

from mmm.mtl_modules.shared_blocks.SharedBlock import ModelInput, SharedBlock
from mmm.neural.activations import ActivationFn, ActivationFunctionConfig


class Squeezer(SharedBlock):
    """
    Takes a feature pyramid and squeezes it into a single feature map.
    """

    class Config(SharedBlock.Config):
        module_name: str = "squeezer"
        out_channels: int = Field(
            -1,
            description="""Number of output channels, -1 for number of input channels.
            In that case, no convolution is applied.
            """,
        )
        activation: ActivationFunctionConfig = ActivationFunctionConfig(fn_type=ActivationFn.GeLU)
        use_channel_attention: bool = False

    def __init__(self, args: Config, enc_out_channels: list[int], enc_strides: list[int]):
        super().__init__(args)
        self.args: Squeezer.Config
        self.enc_out_channels = enc_out_channels
        self.enc_strides = enc_strides

        # self.norm = nn.GroupNorm(num_groups=1, num_channels=self.get_hidden_dim())
        # Use batchnorm because it will be automatically converted into the correct type of norm
        self.norm = nn.BatchNorm2d(num_features=self.get_hidden_dim())
        self.conv = (
            nn.Conv2d(
                in_channels=enc_out_channels[-1],
                out_channels=self.get_hidden_dim(),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            if self.args.out_channels != -1
            else nn.Identity()
        )
        self.activation = args.activation.build_instance()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if self.args.use_channel_attention:
            self.m = nn.AdaptiveMaxPool2d((1, 1))
        self.make_mtl_compatible()

    def get_hidden_dim(self) -> int:
        return self.args.out_channels if self.args.out_channels > 0 else self.enc_out_channels[-1]

    def forward(self, feature_pyramid: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # The order is inspired by torchvision.ops.misc.ConvNormActivation
        # experimental setting to increase the influence between pyramidal and pooling tasks
        if self.args.use_channel_attention:
            atten = self.avg(feature_pyramid[-1]) + self.m(feature_pyramid[-1])
            feat_latent = self.norm(self.conv(atten))
            feat = self.norm(self.conv(feature_pyramid[-1]))
            return self.activation(feat + feat_latent), self.activation(feat_latent)
        else:
            return self.activation(self.norm(self.conv(feature_pyramid[-1]))), self.activation(
                self.norm(self.conv(self.avg(feature_pyramid[-1])))
            )

    def get_example_input(self) -> ModelInput | Tuple[ModelInput, ...]:
        return [torch.rand(1, self.enc_out_channels[-1], 7, 7).to(self.torch_device)]

    def get_dynamic_axes(self) -> Mapping[str, Mapping[int, str]]:
        return {"input": {0: "batch_size", 2: "height", 3: "width"}}
