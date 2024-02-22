from typing import List, Tuple, Literal
import torch
import torch.nn as nn

from ..TorchModule import TorchModule
from ..model_protocols import EncoderModel
from mmm.BaseModel import BaseModel
from mmm.neural.activations import ActivationFn, ActivationFunctionConfig
from mmm.torch_ext import infer_stride_channels_from_features


class ConvNormActivation(nn.Module):
    """
    Order of operations is inspired by torchvision ResNet implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        activation: ActivationFunctionConfig,
        conv_stride: int,
        padding=0,
        kernel_size=3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=padding,
        )

        if norm:
            # Always preliminary define batch norm because shared blocks replace it with the configured norm.
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()

        self.activation = activation.build_instance()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class MiniConvNet(nn.Module, EncoderModel):
    """
    Builds a feature pyramid by applying sequential convolution+activation blocks.

    Extracts the hidden vector using an Adaptivepoolingoperation and a Linear(deepest_depth, hidden_dim) layer.
    """

    class Config(BaseModel):
        architecture: Literal["miniconvnet"] = "miniconvnet"
        num_channels: int = 3
        num_filters: int = 32
        conv_stride: int = 2
        depth: int = 4
        norm: bool = True
        padding: int = 1
        activation: ActivationFunctionConfig = ActivationFunctionConfig(fn_type=ActivationFn.GeLU)

        def build_instance(self, *args, **kwargs):
            return MiniConvNet(self)

    def __init__(self, args: Config) -> None:
        super().__init__()
        self.args = args

        # self.conv1 = ConvNormActivation(
        #     self.args.num_channels,
        #     self.args.num_filters,
        #     args.norm,
        #     args.activation,
        #     conv_stride=2,
        #     kernel_size=3,
        #     padding=1,
        # )

        hidden_convs = [
            ConvNormActivation(
                self.args.num_filters * (i + 1) if i >= 0 else self.args.num_channels,
                self.args.num_filters * (i + 2),
                args.norm,
                args.activation,
                self.args.conv_stride,
                padding=self.args.padding,
            )
            for i in range(-1, self.args.depth - 1)
        ]
        self.cnn_layers = nn.ModuleList(hidden_convs)
        # self.out_channels: List[int] = [self.args.num_channels, self.args.num_filters] + [
        #     c.conv.out_channels for c in hidden_convs
        # ]

        with torch.no_grad():
            self.out_channels, self.strides = infer_stride_channels_from_features(
                self.forward(torch.rand(1, self.args.num_channels, 224, 224))
            )

    def get_feature_pyramid_channels(self) -> List[int]:
        return self.out_channels

    def get_strides(self) -> List[int]:
        return self.strides
        # return [1] + [4, 8, 16, 32]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]
        for conv_layer in self.cnn_layers:
            features.append(conv_layer(features[-1]))
        return features
