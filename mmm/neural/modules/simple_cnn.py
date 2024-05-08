from typing import List, Tuple, Literal
import torch
import torch.nn as nn

from ..TorchModule import TorchModule
from ..model_protocols import EncoderModel
from mmm.BaseModel import BaseModel
from mmm.neural.activations import ActivationFn, ActivationFunctionConfig
from mmm.torch_ext import infer_stride_channels_from_features
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock


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

        with torch.no_grad():
            self.out_channels, self.strides = infer_stride_channels_from_features(
                self.forward(torch.rand(1, self.args.num_channels, 224, 224))
            )

    def get_feature_pyramid_channels(self) -> List[int]:
        return self.out_channels

    def get_strides(self) -> List[int]:
        return self.strides

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]
        for conv_layer in self.cnn_layers:
            features.append(conv_layer(features[-1]))
        return features


class MiniDecoder(SharedBlock):
    """
    Small decoder to be used as ImageGenerationDecoder during pre-training. Images will most likely look bad.
    Works on the lowest feature pyramid and doubles the size of the tensor with every level.

    >>> from mmm.neural.modules.simple_cnn import MiniDecoder
    >>> import torch
    >>> dec = MiniDecoder(MiniDecoder.Config(pixel_embedding=16, upsample_levels=2), [3, 32, 64, 128])
    >>> test = [torch.ones(2, 128, 5, 5)]
    >>> out = dec(test)
    >>> assert out.shape == torch.Size([2, 16, 20, 20])
    """

    class Config(SharedBlock.Config):
        module_name: str = "miniDecoder"
        pixel_embedding: int = 32
        activation: ActivationFunctionConfig = ActivationFunctionConfig(fn_type=ActivationFn.ReLU)
        upsample_levels: int = 4

    def __init__(
        self,
        args: Config,
        enc_out_channels: List[int],
    ):
        super().__init__(args)
        self.args: MiniDecoder.Config
        # blocks upsample such, that the output dim is about H//2, W//2 of the original input size
        hidden_convs = [
            [
                nn.ConvTranspose2d(
                    in_channels=(
                        enc_out_channels[-1] if i == args.upsample_levels else args.pixel_embedding * (i + 1) * 2
                    ),
                    out_channels=args.pixel_embedding * i * 2 if i > 1 else args.pixel_embedding,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                args.activation.build_instance(),
            ]
            for i in reversed(range(1, args.upsample_levels + 1))
        ]
        self.blocks = nn.ModuleList([item for sublist in hidden_convs for item in sublist])

        self.make_mtl_compatible()

    def get_output_dim_per_pixel(self) -> int:
        return self.args.pixel_embedding

    def get_upsampling_factor(self) -> int:
        return self.args.upsample_levels

    def forward(self, features: List[torch.Tensor]):
        feat = features[-1]
        for block in self.blocks:
            feat = block(feat)

        return feat
