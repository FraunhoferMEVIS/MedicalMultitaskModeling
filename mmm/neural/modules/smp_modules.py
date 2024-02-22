"""
Modules wrapped from the segmentation models for PyTorch library
"""

from __future__ import annotations
import logging
from typing import Literal
import torch
import torch.nn as nn
from pydantic import Field

from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.pan.decoder import PANDecoder

from ..TorchModule import TorchModule
from ..activations import ActivationFunctionConfig, ActivationFn


class SMPUnetDecoder(nn.Module):
    """
    Wraps segmentation-models-pytorch's UnetDecoder

    >>> import torch
    >>> from mmm.neural.modules.smp_modules import SMPUnetDecoder
    >>> B, out_channels_for_pixel, pyr_depth, pyr_stride = 4, 16, (3, 64, 256, 512, 1024, 2048), (1, 2, 4, 8, 16, 32)
    >>> test_pyramid = [torch.rand((B, d, 224 // s, 224 // s)) for d, s in zip(pyr_depth, pyr_stride)]
    >>> dec = SMPUnetDecoder(SMPUnetDecoder.Config(pixel_embedding_dim=out_channels_for_pixel), pyr_depth, pyr_stride[-1])
    >>> dec(test_pyramid).shape
    torch.Size([4, 16, 224, 224])
    >>> assert True not in [c > dec.args.hidden_dim for c in dec.channels]  # use hidden_dim as the max number of channels
    """

    class Config(TorchModule):
        architecture: Literal["unetdecoder"] = "unetdecoder"
        pixel_embedding_dim: int = 16
        hidden_dim: int = Field(96, description="The maximum depth of the decoder. Avoids excessive memory usage.")
        head_kernel_size: int = 3
        activation: ActivationFunctionConfig = ActivationFunctionConfig(fn_type=ActivationFn.GeLU)

        def build_instance(self, *args, **kwargs) -> SMPUnetDecoder:
            return SMPUnetDecoder(self, enc_out_channels=args[0], encoder_output_stride=args[1])

    def __init__(self, args: Config, enc_out_channels: list[int], encoder_output_stride: int) -> None:
        super().__init__()
        self.args = args
        # Skip one channel because that is the raw input
        self.channels = [
            min(((2**i) * args.pixel_embedding_dim), args.hidden_dim)
            for i in reversed(range(len(enc_out_channels) - 1))
        ]
        self.final_conv = nn.Conv2d(
            min(args.pixel_embedding_dim, args.hidden_dim),
            self.args.pixel_embedding_dim,
            kernel_size=self.args.head_kernel_size,
            padding=self.args.head_kernel_size // 2,
        )
        self.norm = nn.BatchNorm2d(num_features=self.args.pixel_embedding_dim)
        self.activation = args.activation.build_instance()
        assert encoder_output_stride == 32
        logging.debug(f"Channels of {self.args.architecture} are {self.channels}")
        self.decoder = UnetDecoder(enc_out_channels, tuple(self.channels), n_blocks=len(self.channels))

    def get_upsampling_factor(self) -> int:
        return 1

    def get_output_dim_per_pixel(self) -> int:
        return self.args.pixel_embedding_dim

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        hidden_map: torch.Tensor = self.decoder(*features)
        return self.activation(self.norm(self.final_conv(hidden_map)))


class SMPPyramidAttentionDecoderConfig(TorchModule):
    architecture: Literal["pyramid_attention"] = "pyramid_attention"
    decoder_channels: int = 32

    def build_instance(self, *args, **kwargs) -> SMPPyramidAttentionDecoder:
        return SMPPyramidAttentionDecoder(self, enc_out_channels=args[0], encoder_output_stride=args[1])


class SMPPyramidAttentionDecoder(nn.Module):
    """
    Wraps segmentation-models-pytorch's PAN decoder
    """

    def __init__(
        self,
        args: SMPPyramidAttentionDecoderConfig,
        enc_out_channels: list[int],
        encoder_output_stride: int,
    ) -> None:
        super().__init__()
        self.args = args

        assert len(enc_out_channels) > 4, "By the time of writing, PAN uses the latest four levels"

        assert encoder_output_stride == 32
        self.decoder = PANDecoder(enc_out_channels, self.args.decoder_channels)

    def get_upsampling_factor(self) -> int:
        return 4

    def get_output_dim_per_pixel(self) -> int:
        return self.args.decoder_channels

    def forward(self, features: list[torch.Tensor]):
        return self.decoder(*features)
