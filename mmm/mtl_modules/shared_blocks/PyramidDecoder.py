from typing import Literal, List, Mapping, Sequence, Union

import torch

from .SharedBlock import SharedBlock
from mmm.neural.modules.smp_modules import (
    SMPUnetDecoder,
    SMPPyramidAttentionDecoder,
)
from mmm.neural.modules.smp_modules import SMPPyramidAttentionDecoderConfig


class PyramidDecoder(SharedBlock):
    """
    Wraps a decoder from SMP.
    """

    class Config(SharedBlock.Config):
        module_name: str = "decoder"
        model: Union[SMPUnetDecoder.Config, SMPPyramidAttentionDecoderConfig] = SMPUnetDecoder.Config()

    def __init__(self, args: Config, enc_out_channels: List[int], encoder_output_stride: int) -> None:
        super().__init__(args)
        self.enc_out_channels, self.encoder_output_stride = (
            enc_out_channels,
            encoder_output_stride,
        )
        self.args: PyramidDecoder.Config
        self.model: Union[SMPUnetDecoder, SMPPyramidAttentionDecoder] = self.args.model.build_instance(
            enc_out_channels, encoder_output_stride
        )
        self.make_mtl_compatible()

    def get_output_dim_per_pixel(self) -> int:
        return self.model.get_output_dim_per_pixel()

    def get_upsampling_factor(self) -> int:
        """
        Used by segmentation task to know how much interpolation is required to reconstruct a mask with the input's size
        """
        return self.model.get_upsampling_factor()

    def forward(self, feature_pyramid: List[torch.Tensor]) -> torch.Tensor:
        return self.model(feature_pyramid)

    def get_example_input(self):
        # Usually, the feature maps would have sizes corresponding to the encoder output strides.
        example_input = [torch.rand(1, self.enc_out_channels[0], 224, 224)] + [
            torch.rand(1, c, 224 // (2 ** (i + 2)), 224 // (2 ** (i + 2))).to(self.torch_device)
            for i, c in enumerate(self.enc_out_channels[1:])
        ]
        # assert example_input[-1].shape[3] == 224 // self.encoder_output_stride
        return example_input

    def get_input_names(self) -> Sequence[str]:
        return ["input"] + [f"pyramid_{i}" for i in range(len(self.enc_out_channels) - 1)]

    def get_output_names(self) -> Sequence[str]:
        return ["pixel_embedding"]

    def get_dynamic_axes(self) -> Mapping[str, Mapping[int, str]]:
        return {
            "input": {0: "batch_size"},
            "pixel_embedding": {0: "batch_size", 2: "height", 3: "width"},
            **{
                f"pyramid_{i}": {
                    0: "batch_size",
                    2: f"height_lvl{i}",
                    3: f"width_lvl{i}",
                }
                for i in range(len(self.enc_out_channels) - 1)
            },
        }
