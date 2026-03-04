from typing import List, Mapping, Sequence, Union

import torch

from mmm.neural.modules.smp_modules import SegFormer, SMPUnetDecoder

from .SharedBlock import SharedBlock


class PyramidDecoder(SharedBlock):
    """
    Wraps a decoder from SMP.
    """

    class Config(SharedBlock.Config):
        module_name: str = "decoder"
        model: Union[SMPUnetDecoder.Config, SegFormer.Config] = SMPUnetDecoder.Config()

    def __init__(self, args: Config, enc_out_channels: List[int], encoder_output_strides: list[int]) -> None:
        super().__init__(args)
        self.enc_out_channels, self.encoder_output_stride, self.encoder_output_strides = (
            enc_out_channels,
            encoder_output_strides[-1],
            encoder_output_strides,
        )
        self.args: PyramidDecoder.Config
        self.model: Union[SMPUnetDecoder, SegFormer] = self.args.model.build_instance(
            enc_out_channels, encoder_output_strides
        )
        self.make_mtl_compatible()

    def get_output_dim_per_pixel(self) -> int:
        return self.model.get_output_dim_per_pixel()

    def get_upsampling_factor(self) -> int:
        """
        Used by segmentation task to know how much interpolation is required to reconstruct a mask with the input's size
        """
        return self.model.get_upsampling_factor()

    def get_strides_fpn(self) -> List[int]:
        return self.model.get_strides_fpn()

    def forward_fpn(self, feature_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass that returns feature maps with identical channel numbers.
        """
        return self.model.forward_fpn(feature_pyramid)

    def forward(self, feature_pyramid: List[torch.Tensor]) -> torch.Tensor:
        return self.model(feature_pyramid)

    def get_example_input(self):
        # Usually, the feature maps would have sizes corresponding to the encoder output strides.
        example_input = [torch.rand(1, self.enc_out_channels[0], 224, 224)] + [
            torch.rand(1, c, 224 // (2 ** (i + 2)), 224 // (2 ** (i + 2))).to(self.torch_device)
            for i, c in enumerate(self.enc_out_channels[1:])
        ]
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
