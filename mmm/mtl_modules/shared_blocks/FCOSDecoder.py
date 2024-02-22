import logging
from typing import List, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn

from .SharedBlock import SharedBlock

try:
    from mmcv.cnn import ConvModule
    from mmdet.models.necks.fpn import FPN
except ImportError:
    ConvModule, FPN = None, None


class FCOSDecoder(SharedBlock):
    """
    FPN which converts encoder features into the features that the mmDetection task expects.
    """

    class Config(SharedBlock.Config):
        module_name: str = "fcosfpn"
        intermediate_channels: int = 128
        start_level: int = 1
        num_outputs: int = 5
        reg_convs: int = 4
        clf_convs: int = 4
        dcn_on_last_conv: bool = False

    def __init__(self, args: Config, enc_out_channels: List[int], enc_strides: List[int]) -> None:
        """
        Expects the full encoder channels (including channels of the input) and the full encoder strides.

        It filters them afterwards according to the config.
        """
        super().__init__(args)
        self.args: FCOSDecoder.Config
        # The first component describes the input and can be omitted here
        self.full_enc_channels, self.full_enc_strides = enc_out_channels, enc_strides
        self.enc_out_channels = enc_out_channels[1:]
        self.strides_after_start = enc_strides[self.args.start_level + 1 :]

        self.fpn: FPN = FPN(
            in_channels=self.enc_out_channels,  # example: [256, 512, 1024, 2048],
            out_channels=args.intermediate_channels,
            start_level=self.args.start_level,
            add_extra_convs="on_output",  # use P5
            num_outs=self.args.num_outputs,
            # norm_cfg=dict(type='BN'),
            # act_cfg=dict(type='GELU'),
            relu_before_extra_convs=True,
        )
        # self.reg_fpn: FPN = FPN(
        #     in_channels=self.enc_out_channels,  # example: [256, 512, 1024, 2048],
        #     out_channels=args.intermediate_channels,
        #     start_level=self.args.start_level,
        #     add_extra_convs='on_output',  # use P5
        #     num_outs=self.args.num_outputs,
        #     act_cfg=dict(type='ReLU'),
        #     relu_before_extra_convs=True)
        self.clf_convs = self.build_convstack(args.clf_convs, args.dcn_on_last_conv)
        self.reg_convs = self.build_convstack(args.reg_convs, args.dcn_on_last_conv)
        self.make_mtl_compatible()

    def get_strides(self) -> List[int]:
        # return e.g. [8, 16, 32, 64, 128]
        return self.strides_after_start + [
            self.strides_after_start[-1] * (i + 1) * 2
            for i in range(self.args.num_outputs - len(self.strides_after_start))
        ]

    def build_convstack(self, stackdepth: int, dcn_on_last_conv: bool) -> nn.Sequential:
        """Initialize bbox regression conv layers of the head."""
        res = nn.ModuleList()
        for i in range(stackdepth):
            # chn = self.in_channels if i == 0 else self.args.intermediate_channels
            if dcn_on_last_conv and i == stackdepth - 1:
                conv_cfg = dict(type="DCNv2")
            else:
                conv_cfg = None
            res.append(
                ConvModule(
                    self.args.intermediate_channels,
                    self.args.intermediate_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                    act_cfg=dict(type="GELU"),
                    bias=True,
                )
            )
        return nn.Sequential(*res)

    def forward(self, feature_pyramid: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Skip the first element, which is the input
        fpn_features = self.fpn(feature_pyramid[1:])
        # fpn_reg_features = self.reg_fpn(feature_pyramid[1:])
        clf_features = [self.clf_convs(x) for x in fpn_features]
        reg_features = [self.reg_convs(x) for x in fpn_features]

        return clf_features, reg_features

    def get_example_input(self):
        # using self.full_enc_channels, self.full_enc_strides
        example_input = [
            torch.rand(1, c, 224 // s, 224 // s).to(self.torch_device)
            for c, s in zip(self.full_enc_channels, self.full_enc_strides)
        ]
        return example_input
