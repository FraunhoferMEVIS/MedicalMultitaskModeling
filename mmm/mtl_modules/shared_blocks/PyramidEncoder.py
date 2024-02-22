from pathlib import Path
from typing import List, Mapping, Sequence, cast, Union
from typing_extensions import Annotated

from pydantic import Field
import torch
import torch.nn as nn
import logging
import wandb

from mmm.neural.model_protocols import EncoderModel
from .SharedBlock import SharedBlock
from mmm.neural.activations import ActivationFn, ActivationFunctionConfig
from mmm.neural.pooling import GlobalPooling, GlobalPoolingConfig
from mmm.neural.modules.simple_cnn import MiniConvNet
from mmm.neural.modules.TorchVisionCNN import TorchVisionCNN
from mmm.neural.modules.swinformer import TorchVisionSwinformer
from mmm.neural.modules.TimmEncoder import TimmEncoder

EncoderArchitectureType = Union[
    MiniConvNet.Config,
    TorchVisionCNN.Config,
    TorchVisionSwinformer.Config,
    TimmEncoder.Config,
]


class PyramidEncoder(SharedBlock):
    class Config(SharedBlock.Config):
        model: Annotated[EncoderArchitectureType, Field(discriminator="architecture")] = TorchVisionCNN.Config()
        module_name: str = "encoder"
        pyramid_channels: dict[int, int] = Field(
            default={},
            description="Channels of the feature pyramid, can be left {} if the model's channels are already ok.",
        )

    def __init__(self, args: Config):
        super().__init__(args)
        self.args: PyramidEncoder.Config
        self.model: nn.Module = self.args.model.build_instance(self.args.pyramid_channels)

        self.remapper = nn.ModuleDict(
            {
                # Key of nn.ModuleDict cannot be an int
                f"{pyramid_index}": nn.Conv2d(
                    in_channels=self.model.get_feature_pyramid_channels()[pyramid_index],
                    out_channels=new_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for pyramid_index, new_channels in self.args.pyramid_channels.items()
            }
        )

        self.make_mtl_compatible()

    def get_feature_pyramid_channels(self) -> List[int]:
        """
        Returns the channels of the input and all channels of the feature pyramid.
        For input [B, 2, 224, 224] and feature maps [B, 32, 128, 128], [B, 64, 96, 96] it would return [2, 32, 64]
        """
        channels_ls: list[int] = cast(EncoderModel, self.model).get_feature_pyramid_channels()
        for overwrite_channel, model_channel in self.args.pyramid_channels.items():
            channels_ls[overwrite_channel] = model_channel
        return channels_ls

    def get_strides(self) -> List[int]:
        """
        Indicates how much smaller each featuremap is than the input.
        For a feature-pyramid of input [B, C_input, 64, 64]:
        [B, C_input, 64, 64]
        [B, C_l1, 32, 32]
        [B, C_l2, 16, 16]

        The strides would be [1, 2, 4]
        """
        return cast(EncoderModel, self.model).get_strides()

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        feature_pyramid = self.model(input)

        for pyramid_index in self.remapper.keys():
            remap_conv = self.remapper[pyramid_index]
            feature_pyramid[int(pyramid_index)] = remap_conv(feature_pyramid[int(pyramid_index)])

        return feature_pyramid

    def get_example_input(self):
        return torch.rand(1, self.get_feature_pyramid_channels()[0], 224, 224).to(self.torch_device)

    def get_input_names(self) -> Sequence[str]:
        return ["input"]

    def get_output_names(self) -> Sequence[str]:
        return ["input_identity"] + [
            f"pyramid_{i}" for i in range(len(self.get_strides()) - 1)
        ]  # + ["image_embedding"]

    def get_dynamic_axes(self) -> Mapping[str, Mapping[int, str]]:
        return {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "input_identity": {0: "batch_size", 2: "height", 3: "width"},
            # "image_embedding": {0: "batch_size"},
            **{
                f"pyramid_{i}": {
                    0: "batch_size",
                    2: f"height_{stride}" if stride != 1 else "height",
                    3: f"width_{stride}" if stride != 1 else "width",
                }
                for i, stride in enumerate(self.get_strides()[1:])
            },
        }

    def export_to_onnx(self, path: Path, for_task: str = "") -> None:
        if self.args.model.architecture == "swinformer":
            raise NotImplementedError("ONNX export is not supported for Swinformer")
        return super().export_to_onnx(path, for_task)
