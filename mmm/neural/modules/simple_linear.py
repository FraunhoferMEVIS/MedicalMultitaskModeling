from typing import List, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..TorchModule import TorchModule
from ..model_protocols import EncoderModel
from mmm.BaseModel import BaseModel
from mmm.neural.activations import ActivationFn, ActivationFunctionConfig
from mmm.torch_ext import infer_stride_channels_from_features
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock


class LinearNormActivation(nn.Module):
    """
    Order of operations is inspired by torchvision ResNet implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        activation: ActivationFunctionConfig,
    ) -> None:
        super().__init__()
        self.layer = nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
        )

        if norm:
            # Always preliminary define batch norm because shared blocks replace it with the configured norm.
            self.norm = nn.BatchNorm1d(num_features=out_channels)
        else:
            self.norm = nn.Identity()

        self.activation = activation.build_instance()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects the input to be of shape BxCxN and will return it in the same shape
        """
        return self.activation(self.norm(self.layer(x.permute(0, 2, 1)).permute(0, 2, 1)))


class SimpleLinearNet(nn.Module, EncoderModel):
    """
    Builds a feature pyramid by applying sequential linear+activation blocks.
    Will keep the in_features for its complete depth.
    """

    class Config(BaseModel):
        architecture: Literal["simplelinearnet"] = "simplelinearnet"
        in_features: int = 3
        out_features: int = 10
        depth: int = 1
        norm: bool = True
        activation: ActivationFunctionConfig = ActivationFunctionConfig(fn_type=ActivationFn.GeLU)

        def build_instance(self, *args, **kwargs):
            return SimpleLinearNet(self)

    def __init__(self, args: Config) -> None:
        super().__init__()
        self.args = args

        # if the  network is growing towards the end, we assume dilation
        dilated_network = args.in_features < args.out_features
        factor = (args.in_features - args.out_features) / (args.in_features * args.depth)

        hidden_layers = []
        current_dim = args.in_features
        for _ in range(args.depth - 1):
            if dilated_network:
                next_dim = min(int(current_dim * (1 - factor)), args.out_features)
            else:
                next_dim = int(current_dim * (1 - factor))
            hidden_layers.append(
                LinearNormActivation(
                    in_channels=current_dim,
                    out_channels=next_dim,
                    norm=args.norm,
                    activation=args.activation,
                )
            )
            current_dim = next_dim

        hidden_layers.append(
            LinearNormActivation(
                in_channels=current_dim,
                out_channels=args.out_features,
                norm=args.norm,
                activation=args.activation,
            )
        )
        self.linear_layers = nn.ModuleList(hidden_layers)

        with torch.no_grad():
            self.out_channels, self.strides = infer_stride_channels_from_features(
                self.forward(torch.rand(1, self.args.in_features, 224))
            )

    def get_feature_pyramid_channels(self) -> List[int]:
        return [self.args.out_features for _ in range(self.args.depth)]

    def get_strides(self) -> List[int]:
        return self.strides

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Expects input to be of Dimension BxCxN! Will return list of features of shape BxCxN
        """
        features = [x]

        for layer in self.linear_layers:
            features.append(layer(features[-1]))

        return features
