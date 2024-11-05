from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmm.BaseModel import BaseModel


from mmm.neural.TorchModule import TorchModule


class AttentionPooling(nn.Module):
    """
    Implements Self-attention pooling for feature-maps
    Intended Use is during NIC training in combinationation with a simple_cnn

    >>> pool = AttentionPooling(hidden_dim=512,out_dim=512)
    >>> test = torch.rand(5,512,7,7)
    >>> out = pool(test)
    >>> assert out.shape == torch.Size([5,512,1,1])
    """

    def __init__(
        self, hidden_dim: int, out_dim: int, num_heads: int = 1, gated: bool = False, pre_attention: bool = False
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.gated = gated
        self.pre_attention = pre_attention
        if self.gated:
            self.attention_v = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 3), nn.Tanh())
            self.attention_u = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 3), nn.Sigmoid())
            self.attention = nn.Linear(self.hidden_dim // 3, self.num_heads)
        else:
            self.attention = nn.Linear(self.hidden_dim, self.num_heads)

        self.mapper = nn.Sequential(
            nn.Dropout(0.0) if num_heads < 2 else nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * num_heads, self.hidden_dim if out_dim < 1 else out_dim),
        )
        if self.pre_attention:
            self.pre_attention_net = nn.TransformerEncoderLayer(self.hidden_dim, num_heads)

    def forward(self, x, return_weights=False):
        outs = []
        if return_weights:
            weights = []

        # Dealing with 2D and 1D input vectors
        if len(x.shape) == 4:
            bags = x.flatten(2, 3).permute(0, 2, 1)
        elif len(x.shape) == 3:
            bags = x.permute(0, 2, 1)

        for bag in bags:
            if self.pre_attention:
                bag = self.pre_attention_net(bag)

            if self.gated:
                att = self.attention(self.attention_v(bag) * self.attention_u(bag))
            else:
                att = self.attention(bag)
            att = torch.transpose(att, 1, 0)
            att_soft = F.softmax(att, dim=1)
            if return_weights:
                weights.append(att)

            outs.append(self.mapper(torch.matmul(att_soft, bag).view(1, -1)))
        if return_weights:
            return torch.stack(weights)
        else:
            return torch.stack(outs).view(-1, self.hidden_dim if self.out_dim < 1 else self.out_dim, 1, 1)


class AttentionPoolingConfig(TorchModule):
    num_heads: int = 1
    gated: bool = False
    pre_attention: bool = False

    def build_instance(self, in_dim, out_dim) -> nn.Module:
        return AttentionPooling(
            hidden_dim=in_dim,
            out_dim=out_dim,
            num_heads=self.num_heads,
            gated=self.gated,
            pre_attention=self.pre_attention,
        )


class CombinedPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.m = nn.AdaptiveMaxPool2d(dim)
        self.a = nn.AdaptiveAvgPool2d(dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.a(input) + self.m(input)


class GlobalPooling(str, Enum):
    AveragePooling = "average"
    MaxPooling = "max"
    Combined = "combined"


pooling_converter = {
    GlobalPooling.AveragePooling: nn.AdaptiveAvgPool2d,
    GlobalPooling.MaxPooling: nn.AdaptiveMaxPool2d,
    GlobalPooling.Combined: CombinedPooling,
}


class GlobalPoolingConfig(TorchModule):
    pooling_type: GlobalPooling

    def build_instance(self, *args, **kwargs) -> nn.Module:
        return pooling_converter[self.pooling_type]((1, 1))
