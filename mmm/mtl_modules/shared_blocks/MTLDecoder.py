from typing import List
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder


class SCSEModule(nn.Module):
    """
    Adapted from SMP
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Attention(nn.Module):
    """
    Adapted from SMP
    """

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class Conv2dReLU(nn.Sequential):
    """
    Adapted from SMP
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        stride=1,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.ReLU(inplace=True)

        # Batchnorm2d needs to be used here, the Sharedblock class will replace by the correct norm
        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    """
    Adapted from SMP.

    If attention type is None, the attention module is just an identity.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        attention_type=None,
        third_conv=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)
        if third_conv:
            self.conv3 = Conv2dReLU(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, x, skip=None):
        """
        skip is the large feature map that x will be concatenated to.
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention2(x)
        return x


class LatentBlock(nn.Module):
    def __init__(self, hidden_dim, out_chan, outsize) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, out_chan, outsize, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(True),
        )

    def forward(self, z):
        z = self.map(z)
        z = self.up(z.reshape(-1, self.hidden_dim, 1, 1))
        return z


class ResnetBlock(nn.Module):
    """
    Adapted from Taming transformers Repo
    https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/diffusionmodules/model.py
    """

    def __init__(self, *, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = h * torch.sigmoid(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = h * torch.sigmoid(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """
    Adapted from Taming transformers Repo
    https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/diffusionmodules/model.py
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class MTLDecoder(SharedBlock):
    """
    Builds two-dimensional data-domain samples given
    """

    class Config(SharedBlock.Config):
        module_name: str = "mtldecoder"
        pixel_embedding_dim: int = 16
        out_channels: int = 3
        latent_upsamp_size: int = 8
        use_pyramid_channels: List[bool] = [True, False, False, False]
        include_latent_representation: bool = False
        num_residual_blocks: int = 3

    def __init__(
        self,
        args: Config,
        enc_out_channels: List[int],
        enc_strides: List[int],
        enc_hiddendim: int,
    ):
        super().__init__(args)
        self.args: MTLDecoder.Config
        # assert enc_strides[-1] == 32
        # assert enc_strides[1:] == [2**(i+2) for i in range(len(enc_out_channels[1:]))]
        # assert len(args.use_pyramid_channels) == len(enc_out_channels) - 1, "Wrong number of masked"

        # remove first skip with same spatial resolution
        decoder_channels = [(2**i) * args.pixel_embedding_dim for i in reversed(range(len(enc_out_channels) - 1))]
        encoder_channels = enc_out_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = [x if args.use_pyramid_channels[i] else 0 for i, x in enumerate(encoder_channels[::-1])]

        # computing blocks input and output channels
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        if self.args.include_latent_representation:
            in_channels[0] += enc_hiddendim
        skip_channels = [c for c in encoder_channels[1:]] + [0]
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, decoder_channels)
        ]
        if self.args.include_latent_representation:
            self.upsamp = LatentBlock(enc_hiddendim, enc_hiddendim, self.args.latent_upsamp_size)

        blocks[0] = DecoderBlock(in_channels[0], skip_channels[0], decoder_channels[0], attention_type="scse")
        blocks.append(DecoderBlock(args.pixel_embedding_dim, 0, args.pixel_embedding_dim))
        # for b in blocks:
        #     [c.apply(weights_init) for c in b.children() if isinstance(c, nn.Conv2d)]
        self.blocks = nn.ModuleList(blocks)

        if self.args.num_residual_blocks > 0:
            resblocks = []
            # in-conv
            resblocks.append(nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, padding=1, stride=1))
            for _ in range(self.args.num_residual_blocks):
                resblocks.append(ResnetBlock(in_channels=in_channels[0], out_channels=in_channels[0]))
                resblocks.append(AttnBlock(in_channels=in_channels[0]))

            # norm out
            resblocks.append(nn.GroupNorm(num_groups=32, num_channels=in_channels[0], eps=1e-6, affine=True))
            # out-conv
            resblocks.append(nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, padding=1, stride=1))
            self.residuals = nn.ModuleList(resblocks)

        self.make_mtl_compatible()
        # self.apply(weights_init)

    def get_output_dim_per_pixel(self) -> int:
        return self.args.pixel_embedding_dim

    def get_upsampling_factor(self) -> int:
        return 1

    def forward(self, features: List[torch.Tensor], z: torch.Tensor):
        """
        Computes the pixel embedding. Useful for the forward pass of segmentation tasks.
        """
        # remove first skip with same spatial resolution (which is the original input)
        features = features[1:]
        # reverse channels to start from head of encoder ()
        features = features[::-1]

        skips = features[1:]

        # Build initial state of the upsampling routine
        if self.args.include_latent_representation:
            z_featurized = self.upsamp(z)
            z_featurized = F.interpolate(z_featurized, size=features[0].shape[2:], mode="nearest")
            x = z_featurized
            if self.args.use_pyramid_channels[0]:
                x = torch.concat([features[0], z_featurized], dim=1)
        else:
            x = features[0]

        if self.args.num_residual_blocks > 0:
            for i, resblock in enumerate(self.residuals):
                x = resblock(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) and self.args.use_pyramid_channels[i + 1] else None
            x = decoder_block(x, skip)

        return x
