import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DenseFeatureMixer(nn.Module):
    """
    Uses semantic features (1D vectors per image) to predict how the corresponding dense features should be mixed.

    If smart the attention is computed using a (ZxZ parameters!) linear layer.
    Otherwise, it is computed using a (Zx2 paramters!) scaling and bias layer.
    """

    def __init__(
        self, latent_dim: int, pixel_feature_dim: int, in_place: bool = False, layer_scale=1e-6, smart: bool = True
    ):
        super().__init__()

        # Filter creator creates
        self.C, self.Z = pixel_feature_dim, latent_dim
        self.in_place, self.smart = in_place, smart

        if self.smart:
            self.query = nn.Linear(self.Z, self.Z)
            self.key = nn.Linear(self.Z, self.Z)
        else:
            self.query = nn.LayerNorm(self.Z)
            self.key = nn.LayerNorm(self.Z)
        self.pixel_norm = nn.GroupNorm(num_groups=1, num_channels=self.C, affine=True)
        self.activation = nn.GELU()

        self.layer_scale = nn.Parameter(torch.ones(pixel_feature_dim, 1, 1) * layer_scale)

    def compute_relationships(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise relationships between tokens of a single bag.

        Args:
            tokens: (#token=B, #features=Z)

        Returns:
            torch.Tensor: (1, #token, #token)
        """
        # For computing q there are two rivaling approaches:
        # 1. Scale it before computing the dot product (as in PyTorch implementation of multi_head_attention_forward)
        # 2. Scale it after computing the dot product (as in original paper)
        q = self.query(tokens)  # q.shape = #token, #features
        q = q / (q.shape[-1] ** 0.5)

        k = self.key(tokens)  # k.shape = #token, #features

        # We want to be able to extend this to multiple sequences, so we use bmm with a batch dimension of 1
        score = torch.bmm(q.unsqueeze(0), k.unsqueeze(0).transpose(-2, -1))  # score.shape = 1, #token, #token

        # Compute the softmax over the last dimension, such each weighted sum of values is 1
        # return F.softmax(score, dim=-1)  # 1, #token, #token

        # Alternatively, as done in https://github.com/aL3x-O-o-Hung/GLCSA_ECLoss/blob/main/GLCSA_network.py
        return F.sigmoid(score)  # 1, #token, #token

    def forward_group(
        self,
        tokens: torch.Tensor,  # (#token=B, #features=Z)
        pixel_features: torch.Tensor,  # (#token=B, C, H, W)
    ):
        """
        Computes mixing filters for a single group of tokens (the tokens of one training subject).
        """
        B, Z = tokens.shape
        B2, C, H, W = pixel_features.shape
        assert B == B2, f"Batch size mismatch of semantic and dense features: {B} != {B2}"
        if not hasattr(self, "layer_scale"):  # backwards compatibility
            logging.warning(f"Please update your model, mixer has no layer scale!")
            self.layer_scale = nn.Parameter(torch.ones(C, 1, 1) * 1e-6).to(pixel_features.device)
            self.query = nn.LayerNorm(self.Z).to(tokens.device)
            self.key = nn.LayerNorm(self.Z).to(tokens.device)

        # Semantically, the pairwise embeddings should score the relatedness of the tokens
        # Which informs how the filter should be applied for joining pairwise dense features
        relationships = self.compute_relationships(tokens)

        # In transformer language we need (#Sequences, Sequence_length, #ValueChannels) = (H W), (bag size), (pixel_dim)
        value = rearrange(pixel_features, "b c h w -> (h w) b c")
        value = torch.bmm(relationships.expand(value.shape[0], -1, -1), value)
        assert value.shape == (H * W, B, C)
        # Transform back into the original shape
        value = rearrange(value, "(h w) b c -> b c h w", h=H, w=W)

        results = self.layer_scale * self.activation(self.pixel_norm(value))
        return pixel_features + results
        # value = self.activation(self.pixel_norm(value))
        # return pixel_features + value

    def forward(
        self,
        tokens: torch.Tensor,  # (#token=B, #features=Z)
        pixel_features: torch.Tensor,  # (#token=B, C, H, W)
        group_indices: torch.Tensor,  # (#token=B) sorted list of group membership like [0, 0, 0, 1, 2, 2]
    ) -> torch.Tensor:  # (#token=B, C, #token, 1, 1)
        """
        In scaled dot product attention, the value is the token itself,
        and the output value dimensionality corresponds to the number of tokens.
        """
        res = pixel_features if self.in_place else pixel_features.clone()
        for group in group_indices.unique(sorted=True):
            group_mask = group_indices == group
            group_tokens = tokens[group_mask]
            group_pixel_features = pixel_features[group_mask]

            res[group_mask] = self.forward_group(group_tokens, group_pixel_features)
        return res
