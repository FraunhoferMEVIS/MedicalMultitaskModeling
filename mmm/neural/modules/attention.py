import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmm.BaseModel import BaseModel


# Adapted from https://github.com/jaketae/alibi for custom positions
def get_relative_positions_custom(custom_positions: torch.Tensor, symmetric: bool) -> torch.Tensor:
    """
    For custom positions such as [0, 0, 1, 3, 4, 5]. In other words, its ok for positions to be non-consecutive.

    To encode symmetric distances, set `summetric=True`.
    """
    x = custom_positions[None, :]  # [1, S]
    y = custom_positions[:, None]  # [S, 1]
    res = x - y  # [S, S]
    if symmetric:
        res = -1 * torch.abs(res)  # [S, S]
    return res


def get_alibi_slope(num_heads: int) -> torch.Tensor:
    """
    Computes the m parameter for the ALiBi attention mechanism which controls the strength of the bias for each head.
    """
    x = (2**8) ** (1 / num_heads)
    return torch.tensor([1 / x ** (i + 1) for i in range(num_heads)]).unsqueeze(-1).unsqueeze(-1)


def build_mha_alibi_mask(m: torch.Tensor, positions: torch.Tensor, symmetric: bool) -> torch.Tensor:
    H = m.size(0)  # Number of heads
    rel_pos = get_relative_positions_custom(positions, symmetric)  # [S, S]
    alibi_mask = m * rel_pos  # [H, S, S] can be added to [B, H, S, S] attention mask
    # However, in our case, each batchitem has different positions, so we don't expand across batch dimension here.
    return alibi_mask


def create_multihead_padding_mask(key_padding_mask: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expands boolean key_padding_mask to multihead attention mask with -inf at padding positions.
    The boolean key_padding_mask might come from `mmm.utils.build_batch_of_sequences`

    Args:
        key_padding_mask: Boolean tensor of shape [batch_size, sequence_length]
        num_heads: Number of attention heads

    Returns:
        Float tensor of shape [batch_size, num_heads, sequence_length, sequence_length]
        with -inf at padded positions and 0 elsewhere

    Without masking, changing one item in the sequence will impact all other items

    >>> seq_batch_heads = torch.rand(2, nh:=2, S:=3, 4)  # [B, H, SeqLen, E]
    >>> v_before = sdpa_weights(seq_batch_heads, seq_batch_heads, False) @ seq_batch_heads
    >>> seq_batch_heads[1, 0, -1, :] = torch.rand(4)  # Change the padded embedding of the second sequence
    >>> v_after = sdpa_weights(seq_batch_heads, seq_batch_heads, False) @ seq_batch_heads
    >>> [torch.allclose(v_before[..., instance_index, :], v_after[..., instance_index, :]) for instance_index in range(S)]
    [False, False, False]

    Now, we use a padding mask to ensure that the change in the second sequence does not affect its other items:

    >>> padding_mask, fixnans = create_multihead_padding_mask(torch.tensor([[False, False, False], [False, False, True]]), nh)
    >>> v_before = sdpa_weights(seq_batch_heads, seq_batch_heads, False, attn_mask=padding_mask) @ seq_batch_heads
    >>> seq_batch_heads[1, 0, -1, :] = torch.rand(4)  # Change the padded embedding of the second sequence
    >>> v_after = sdpa_weights(seq_batch_heads, seq_batch_heads, False, attn_mask=padding_mask) @ seq_batch_heads
    >>> [torch.allclose(v_before[..., instance_index, :], v_after[..., instance_index, :]) for instance_index in range(S)]
    [True, True, False]

    First sequence should not be affected by the change in the second sequence.
    >>> torch.allclose(v_before[0, ...], v_after[0, ...])
    True

    The safety mask (fixnans) will do nothing for non-padded
    """
    batch_size, seq_len = key_padding_mask.size()

    mask = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, num_heads, -1, -1).expand(-1, -1, seq_len, -1)

    # mask every position for padded tokens
    mask = mask | mask.transpose(-1, -2)

    # Create a safety mask avoiding nans by enabling padding tokens to attend to themselves
    safety_mask = (
        torch.eye(seq_len, dtype=torch.bool, device=key_padding_mask.device)[None, None].expand(
            batch_size, num_heads, -1, -1
        )
        & mask
    )

    return mask.float().masked_fill(mask, float("-inf")).to(key_padding_mask.device), safety_mask


class MHAMask(BaseModel):
    """ """

    def build_mask(self, b: int, seqlen: int, num_heads: int, device: torch.device):
        """
        From the PyTorch docs: "A 2D mask will be broadcasted for all
        the batches while a 3D mask allows to specify a different mask for the entries of each batch."
        In our case, we always use multihead float masks of the form [batchsize, num_heads, seqlen, seqlen].
        Masked out values should be set to -inf.

        To combine such masks, it is valid to use addition.
        """
        raise NotImplementedError


class DropRandomMHAMask(MHAMask):
    mask_type: Literal["droprandom"] = "droprandom"
    drop_prob: float = 0.5
    same_for_each_head: bool = True

    def build_mask(self, b: int, seqlen: int, num_heads: int, device: torch.device):
        """
        It always attends to itself and randomly drops attention to other tokens.
        """

        if self.same_for_each_head:
            boolmask_allheads = torch.rand(size=(b, seqlen, seqlen), device=device) > self.drop_prob
            boolmask = boolmask_allheads.unsqueeze(1).expand(-1, num_heads, -1, -1)
        else:
            boolmask = torch.rand(size=(b, num_heads, seqlen, seqlen), device=device) > self.drop_prob

        # Make sure that in each head, each token attends to itself
        identity_mask = torch.eye(seqlen, dtype=torch.bool, device=device)[None, None].expand(b, num_heads, -1, -1)
        boolmask = identity_mask | boolmask

        return torch.zeros_like(boolmask, dtype=torch.float).masked_fill(boolmask.logical_not(), float("-inf"))


def sdpa_weights(query, key, train: bool, attn_mask: torch.Tensor | None = None, dropout_p=0.0) -> torch.Tensor:
    """
    Equivalent to `torch.nn.functional.scaled_dot_product_attention` but enables attention weight extraction.

    Causal mask can be created by `torch.ones(S, S, dtype=torch.bool).tril(diagonal=0)`

    Args:
        query (torch.Tensor): Query tensor of shape (B, nh, S, E)
        key (torch.Tensor): Key tensor of shape (B, nh, S, E)
        attn_mask (torch.FloatTensor, optional): Attention mask, None for full attention without padding.
            (B, S, S) will apply the same bias to all heads.
            (B, nh, S, S) will apply enable different bias to each head.
            Use -inf for masking out attention as `torch.softmax` will ignore it.
        dropout_p (float, optional): Dropout probability. Should be 0.0 during inference.

    >>> import torch
    >>> from mmm.neural.modules.attention import sdpa_weights
    >>> q, k, v = torch.rand(2, nh:=2, 3, 4), torch.rand(2, nh, 3, 4), torch.rand(2, nh, 3, 4)
    >>> attn_weights = sdpa_weights(q, k, train=False)
    >>> value = attn_weights @ v
    >>> torch.allclose(value, F.scaled_dot_product_attention(q, k, v))
    True
    """
    batch_size, num_heads, seq_len, embedding_dim = query.size()

    attn_weight = query @ key.transpose(-2, -1) * (1.0 / math.sqrt(embedding_dim))

    if attn_mask is not None:
        attn_weight += attn_mask

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=train)
    return attn_weight


class MHABlock(nn.Module):
    """
    Computes multi-head attention.

    Attention weights can be verified as:

    >>> seq_batch = torch.rand(B:=2, S:=3, E:=4)
    >>> mha_block = MHABlock(embed_dim=E, nheads=2, dropout=0.0)
    >>> r1, a1 = mha_block(seq_batch, need_weights=True)
    >>> r2, a2 = mha_block(seq_batch, need_weights=False)
    >>> torch.allclose(r1, r2), list(a1.shape), a2
    (True, [2, 2, 3, 3], None)

    Args:
        embed_dim: Size of embedding dim for query, key, value
        E_total: Total embedding dim of combined heads post input projection. Each head has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        embed_dim: int,
        nheads: int,
        dropout: float = 0.1,
        bias=True,
    ):
        super().__init__()
        self.nheads, self.dropout, self.bias = nheads, dropout, bias
        assert embed_dim % nheads == 0, f"{embed_dim=} is not divisible by {nheads=}"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.E_head = embed_dim // nheads

    def forward(self, v: torch.Tensor, attn_mask=None, need_weights=False) -> torch.Tensor:
        """
        Args:
            value (torch.Tensor): value of shape B=batchsize, S=sequence length, E=embedding dim
            attn_mask (torch.Tensor, optional): attention mask of shape BxSxS
            is_causal (bool, optional): Whether to apply causal mask. Default: False
            need_weights (bool, optional): materializing attention weights is expensive
        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        B, T, E = v.size()  # batch size, sequence length, embedding dimensionality
        q, k, v = torch.chunk(self.qkv_proj(v), 3, dim=-1)
        k = k.view(B, T, self.nheads, E // self.nheads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.nheads, E // self.nheads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.nheads, E // self.nheads).transpose(1, 2)  # (B, nh, T, hs)

        if need_weights:
            # (N, nheads, L_t, E_head)
            attn_weights = sdpa_weights(q, k, attn_mask=attn_mask, dropout_p=self.dropout, train=self.training)
            attn_output = attn_weights @ v

            attn_output = rearrange(attn_output, "N nheads L_t E_head -> N L_t (nheads E_head)")
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=(self.dropout if self.training else 0), attn_mask=attn_mask
            )
            attn_weights = None

            # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
            attn_output = attn_output.transpose(1, 2).flatten(-2)
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class MLP(nn.Module):
    def __init__(self, n_embd: int, expansion_factor: int, bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, expansion_factor * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(expansion_factor * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class AttnMLPBlock(nn.Module):
    def __init__(self, n_embd: int, nheads: int, expansion_factor: int = 4, bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = MHABlock(embed_dim=n_embd, nheads=nheads, dropout=dropout, bias=bias)
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, expansion_factor, bias=bias, dropout=dropout)

    def forward(self, x, attn_mask=None, need_weights: bool = False):
        attn_output, attn_weights = self.attn(self.ln_1(x), attn_mask=attn_mask, need_weights=need_weights)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


# class MHALayer(nn.Module):
#     def __init__(self, d_model: int, nhead: int, expansion_factorloat = 0.1, bias=True):
#         super().__init__()
#         self.self_attn = MHABlock(embed_dim=d_model, nheads=nhead, dropout=dropout)
#         dim_feedforward = d_model * expansion_factor

#         self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
#         self.norm1 = nn.LayerNorm(d_model, bias=bias)
#         self.norm2 = nn.LayerNorm(d_model, bias=bias)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#     def _sa_block(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
#         x, attn_weights = self.self_attn(x, attn_mask=attn_mask, need_weights=True)
#         return self.dropout1(x), attn_weights

#     def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.linear2(self.dropout(F.gelu(self.linear1(x))))
#         return self.dropout2(x)

#     def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
#         """
#         Args:
#             x (torch.Tensor): Input tensor of shape (B, S, E), where B=batchsize, S=sequence length, E=embedding dim
#             attn_mask (torch.Tensor | None): Attention mask of shape (B, S, S) or None for full attention.
#         """
#         attention_output, attn_weights = self._sa_block(self.norm1(x), attn_mask)
#         x = x + attention_output
#         x = x + self._ff_block(self.norm2(x))
#         return x, attn_weights
