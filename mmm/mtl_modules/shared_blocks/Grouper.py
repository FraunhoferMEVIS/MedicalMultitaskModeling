import random
import uuid
from typing import Annotated, Literal

import logfire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from torchvision.utils import make_grid

from mmm.BaseModel import BaseModel
from mmm.mmm_types.GroupUsage import GroupingStrategy, GroupUsage, IncompatibleUsageError, MaskingStrategy
from mmm.neural.modules.attention import (
    AttnMLPBlock,
    DropRandomMHAMask,
    MHAMask,
    build_mha_alibi_mask,
    create_multihead_padding_mask,
    get_alibi_slope,
)
from mmm.torch_ext import CachingSubCaseDSSampler, SubCaseType
from mmm.utils import build_batch_of_instances, build_batch_of_sequences, extract_ids_from_batch

from .SharedBlock import SharedBlock


def determine_group_id(subcase: SubCaseType) -> int:
    if "group_id" in subcase["meta"]:
        group_id = subcase["meta"]["group_id"]
    else:
        group_id = subcase["meta"]["supermeta"]["group_id"]
    return group_id


class FullGroupsampler(CachingSubCaseDSSampler):
    def __init__(self):
        super().__init__()
        self.current_group = None
        self.current_index = 0

    def sample_from_cache(self, draining_phase):
        if (
            (self.current_group is None)
            or (self.current_index >= len(self.cacheds.subcases))
            or (determine_group_id(self.cacheds.subcases[self.current_index]) != self.current_group)
        ):
            self.current_index = random.randint(0, len(self.cacheds.subcases) - 1)
            self.current_group = determine_group_id(self.cacheds.subcases[self.current_index])
            # Move the index to the first subcase of the new group
            while (
                self.current_index > 0
                and determine_group_id(self.cacheds.subcases[self.current_index - 1]) == self.current_group
            ):
                self.current_index -= 1
        return self.current_index


class GroupSampler(CachingSubCaseDSSampler):
    class Config(BaseModel):
        sampler_per_group: tuple[int, int] = (20, 50)
        removal_prob: float = 1.0

    def __init__(self, cfg: Config):
        self.cfg = cfg
        super().__init__()
        self.samples_left_for_group = 0
        self.current_group = None
        self.groups_invalid = True

    def hook_new_subcases(self, subcases: list):
        """
        Each subcase has a "group_id" in the "meta" dictionary.
        This functions returns the group_ids of the subcases
        """
        self.group_map: dict = {}  # maps group id to list of subcases
        for i, subcase in enumerate(self.cacheds.subcases + subcases):
            if (group_id := determine_group_id(subcase)) not in self.group_map:
                self.group_map[group_id] = []

            self.group_map[group_id].append(i)
        self.groups_invalid = False
        return subcases

    def decide_removal(self, popped_case: SubCaseType, draining_phase: bool, index: int) -> bool:
        if draining_phase or self.cfg.removal_prob >= 1.0:
            res = True
        else:
            res = random.random() < self.cfg.removal_prob
        if res:
            # After removal all indices are invalid, recreate the group map
            self.groups_invalid = True
        return res

    def sample_from_cache(self, draining_phase: bool) -> int:
        if self.groups_invalid:
            self.hook_new_subcases([])

        if self.current_group not in self.group_map or self.samples_left_for_group <= 0:
            self.current_group = random.choice(list(self.group_map.keys()))
            self.samples_left_for_group = random.randint(*self.cfg.sampler_per_group)
        self.samples_left_for_group -= 1
        subcase_index = random.choice(self.group_map[self.current_group])
        return subcase_index


def make_grid_for_supercase(
    training_ims: torch.Tensor,
    supercase_indices: torch.Tensor,
    group_index,
    grouper_metainfo: dict,
    max_images=25,
    with_masks: list[torch.Tensor] | None = None,
    with_boxes: list | None = None,
):
    """
    Args:
        training_ims: [B, 3, H, W] tensor of images
        supercase_indices: [B] tensor of supercase indices
        group_index: int of the group to visualize
    """
    vis_indices = torch.where(supercase_indices == group_index)[0].cpu()

    if len(vis_indices) > max_images:
        caption = f"Showing first {max_images}/{len(vis_indices)} images"
        vis_indices = vis_indices[:max_images]
    else:
        caption = f"Showing all {len(vis_indices)} images"

    rows = int(np.sqrt(len(vis_indices)))
    # cols = training_ims[vis_indices].shape[0] // rows
    grid_img = make_grid(training_ims[vis_indices], nrow=rows, padding=0)

    if grouper_metainfo is not None:
        grouper_metastr = ""
        for key, value in grouper_metainfo.items():
            if isinstance(value, torch.Tensor):
                if value.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    grouper_metastr += f"{key} ({value.shape}): {np.array2string(value.cpu().numpy(), precision=2, floatmode='fixed', threshold=2500, suppress_small=True)}\n"
                else:
                    grouper_metastr += f"{key}: {value.tolist()}\n"
            else:
                grouper_metastr += f"{key}: {value}\n"
        caption = f"{caption}\n{grouper_metastr}"

    with_annos = {}

    if with_masks is not None:
        caption = f"{[torch.unique(m, return_counts=True) for m in with_masks]}\n{caption}"
        with_annos["masks"] = [
            make_grid(m[vis_indices].unsqueeze(1).expand(-1, 3, -1, -1), nrow=rows, padding=0)[0] for m in with_masks
        ]

    if with_boxes is not None:
        # Move boxes to their respective images
        for i, vis_idx in enumerate(vis_indices):
            # Find the place for this box
            row, col = divmod(i, rows)
            add_y, add_x = row * training_ims.shape[-2], col * training_ims.shape[-1]
            with_boxes["gtboxes"][vis_idx]["bboxes"][:, [0, 2]] += add_x
            with_boxes["gtboxes"][vis_idx]["bboxes"][:, [1, 3]] += add_y
            with_boxes["predboxes"][vis_idx]["bboxes"][:, [0, 2]] += add_x
            with_boxes["predboxes"][vis_idx]["bboxes"][:, [1, 3]] += add_y

        # Add only the vis_indices to the boxes
        with_annos["boxes"] = {}
        with_annos["boxes"]["gtboxes"] = [with_boxes["gtboxes"][i] for i in vis_indices]
        with_annos["boxes"]["predboxes"] = [with_boxes["predboxes"][i] for i in vis_indices]

    return grid_img, caption, vis_indices, with_annos


class Reducer(nn.Module):
    def forward(self, x, supercase_indices, positions=None):
        raise NotImplementedError

    def check_usage(self, usage: GroupUsage) -> bool:
        return True

    def get_last_weights(self):
        raise NotImplementedError

    def rate_instance_relevance(self, x, supercase_indices):
        raise NotImplementedError


class TransformerReducer(Reducer):
    """
    Transformer pooling inspired by
    CAMIL https://arxiv.org/abs/2305.05314
    Snuffy https://arxiv.org/abs/2408.08258

    TransfomerEncoderLayer for contextualization. Taking the avg of the output ensures
    important features talk to each other and all instances get gradient.
    MaxPooling adapted from Snuffy for the instance level importance.

    This pooling strategy should be lightweight and fast.
    """

    def __init__(self, embedding_dim, num_heads: int = 8) -> None:
        super(TransformerReducer, self).__init__()
        self.embedding_dim = embedding_dim
        # Different number of attention heads
        self.num_heads = num_heads
        self.transformer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embedding_dim * 2,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )

    def get_last_weights(self):
        return torch.cat(self.weights).reshape(-1, 1)

    def forward(self, x, supercase_indices, positions=None):
        outs = []
        self.weights = []
        for idxs in supercase_indices.unique(sorted=True):
            case = x[supercase_indices == idxs]
            attenuated = self.transformer(case.view(1, -1, self.embedding_dim))
            att = attenuated.mean(dim=1)
            # max_instance = torch.max(att, dim=1)[0]
            # att = att + max_instance
            with torch.no_grad():
                _, w = self.transformer.self_attn(case, case, case)

            outs.append(att)
            self.weights.append(w.mean(dim=1))

        return torch.stack(outs, dim=0).reshape(-1, self.embedding_dim)


class AttentionPoolingReducer(Reducer):
    """
    Attention pooling akin to ABMIL https://arxiv.org/abs/1802.04712
    an be used with gated attention (default) or regular attention.
    Will apply a linear layer after the attention pooling to ensure embedding_dim is met.
    """

    class Config(BaseModel):
        grouper_type: Literal["attention"] = "attention"
        gated: bool = True
        reduction: Literal["mean", "max", "linear"] = "linear"
        num_heads: int = 1

        def build(self, embedding_dim):
            return AttentionPoolingReducer(self, embedding_dim)

    def check_usage(self, usage):
        return usage.grouping is GroupingStrategy.single

    def __init__(self, cfg: Config, embedding_dim) -> None:
        super(AttentionPoolingReducer, self).__init__()
        self.cfg, self.embedding_dim = cfg, embedding_dim
        # Different number of attention heads
        if not self.cfg.gated:
            self.attention = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                nn.Tanh(),
                nn.Linear(self.embedding_dim // 2, self.cfg.num_heads),
            )
        else:
            self.u = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                nn.Sigmoid(),
            )
            self.v = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                nn.Tanh(),
            )
            self.attention = nn.Sequential(
                nn.Linear(self.embedding_dim // 2, self.cfg.num_heads),
            )
        if self.cfg.reduction == "linear":
            self.mapper = nn.Linear(self.embedding_dim * self.cfg.num_heads, self.embedding_dim)

    def get_last_weights(self):
        return torch.cat(self.weights).reshape(-1, self.cfg.num_heads)

    def _attenuate(self, x):
        if not self.cfg.gated:
            att = self.attention(x)
        else:
            u = self.u(x)
            v = self.v(x)
            att = self.attention(v * u)
        return att

    def forward(self, x, supercase_indices, group_usage: GroupUsage, positions=None):
        outs = []
        self.weights = []
        # since we don't know about the individual bag sizes
        # we iterate over the unique indices and perform the calculations individually
        for idxs in supercase_indices.unique(sorted=True):
            case = x[supercase_indices == idxs]
            att = self._attenuate(case)
            att = torch.transpose(att, 1, 0)
            att = F.softmax(att, dim=1)

            if self.cfg.reduction == "max":
                reduced_case = torch.matmul(att, case).view(self.cfg.num_heads, -1).max(dim=0)[0].view(1, -1)
            elif self.cfg.reduction == "linear":
                reduced_case = self.mapper(torch.matmul(att, case).view(1, -1))
            else:  # default 'mean'
                reduced_case = torch.matmul(att, case).view(self.cfg.num_heads, -1).mean(dim=0).view(1, -1)

            outs.append(reduced_case)
            self.weights.append(att.view(-1, self.cfg.num_heads))

        # Arange weights in a diagnal matrix
        diag_weights = torch.block_diag(*[w for w in self.weights])

        meta_info = {
            "lastweights": diag_weights.cpu(),  # one weight per instance
            "supercase_indices": supercase_indices.cpu(),
            "attn_weights": torch.cat(self.weights).cpu(),  # one weight per instance
        }
        return torch.stack(outs, dim=0).reshape(-1, self.embedding_dim), meta_info

    def rate_instance_relevance(self, x, supercase_indices):
        weights = []
        for idxs in supercase_indices.unique():
            case = x[supercase_indices == idxs]
            att = self.attention(case)
            att = torch.transpose(att, 1, 0)
            att = F.softmax(att, dim=1)
            weights.append(att.view(-1, self.cfg.num_heads))
        return torch.cat(weights).reshape(-1, self.cfg.num_heads)

    def reduces_to_single_instance(self) -> bool:
        return True


class MHAReducer(Reducer):
    """
    The docs of torch.nn.modules.activation.MultiheadAttention describe how to build src_mask and src_key_padding_mask.
    """

    class Config(BaseModel):
        grouper_type: Literal["mha"] = "mha"
        num_layers: int = 4
        num_heads: int = 12
        dropout: float = 0.1
        expansion_factor: int = Field(
            default=4, description="Original transformer uses 4, swin uses 4. GPT2 uses 4. Alibi uses 1."
        )

        redundancy_masks: tuple[DropRandomMHAMask, DropRandomMHAMask] = Field(
            (
                DropRandomMHAMask(drop_prob=0.5, same_for_each_head=False),
                DropRandomMHAMask(drop_prob=0.1, same_for_each_head=False),
            ),
            description="Masking enables training based on multiple variants of the same bag.",
        )
        fullattention_masks: tuple[DropRandomMHAMask, DropRandomMHAMask] = Field(
            (DropRandomMHAMask(drop_prob=0.1, same_for_each_head=False), DropRandomMHAMask(drop_prob=0.0)),
            description="Applied when tasks are configured to use full attention.",
        )

        def build(self, embedding_dim):
            return MHAReducer(self, embedding_dim)

    def __init__(self, cfg: Config, embedding_dim) -> None:
        super().__init__()
        self.cfg, self.embedding_dim = cfg, embedding_dim

        # First and last layer require attention weights.
        self.layers = nn.ModuleList(
            [
                AttnMLPBlock(
                    n_embd=embedding_dim,
                    nheads=cfg.num_heads,
                    expansion_factor=cfg.expansion_factor,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )

    @staticmethod
    def reshape_weights_into_instance_heads(attn_weights):
        """
        Takes raw attention weights of shape (batchsize=1, heads, seqlen, seqlen)
        and returns weights per instance (seqlen, heads).
        """
        batchsize, heads, seqlen, seqlen = attn_weights.shape
        assert batchsize == 1
        return attn_weights.sum(dim=2).squeeze(dim=0).permute(1, 0)

    def rate_instance_relevance(self, x, supercase_indices, group_usage: GroupUsage | None = None, positions=None):
        if group_usage is None:
            group_usage = GroupUsage(grouping=GroupingStrategy.full, masking=MaskingStrategy.fullattention)
        mha_inputs, src_key_padding_mask = build_batch_of_sequences(x, supercase_indices)
        attn_mask = self.get_attn_bias(mha_inputs, src_key_padding_mask, supercase_indices, group_usage)
        assert positions is None, "Position bias not supported for instance relevance rating in MHAReducer"
        x, attn_weights, last_weights = self.forward_model(mha_inputs, attn_mask)

        return self.reshape_weights_into_instance_heads(attn_weights)

    def build_mha_mask(self, for_inputs, group_usage: GroupUsage):
        # strategy = group_usage.masking if self.training else MaskingStrategy.fullattention

        if group_usage.masking is MaskingStrategy.redundancy:
            masking = self.cfg.redundancy_masks[0 if self.training else 1]
        elif group_usage.masking is MaskingStrategy.fullattention:
            masking = self.cfg.fullattention_masks[0 if self.training else 1]
        else:
            raise ValueError(f"Unknown strategy {group_usage.masking}")
        return masking.build_mask(
            b=for_inputs.shape[0],
            seqlen=for_inputs.shape[1],
            device=for_inputs.device,
            num_heads=self.cfg.num_heads,
        )

    def forward_model(self, mha_inputs, attn_mask):
        x, attn_weights = self.layers[0](mha_inputs, attn_mask=attn_mask, need_weights=True)
        for layer in self.layers[1:-1]:
            x, _ = layer(x, attn_mask=attn_mask)
        x, last_weights = self.layers[-1](x, attn_mask=attn_mask, need_weights=True)
        return x, attn_weights, last_weights

    def get_attn_bias(self, mha_inputs, src_key_padding_mask, supercase_indices, group_usage: GroupUsage):
        mha_padding, safety_mask = create_multihead_padding_mask(src_key_padding_mask, self.cfg.num_heads)
        mha_mask = self.build_mha_mask(mha_inputs, group_usage)

        attn_mask = mha_padding + mha_mask
        # For padding tokens (which happens when there is a shorter and a longer sequence in the batch)
        # The combination with mha_mask might lead to full masked rows in the attention mask.
        # In those cases, the result would be a linear combination weighted only by -inf, resulting in NaN values.
        # As we don't care about the padding tokens, make each padding token attend to at least the first token.
        attn_mask[safety_mask] = 0
        return attn_mask

    def get_pos_bias(self, mha_inputs, src_key_padding_mask, supercase_indices, group_usage: GroupUsage, positions):
        pos = build_batch_of_sequences(torch.tensor(positions).to(mha_inputs.device), supercase_indices)[0].long()
        m = get_alibi_slope(self.cfg.num_heads).to(mha_inputs.device)
        pos_mask = torch.cat(
            [
                build_mha_alibi_mask(m=m, positions=pos[b], symmetric=True).unsqueeze(0)
                for b in range(mha_inputs.shape[0])
            ],
            dim=0,
        ).to(mha_inputs.device)
        return pos_mask

    def forward(self, instances, supercase_indices, group_usage: GroupUsage, positions=None):
        mha_inputs, src_key_padding_mask = build_batch_of_sequences(instances, supercase_indices)
        attn_mask = self.get_attn_bias(mha_inputs, src_key_padding_mask, supercase_indices, group_usage)
        if positions is not None:
            pos_mask = self.get_pos_bias(mha_inputs, src_key_padding_mask, supercase_indices, group_usage, positions)
            attn_mask += pos_mask
        x, attn_weights, last_weights = self.forward_model(mha_inputs, attn_mask)

        with torch.no_grad():
            metainfos = {
                "lastweights": last_weights.cpu(),  # one weight per instance
                "padding": src_key_padding_mask.cpu(),
                "supercase_indices": supercase_indices.cpu(),
                "attn_weights": attn_weights.cpu(),  # one weight per instance
            }
            if positions is not None:
                metainfos["positions"] = torch.tensor(positions).to(mha_inputs.device).cpu().flatten()

        # Instead, return the mean within the inverted src_key_padding_mask
        if group_usage.grouping is GroupingStrategy.full:  # return all token representations
            return build_batch_of_instances(x, supercase_indices), metainfos
        elif group_usage.grouping is GroupingStrategy.single:
            return x.sum(dim=1) / (~src_key_padding_mask).sum(dim=1).unsqueeze(1), metainfos
        else:
            raise ValueError(f"Unsupported grouping strategy {group_usage.grouping} for {self.__class__.__name__}")

    def check_usage(self, usage: GroupUsage) -> bool:
        # return not usage.grouping is GroupingStrategy.mixed
        return not usage.grouping is GroupingStrategy.mixed


class WeightedAvgPoolReducer(Reducer):
    """
    Simple weighted average to reduce weight and pool all instances within the Bag
    """

    class Config(BaseModel):
        grouper_type: Literal["weighted"] = "weighted"

        def build(self, embedding_dim):
            return WeightedAvgPoolReducer(self, embedding_dim)

    def __init__(self, cfg: Config, embedding_dim: int) -> None:
        super(WeightedAvgPoolReducer, self).__init__()
        self.cfg, self.embedding_dim = cfg, embedding_dim
        self.weightgiver = nn.Linear(self.embedding_dim, 1)

    def check_usage(self, usage: GroupUsage) -> bool:
        return usage.grouping is GroupingStrategy.single

    def average_group_pool(self, subcases: torch.Tensor, supercase_indices: torch.Tensor):
        counts = torch.bincount(supercase_indices)
        supercase_repr = torch.zeros(counts.shape[0], self.embedding_dim, device=subcases.device)
        supercase_repr.index_add_(0, supercase_indices, subcases)
        return supercase_repr / counts.float().unsqueeze(1)

    def get_last_weights(self):
        return self.weights

    def forward(self, x, supercase_indices, group_usage: GroupUsage, positions=None):
        assert group_usage.grouping is GroupingStrategy.single, f"{type(self)} only supports single grouping"
        weights = F.sigmoid(self.weightgiver(x))
        self.weights = weights
        # Apply weights to subcases
        x = x * weights
        group_embedding = self.average_group_pool(x, supercase_indices), weights
        return group_embedding

    def rate_instance_relevance(self, x, supercase_indices):
        return F.sigmoid(self.weightgiver(x))


class Grouper(SharedBlock):
    _warned_non_adjacent_tasks: set[str] = set()

    class Config(SharedBlock.Config):
        reducer: Annotated[
            WeightedAvgPoolReducer.Config | AttentionPoolingReducer.Config | MHAReducer.Config,
            Field(discriminator="grouper_type"),
        ] = MHAReducer.Config()  # WeightedAvgPoolReducer.Config()
        module_name: str = "grouper"

    def __init__(self, args: Config, embedding_dim: int):
        super().__init__(args)
        self.args: Grouper.Config = args
        self.embedding_dim = embedding_dim

        self.reducer = self.args.reducer.build(embedding_dim)
        self.make_mtl_compatible()

    def raise_for_usageerror(self, usage: GroupUsage):
        if not self.reducer.check_usage(usage):
            raise IncompatibleUsageError(f"{self.reducer} does not support {usage}")

    def rate_instance_relevance(self, x, supercase_indices):
        return self.reducer.rate_instance_relevance(x, supercase_indices)

    @staticmethod
    def extract_ids_from_batch(ids: list[str | None], for_task_name: str | None = None) -> torch.Tensor:
        """
        Takes a list of ids like ["a", "a", "b"] and return a torch.Tensor([0, 0, 1])

        Each None is replaced by a unique id.
        """
        ids_without_none = [i if i is not None else f'None_{str(uuid.uuid4()).replace("-", "_")}' for i in ids]
        res = extract_ids_from_batch(ids_without_none)
        if for_task_name is None or for_task_name not in Grouper._warned_non_adjacent_tasks:
            if not torch.all(res.sort().values == res):
                logfire.warning(
                    "Processing in {for_task_name} is more efficient if group members are adjacent in the batch. {ids}",
                    for_task_name=for_task_name,
                    ids=res.tolist(),
                )
                if for_task_name is not None:
                    Grouper._warned_non_adjacent_tasks.add(for_task_name)
        return res

    def group_targets(
        self,
        subcase_labels: torch.Tensor,
        supercase_indices: torch.Tensor,
        group_usage: GroupUsage,
        return_indices=False,
    ):
        """
        For example in classification, all subcases have the same target.
        In consequence, each subcase label is the correct supercase label.
        """
        if group_usage.grouping is GroupingStrategy.single:
            # For each supercase we need to identify a subcase where we can extract the label from
            subcase_representative_indices = torch.LongTensor(
                [
                    torch.where(supercase_indices == supercase_index)[0][0]
                    for supercase_index in supercase_indices.unique(sorted=True)
                ]
            )
            if return_indices:
                return subcase_labels[subcase_representative_indices], subcase_representative_indices
            return subcase_labels[subcase_representative_indices]
        elif group_usage.grouping is GroupingStrategy.full:
            if return_indices:
                return subcase_labels, torch.arange(len(subcase_labels))
            return subcase_labels
        else:
            raise NotImplementedError(f"Grouping strategy {group_usage.grouping} not implemented.")

    def forward(self, subcases: torch.Tensor, supercase_indices: torch.Tensor, group_usage: GroupUsage, positions=None):
        """
        subcases (float): (batch_size, embedding_dim)
        supercase_indices (long): (batch_size,)
        group_usage (GroupUsage): describes the grouping strategy and masking strategy
        positions (LongTensor | None): positions for the subcases that match the supercase_indices
        """
        # If the reducer returns a single value, return its value, None to signal no weights
        res = self.reducer(subcases, supercase_indices, group_usage, positions=positions)
        if isinstance(res, tuple):
            return res
        return res, None
