import logging
import random
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmm.BaseModel import BaseModel
from mmm.torch_ext import CachingSubCaseDS, CachingSubCaseDSSampler
from torchvision.utils import make_grid

from .SharedBlock import SharedBlock


class GroupSampler(CachingSubCaseDSSampler):
    class Config(BaseModel):
        sampler_per_group: tuple[int, int] = (20, 50)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        super().__init__()
        self.samples_left_for_group = 0
        self.current_group = None

    def hook_new_subcases(self, subcases: list):
        """
        Each subcase has a "group_id" in the "meta" dictionary.
        This functions returns the group_ids of the subcases
        """
        self.group_map: dict = {}  # maps group id to list of subcase indices
        for i, subcase in enumerate(self.cacheds.subcases + subcases):
            if "group_id" in subcase["meta"]:
                group_id = subcase["meta"]["group_id"]
            else:
                group_id = subcase["meta"]["supermeta"]["group_id"]
            if group_id not in self.group_map:
                self.group_map[group_id] = []
            self.group_map[group_id].append(i)
        return subcases

    def sample_from_cache(self, draining_phase: bool) -> int:
        # After removal all indices are invalid, recreate the group map
        self.hook_new_subcases([])
        if self.current_group not in self.group_map or self.samples_left_for_group <= 0:
            self.current_group = random.choice(list(self.group_map.keys()))
            self.samples_left_for_group = random.randint(*self.cfg.sampler_per_group)
        self.samples_left_for_group -= 1
        subcase_index = random.choice(self.group_map[self.current_group])
        return subcase_index


def make_grid_for_supercase(training_ims, supercase_indices, group_index, grouper_weights):
    vis_indices = torch.where(supercase_indices == group_index)[0].cpu()
    rows = int(np.sqrt(len(vis_indices)))
    # cols = training_ims[vis_indices].shape[0] // rows
    grid_img = make_grid(training_ims[vis_indices], nrow=rows)

    vis_cases_weights = grouper_weights[vis_indices]
    # with multiple weights available for one image each row is a list of the corresponding weights
    weight_chunks = []
    for vis_case in vis_cases_weights:
        weight_chunks.append(",".join([f"{single_w:.2f}" for single_w in vis_case]))
    weight_rows = [f"[{w}]" for w in weight_chunks]
    weight_rows[-1] = weight_rows[-1] + " " * (len(weight_rows[0]) - len(weight_rows[-1]))
    weight_str = "\n".join(weight_rows)
    return grid_img, weight_str, vis_indices


class Reducer(nn.Module):
    def forward(self, x, supercase_indices):
        raise NotImplementedError

    def get_last_weights(self):
        raise NotImplementedError

    def rate_instance_relevance(self, x, supercase_indices):
        raise NotImplementedError


class AttentionPoolingReducer(Reducer):
    """
    Attention pooling akin to ABMIL https://arxiv.org/abs/1802.04712
    an be used with gated attention (default) or regular attention.
    Will apply a linear layer after the attention pooling to ensure embedding_dim is met.
    """

    def __init__(self, embedding_dim, num_heads: int = 1) -> None:
        super(AttentionPoolingReducer, self).__init__()
        self.embedding_dim = embedding_dim
        # Different number of attention heads
        self.num_heads = num_heads
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(self.embedding_dim // 2, num_heads),
        )

    def get_last_weights(self):
        return torch.cat(self.weights).reshape(-1, self.num_heads)

    def forward(self, x, supercase_indices):
        outs = []
        self.weights = []
        # since we don't know about the individual bag sizes
        # we iterate over the unique indices and perform the calculations individually
        for idxs in supercase_indices.unique(sorted=True):
            case = x[supercase_indices == idxs]
            att = self.attention(case)
            att = torch.transpose(att, 1, 0)
            att = F.softmax(att, dim=1)

            outs.append(torch.matmul(att, case).view(self.num_heads, -1).mean(dim=0).view(1, -1))
            self.weights.append(att.view(-1, self.num_heads))

        return torch.stack(outs, dim=0).reshape(-1, self.embedding_dim)

    def rate_instance_relevance(self, x, supercase_indices):
        weights = []
        for idxs in supercase_indices.unique():
            case = x[supercase_indices == idxs]
            att = self.attention(case)
            att = torch.transpose(att, 1, 0)
            att = F.softmax(att, dim=1)
            weights.append(att.view(-1, self.num_heads))
        return torch.cat(weights).reshape(-1, self.num_heads)


class CLAMReducer(Reducer):
    """
    CLAM will select the top 10 instances of each bag and return them in addition to the pooled Bag.
    Akin to https://www.nature.com/articles/s41551-020-00682-w
    """

    def __init__(self, embedding_dim, num_heads: int = 1, num_instances: int = 10) -> None:
        super(CLAMReducer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_instances = num_instances
        self.num_heads = num_heads
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(self.embedding_dim // 2, num_heads),
        )

    def get_last_weights(self):
        return torch.cat(self.weights).reshape(-1, self.num_heads)

    def forward(self, x, supercase_indices):
        outs = []
        self.weights = []
        for idxs in supercase_indices.unique(sorted=True):
            case = x[supercase_indices == idxs]
            att = self.attention(case)
            att = torch.transpose(att, 1, 0)
            top_pos_idx = torch.topk(att, k=min(self.num_instances, att.size(1)))[1][-1]
            top_pos = torch.index_select(case, 0, top_pos_idx)
            att = F.softmax(att, dim=1)
            outs.append(torch.matmul(att, case).view(self.num_heads, -1).mean(dim=0).view(1, -1))
            for b in top_pos:
                outs.append(b.view(1, -1))
            self.weights.append(att.view(-1, self.num_heads))

        return torch.stack(outs, dim=0).reshape(-1, self.embedding_dim)

    def rate_instance_relevance(self, x, supercase_indices):
        weights = []
        for idxs in supercase_indices.unique():
            case = x[supercase_indices == idxs]
            att = self._attention(case)
            att = F.softmax(att, dim=1)
            weights.append(att.view(-1, self.num_heads))
        return torch.cat(weights).reshape(-1, self.num_heads)


class WeightedAvgPoolReducer(Reducer):
    """
    Simple weighted average to reduce weight and pool all instances within the Bag
    """

    def __init__(self, embedding_dim) -> None:
        super(WeightedAvgPoolReducer, self).__init__()
        self.embedding_dim = embedding_dim
        self.weightgiver = nn.Linear(self.embedding_dim, 1)

    def average_group_pool(self, subcases: torch.Tensor, supercase_indices: torch.Tensor):
        counts = torch.bincount(supercase_indices)
        supercase_repr = torch.zeros(counts.shape[0], self.embedding_dim, device=subcases.device)
        supercase_repr.index_add_(0, supercase_indices, subcases)
        return supercase_repr / counts.float().unsqueeze(1)

    def get_last_weights(self):
        return self.weights

    def forward(self, x, supercase_indices):
        weights = F.sigmoid(self.weightgiver(x))
        self.weights = weights
        # Apply weights to subcases
        x = x * weights
        return self.average_group_pool(x, supercase_indices)

    def rate_instance_relevance(self, x, supercase_indices):
        return F.sigmoid(self.weightgiver(x))


class Grouper(SharedBlock):
    class Config(SharedBlock.Config):
        version: Literal["weighted", "attention", "clam-attention"] = "attention"
        attention_heads: int = 1
        module_name: str = "grouper"

    def __init__(self, args: Config, embedding_dim: int):
        super().__init__(args)
        self.args: Grouper.Config = args
        self.embedding_dim = embedding_dim

        # different versions also need to be treated differently
        if self.args.version == "weighted":
            self.reducer = WeightedAvgPoolReducer(self.embedding_dim)
        elif self.args.version == "attention":
            self.reducer = AttentionPoolingReducer(self.embedding_dim, num_heads=self.args.attention_heads)
        elif self.args.version == "clam-attention":
            self.reducer = CLAMReducer(self.embedding_dim, num_heads=1, num_instances=10)
        else:
            raise NotImplementedError("The selected grouper is not implemented")
        self.make_mtl_compatible()

    def rate_instance_relevance(self, x, supercase_indices):
        return self.reducer.rate_instance_relevance(x, supercase_indices)

    @staticmethod
    def extract_ids_from_batch(ids: list[str]):
        """
        Takes a list of ids like ["a", "a", "b"] and return a torch.Tensor([0, 0, 1])
        """
        unique_ids = list(set(ids))
        id_to_index = {id_: i for i, id_ in enumerate(unique_ids)}
        return torch.tensor([id_to_index[id_] for id_ in ids])

    def group_targets(self, subcase_labels: torch.Tensor, supercase_indices: torch.Tensor):
        """
        For example in classification, all subcases have the same target.
        In consequence, each subcase label is the correct supercase label.
        """
        # For each supercase we need to identify a subcase where we can extract the label from
        subcase_representative_indices = torch.LongTensor(
            [torch.where(supercase_indices == supercase_index)[0][0] for supercase_index in supercase_indices.unique()]
        )

        # for CLAM instances are also returned. Therefore the groupings needs to be adjusted
        if hasattr(self.reducer, "num_instances"):
            label = []
            num_instances = self.reducer.num_instances
            for l in subcase_labels[subcase_representative_indices]:
                label.extend([l for _ in range(num_instances + 1)])

            return torch.tensor(label).view(-1).to(subcase_labels.device)
        else:
            return subcase_labels[subcase_representative_indices]

    def forward(self, subcases: torch.Tensor, supercase_indices: torch.Tensor):
        """
        subcases (float): (batch_size, embedding_dim)
        supercase_indices (long): (batch_size,)
        """
        reduced = self.reducer(subcases, supercase_indices)
        weights = self.reducer.get_last_weights()
        return reduced, weights
