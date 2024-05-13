import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.utils import make_grid

from typing import Literal

from .SharedBlock import SharedBlock


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


class AttentionPoolingReducer(nn.Module):
    """
    Attention pooling akin to ABMIL https://arxiv.org/abs/1802.04712
    an be used with gated attention (default) or regular attention.
    Will apply a linear layer after the attention pooling to ensure embedding_dim is met.
    """

    def __init__(self, embedding_dim, gated: bool = True, num_heads: int = 1) -> None:
        super(AttentionPoolingReducer, self).__init__()
        self.embedding_dim = embedding_dim

        # Different number of attention heads
        self.num_heads = num_heads

        # boolean if to use gated attention or not
        self.gated = gated
        if gated:
            self.attention_v = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim // 3), nn.Tanh())
            self.attention_u = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim // 3), nn.Sigmoid())
            self.attention = nn.Linear(self.embedding_dim // 3, num_heads)
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim // 3),
                nn.Tanh(),
                nn.Linear(self.embedding_dim // 3, num_heads),  # weight which feature is most important
            )

        # in order to map the output dimennsion of multiple attention heads
        # back to the embedding_dim, we use a linear mapper
        self.mapper = nn.Linear(self.embedding_dim * num_heads, self.embedding_dim)

    def _attenuate(self, x):
        """
        Attentuate one bag of instances (x)
        """
        if self.gated:
            return self.attention(self.attention_v(x) * self.attention_u(x))
        else:
            return self.attention(x)

    def forward(self, x, supercase_indices):
        outs = []
        weights = []
        # since we don't know about the individual bag sizes
        # we iterate over the unique indices and perform the calculations individually
        # This should be optimized for runtime in the future.
        for idxs in supercase_indices.unique():
            case = x[supercase_indices == idxs]
            att = self._attenuate(case)  # Bags x Num Instances x ATTENTION_BRANCHES
            att = torch.transpose(att, 1, 0)  # Bags x ATTENTION_BRANCHES x Num Instances
            att = F.softmax(att, dim=1).clamp(1e-5)  # softmax over instances of one Bag

            outs.append(self.mapper(torch.matmul(att, case).view(1, -1)))
            weights.append(att.view(-1, self.num_heads))

        return torch.cat(outs).reshape(-1, self.embedding_dim), torch.cat(weights).reshape(-1, self.num_heads)


class WeightedAvgPoolReducer(nn.Module):
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

    def forward(self, x, supercase_indices):
        weights = F.sigmoid(self.weightgiver(x))
        # Apply weights to subcases
        x = x * weights
        return self.average_group_pool(x, supercase_indices), weights


class Grouper(SharedBlock):
    class Config(SharedBlock.Config):
        version: Literal["weighted", "attention", "gated-attention"] = "weighted"
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
            self.reducer = AttentionPoolingReducer(self.embedding_dim, gated=False, num_heads=self.args.attention_heads)
        elif self.args.version == "gated-attention":
            self.reducer = AttentionPoolingReducer(self.embedding_dim, gated=True, num_heads=self.args.attention_heads)
        else:
            raise NotImplementedError
            "The selected grouper is not implemented"
        self.make_mtl_compatible()

    @staticmethod
    def extract_ids_from_batch(ids: list[str]):
        """
        Takes a list of ids like ["a", "a", "b"] and return a torch.Tensor([0, 0, 1])
        """
        unique_ids = list(set(ids))
        id_to_index = {id_: i for i, id_ in enumerate(unique_ids)}
        return torch.tensor([id_to_index[id_] for id_ in ids])

    def unify_bagsizes(
        self, bag: torch.Tensor, supercase_indices: torch.Tensor, supercase_labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        function to unify the sizes of a batch of bags to keep attention based training more efficient
        and keep the flixibility of different bag sizes. It is assumed, that all images have the same CxHxW dimensions
        MIGHT RESULT IN MULTIPLE GRADIENTS FOR ONE IMAGE!
        """
        if self.args.version != "weighted":
            _, c, h, w = bag.shape
            # get supercase infos of how many different bags there are and select the biggest one
            uniques, counts = supercase_indices.unique(return_counts=True)
            if torch.all(counts == counts[0]):
                return bag, supercase_indices, supercase_labels
            else:
                new_bag_size = counts.max()
                # get all instances of each bag
                l = []
                new_indices = []
                new_labels = []
                for u in uniques:
                    idx = torch.argwhere(supercase_indices == u)
                    label = supercase_labels[idx].unique()
                    b_tmp = bag[idx].reshape(-1, c, h, w)
                    # and repeat them new_bag_size_time (complete overkill and need to be optimized)
                    l.append(b_tmp.repeat(new_bag_size, 1, 1, 1)[:new_bag_size])
                    new_indices.append(torch.Tensor([u for _ in range(new_bag_size)]))
                    new_labels.append(torch.Tensor([label for _ in range(new_bag_size)]))
                unified_bag = torch.stack(l).reshape(new_bag_size * len(counts), c, h, w)  # Batch x N x hidden_dim
                new_indices = torch.stack(new_indices).flatten()
                new_labels = torch.stack(new_labels).flatten()
                return (
                    unified_bag,
                    new_indices.long(),
                    new_labels.type(supercase_labels.dtype).to(supercase_labels.device),
                )
        else:
            return bag, supercase_indices, supercase_labels

    def group_targets(self, subcase_labels: torch.Tensor, supercase_indices: torch.Tensor):
        """
        For example in classification, all subcases have the same target.
        In consequence, each subcase label is the correct supercase label.
        """
        # For each supercase we need to identify a subcase where we can extract the label from
        subcase_representative_indices = torch.LongTensor(
            [torch.where(supercase_indices == supercase_index)[0][0] for supercase_index in supercase_indices.unique()]
        )
        return subcase_labels[subcase_representative_indices]

    def forward(self, subcases: torch.Tensor, supercase_indices: torch.Tensor):
        """
        subcases (float): (batch_size, embedding_dim)
        supercase_indices (long): (batch_size,)
        """

        return self.reducer(subcases, supercase_indices)
