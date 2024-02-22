import logging
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
    # Put the weights into rows
    weight_chunks = [vis_cases_weights[i : i + rows] for i in range(0, len(vis_cases_weights), rows)]
    weight_rows = [",".join([f"{w.item():.2f}" for w in weight_row]) for weight_row in weight_chunks]
    # Make sure the last row has the same length as the others
    weight_rows[-1] = weight_rows[-1] + " " * (len(weight_rows[0]) - len(weight_rows[-1]))
    weight_str = "\n".join(weight_rows)
    return grid_img, weight_str, vis_indices


class WeightedAvgPoolReducer(nn.Module):
    """
    Simple weighted average to reduce weight and pool all instances within the Bag
    """

    def __init__(self, embedding_dim) -> None:
        super(WeightedAvgPoolReducer, self).__init__()
        self.embedding_dim = embedding_dim
        self.weightgiver = nn.Linear(self.embedding_dim, 1)

    def average_group_pool(self, subcases: torch.Tensor, supercase_indexes: torch.Tensor):
        counts = torch.bincount(supercase_indexes)
        supercase_repr = torch.zeros(counts.shape[0], self.embedding_dim, device=subcases.device)
        supercase_repr.index_add_(0, supercase_indexes, subcases)
        return supercase_repr / counts.float().unsqueeze(1)

    def forward(self, x, supercase_indexes):
        weights = F.sigmoid(self.weightgiver(x))
        # Apply weights to subcases
        x = x * weights
        return self.average_group_pool(x, supercase_indexes), weights


class Grouper(SharedBlock):
    class Config(SharedBlock.Config):
        version: Literal["weighted", "channel-attention"] = "weighted"
        module_name: str = "mil-grouper"

    def __init__(self, args: Config, embedding_dim: int):
        super().__init__(args)
        self.args: Grouper.Config = args
        self.embedding_dim = embedding_dim
        # different versions also need to be treated differently
        if self.args.version == "weighted":
            self.reducer = WeightedAvgPoolReducer(self.embedding_dim)
        else:
            raise NotImplementedError(f"Version {self.args.version} not implemented.")
        self.make_mtl_compatible()

    @staticmethod
    def extract_ids_from_batch(ids: list[str]):
        """
        Takes a list of ids like ["a", "a", "b"] and return a torch.Tensor([0, 0, 1])
        """
        unique_ids = list(set(ids))
        id_to_index = {id_: i for i, id_ in enumerate(unique_ids)}
        return torch.tensor([id_to_index[id_] for id_ in ids])

    def unify_bagsizes(self, bag: torch.Tensor, supercase_indices: torch.Tensor) -> torch.Tensor:
        """
        function to unify the sizes of a batch of bags to keep attention based training more efficient
        and keep the flixibility of different bag sizes. It is assumed, that all images have the same CxHxW dimensions
        MIGHT RESULT IN MULTIPLE GRADIENTS FOR ONE IMAGE!
        """
        if self.args.version != "weighted":
            _, c, h, w = bag.shape
            # get supercase infos of how many different bags there are and select the biggest one
            uniques, counts = supercase_indices.unique(return_counts=True)
            new_bag_size = counts.max()

            # get all instances of each bag
            l = []
            for u in uniques:
                idx = torch.argwhere(supercase_indices == u)
                b_tmp = bag[idx].reshape(-1, c, h, w)
                # and repeat them new_bag_size_time (complete overkill and need to be optimized)
                l.append(b_tmp.repeat(new_bag_size, 1, 1, 1)[:new_bag_size])
            unified_bag = torch.stack(l).reshape(new_bag_size * len(counts), c, h, w)  # Batch x N x hidden_dim
            return unified_bag
        else:
            return bag

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

    def forward(self, subcases: torch.Tensor, supercase_indexes: torch.Tensor):
        """
        subcases (float): (batch_size, embedding_dim)
        supercase_indexes (long): (batch_size,)
        """

        return self.reducer(subcases, supercase_indexes)
