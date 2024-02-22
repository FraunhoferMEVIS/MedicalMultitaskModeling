import pytest
import torch
from mmm.mtl_modules.shared_blocks.Grouper import Grouper


def get_pseudo_bag():
    return torch.Tensor(
        [
            [1, 2, 1],
            [1, 2, 2],
            [1, 2, 3],
            [1, 2, 4],
            [2, 1, 1],
            [2, 10, 2],
            [2, 1, 1],
            [2, 10, 2],
            [3, 7, 1],
            [3, 7, 1],
            [3, 7, 1],
            [3, 7, 1],
            [4, 12, 2],
            [4, 5, 3],
            [3, 4, 5],
            [3, 4, 5],
        ]
    ), ["1", "1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "3", "4", "4", "4", "4"]


def get_pseudo_bag_images():
    test = torch.ones((4, 3, 224, 224))
    for i in range(4):
        test[i] += i

    indexes = torch.LongTensor([0, 0, 0, 1])

    return test, indexes


def test_weightedavgpool_reducer():
    bag, ids = get_pseudo_bag()

    grouper: Grouper = Grouper(embedding_dim=3, args=Grouper.Config(version="weighted"))

    supercase_indexes = grouper.extract_ids_from_batch(ids)

    reduced, weights = grouper(bag, supercase_indexes)

    assert reduced.shape == torch.Size([4, 3])
    assert weights is not None
