import pytest
import torch
from mmm.mtl_modules.shared_blocks.Grouper import Grouper


@pytest.fixture(
    ids=["unified", "diverse"],
    params=[
        (
            torch.Tensor(
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
            ),
            ["1", "1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "3", "4", "4", "4", "4"],
        ),
        (
            torch.Tensor(
                [
                    [1, 2, 1],  # 1
                    [1, 2, 2],  # 1
                    [1, 2, 3],  # 1
                    [1, 2, 4],  # 1
                    [2, 10, 2],  # 2
                    [3, 7, 1],  # 3
                    [3, 7, 1],  # 3
                    [4, 12, 2],  # 4
                    [4, 5, 3],  # 4
                    [3, 4, 5],  # 4
                ]
            ),
            ["1", "1", "1", "1", "2", "3", "3", "4", "4", "4"],
        ),
    ],
)
def pseudo_bag(request) -> str:
    return request.param


def test_pooling_reducer(pseudo_bag: tuple, grouper_args: Grouper.Config):
    bag, ids = pseudo_bag

    grouper: Grouper = Grouper(embedding_dim=3, args=grouper_args)
    supercase_indexes = grouper.extract_ids_from_batch(ids)

    reduced, weights = grouper(bag, supercase_indexes)

    assert reduced.shape == torch.Size(
        [4, 3]
    ), f"Failed {grouper_args.version}-grouper and {grouper_args.attention_heads} weights"
    assert weights is not None, f"Failed {grouper_args.version}-grouper and {grouper_args.attention_heads} weights"
