from mmm.mmm_types.GroupUsage import GroupUsage, GroupingStrategy, IncompatibleUsageError, MaskingStrategy
from mmm.neural.modules.attention import DropRandomMHAMask, create_multihead_padding_mask
from mmm.utils import build_batch_of_sequences
import pytest
import torch
from mmm.mtl_modules.shared_blocks.Grouper import (
    Grouper,
    WeightedAvgPoolReducer,
    AttentionPoolingReducer,
    MHAReducer,
)


@pytest.fixture(
    ids=[
        # "weighted",
        # "attention",
        # "attention-4",
        "mha-1-1",
        "mha-2-2",
    ],
    params=[
        # Grouper.Config(reducer=WeightedAvgPoolReducer.Config()),
        # Grouper.Config(reducer=AttentionPoolingReducer.Config(num_heads=1)),
        # Grouper.Config(reducer=AttentionPoolingReducer.Config(num_heads=4)),
        Grouper.Config(reducer=MHAReducer.Config(num_layers=1, num_heads=1)),
        Grouper.Config(reducer=MHAReducer.Config(num_layers=2, num_heads=2)),
    ],
)
def grouper_args_usage(request):
    return request.param


@pytest.fixture(
    ids=["single", "mixed", "full"],
    params=[
        GroupingStrategy.single,
        GroupingStrategy.mixed,
        GroupingStrategy.full,
    ],
)
def group_strategy(request):
    return request.param


@pytest.fixture(
    ids=["unified", "diverse", "unsorted"],
    params=[
        (
            torch.Tensor(
                [
                    [1, 2, 1, 1],
                    [1, 2, 2, 2],
                    [1, 2, 3, 3],
                    [1, 2, 4, 4],
                    [2, 1, 1, 1],
                    [2, 10, 2, 2],
                    [2, 1, 1, 1],
                    [2, 10, 2, 2],
                    [3, 7, 1, 1],
                    [3, 7, 1, 1],
                    [3, 7, 1, 1],
                    [3, 7, 1, 1],
                    [4, 12, 2, 2],
                    [4, 5, 3, 3],
                    [3, 4, 5, 5],
                    [3, 4, 5, 5],
                    [3, 8, 1, 5],
                ]
            ),
            [
                "1",
                "1",
                "1",
                "1",
                "2",
                "2",
                "2",
                "2",
                "3",
                "3",
                "3",
                "3",
                "4",
                "4",
                "4",
                "4",
                "4",
            ],
        ),
        (
            torch.Tensor(
                [
                    [1, 2, 1, 1],
                    [1, 2, 2, 2],
                    [1, 2, 3, 3],
                    [1, 2, 4, 4],
                    [2, 10, 2, 2],
                    [3, 7, 1, 1],
                    [3, 7, 1, 1],
                    [4, 12, 2, 2],
                    [4, 5, 3, 3],
                    [3, 4, 5, 5],
                ]
            ),
            ["1", "1", "1", "1", "2", "3", "3", "4", "4", "4"],
        ),
        (
            torch.Tensor(
                [
                    [0, 1, 2, 1],
                    [0, 1, 2, 2],
                    [0, 1, 2, 3],
                    [0, 1, 2, 4],
                    [0, 2, 1, 1],
                    [0, 2, 10, 2],
                    [0, 2, 1, 1],
                    [0, 2, 10, 2],
                ]
            ),
            ["2", "1", "3", "1", "2", "2", "2", "2"],
        ),
    ],
)
def pseudo_bag(request) -> tuple[torch.Tensor, list[str]]:
    return request.param


def test_pooling_reducer(pseudo_bag: tuple, grouper_args_usage: Grouper.Config, group_strategy: GroupingStrategy):
    bag, ids = pseudo_bag
    grouper_args_pooling = grouper_args_usage
    group_usage = GroupUsage(grouping=group_strategy)

    # For all groupers
    EMBEDDING_DIM = 4
    grouper: Grouper = Grouper(embedding_dim=EMBEDDING_DIM, args=grouper_args_pooling)
    try:
        grouper.raise_for_usageerror(group_usage)
    except IncompatibleUsageError:
        pytest.skip(f"Skipping {grouper_args_pooling.reducer.grouper_type} with {group_strategy=}")
    supercase_indexes = grouper.extract_ids_from_batch(ids)

    reduced, weights = grouper(bag, supercase_indexes, group_usage)

    if group_usage.grouping is GroupingStrategy.single:
        assert reduced.shape[0] == len(set(ids))
    elif group_usage.grouping is GroupingStrategy.mixed:
        assert reduced.shape[0] >= len(set(ids))
    elif group_usage.grouping is GroupingStrategy.full:
        assert reduced.shape[0] == bag.shape[0]

    assert reduced.shape[1] == bag.shape[1]


def test_mha_reducer_nonan():
    """
    whenever padded tokens are fully masked (full row of -inf in the attention bias), nans occur.
    """
    mha_reducer = MHAReducer(
        MHAReducer.Config(
            num_heads=2,
            dropout=0,
            redundancy_masks=(
                DropRandomMHAMask(drop_prob=1.0),
                DropRandomMHAMask(drop_prob=1.0),
            ),
        ),
        embedding_dim=4,
    )
    out, weights = mha_reducer.forward(
        torch.rand(3, 4),
        supercase_indices=torch.tensor([0, 0, 1]),
        group_usage=GroupUsage(
            grouper_key="grouper",
            masking=MaskingStrategy.redundancy,
            grouping=GroupingStrategy.full,
        ),
        positions=None,
    )
    assert not torch.isnan(out).any(), "Output contains NaNs"


def test_mha_reducer_padding():
    """
    Tests that changing the padding tokens does not lead to changes in the non-padding tokens.
    """
    mha_reducer = MHAReducer(
        MHAReducer.Config(
            num_heads=2,
            dropout=0,
            redundancy_masks=(
                DropRandomMHAMask(drop_prob=0.0),
                DropRandomMHAMask(drop_prob=0.0),
            ),
        ),
        embedding_dim=4,
    )

    mha_inputs, src_key_padding_mask = build_batch_of_sequences(
        instances := torch.rand(3, 4), supercases := torch.tensor([0, 1, 1])
    )
    # Second sequence is padded
    assert torch.all(src_key_padding_mask == torch.tensor([[False, True], [False, False]]))
    mha_inputs_different_pad = mha_inputs.clone()
    mha_inputs_different_pad[0, 1, :] = torch.rand(4)
    attn_bias = mha_reducer.get_attn_bias(
        mha_inputs=mha_inputs,
        src_key_padding_mask=src_key_padding_mask,
        supercase_indices=supercases,
        group_usage=GroupUsage(
            grouper_key="grouper",
            masking=MaskingStrategy.redundancy,
            grouping=GroupingStrategy.full,
        ),
    )
    outputs1, attn1, lattn1 = mha_reducer.forward_model(mha_inputs, attn_bias)
    outputs2, attn2, lattn2 = mha_reducer.forward_model(mha_inputs_different_pad, attn_bias)
    assert torch.allclose(attn1, attn2)
    assert torch.allclose(lattn1, lattn2)
    assert torch.allclose(outputs1[0, 0, :], outputs2[0, 0, :])
