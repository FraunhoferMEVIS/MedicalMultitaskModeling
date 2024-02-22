import os
import torch
from torch.utils.data import Dataset, DataLoader

from mmm.torch_ext import SubCaseDataset, CachingSubCaseDS


class SuperCaseDataset(Dataset):
    def __init__(self) -> None:
        self.supercases = [(0, 1, 2, 3), (4, 5, 6), (7,), (8, 9)]

    def __len__(self):
        return len(self.supercases)

    def __getitem__(self, index: int):
        return self.supercases[index]


def test_cached_subcase_ds_multiproc():
    ds = SuperCaseDataset()
    subcaseds = CachingSubCaseDS(
        ds,
        lambda supercase: [x for x in supercase],
        CachingSubCaseDS.Config(subcase_cache_size=2),
    )
    dl = DataLoader(subcaseds, batch_size=2, num_workers=3)

    batches = [x for x in dl]
    res = list(torch.concat(batches))  # flat list of subcases
    assert len(set(res)) == len(res) == 10

    # Test that a dataset correctly resets its state
    batches = [x for x in dl]
    res = list(torch.concat(batches))  # flat list of subcases
    assert len(set(res)) == len(res) == 10


def test_cached_subcase_ds():
    ds = SuperCaseDataset()
    subcaseds = CachingSubCaseDS(
        ds,
        lambda supercase: [x for x in supercase],
        CachingSubCaseDS.Config(subcase_cache_size=2),
    )
    dl = DataLoader(subcaseds, batch_size=3)

    batches = [x for x in dl]
    res = list(torch.concat(batches))  # flat list of subcases
    assert len(set(res)) == len(res) == 10


def test_expanded_ds(tmp_path):
    class SrcData(Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, i):
            return [f"{x}" for x in range(i + 1)]

    os.environ["ML_DATA_CACHE"] = str(tmp_path)
    srcdata = SrcData()

    targetdata = SubCaseDataset(
        srcdata,
        fn_length_of_case=lambda case: len(case),
        fn_extract_case_by_index=lambda case, i: case[i],
        cache_foldername="test_expanded_ds",
    )

    assert len([targetdata[i] for i in range(len(targetdata))]) == 6
