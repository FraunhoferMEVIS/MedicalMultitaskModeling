from typing import Tuple
import numpy as np

import torch
from torch.utils.data import DataLoader

from mmm.data_loading.utils import batch_concat
from torch.utils.data import Dataset


class BatchMockupDataset(Dataset):
    """
    Dataset which yields full batches of (image data, classification labels).
    """

    def __init__(self, N: int) -> None:
        super().__init__()
        self.N = N

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.ones(index, 3, 32, 32) * index,
            torch.ones(index + 1).long() * index,
        )


def test_batch_concat():
    """
    Tests if the default behaviour of PyTorch Dataloaders is consistent
    with our collate_fn implementation `batch_concat`.

    Tests simple datasets which already yield correct Tensors in the correct format.
    """
    ds = BatchMockupDataset(16)

    # Build target manually
    x_batch = torch.concat([ds[i][0] for i in range(len(ds))])
    y_batch = torch.concat([ds[i][1] for i in range(len(ds))])
    assert x_batch.size() == (120, 3, 32, 32)

    # ds contains [1, 2, 2, 3, 3, 3, 4, ...], so zero-indexed item 6 contains threes
    slice_from_third_element = x_batch[6].numpy()
    values = list(np.unique(slice_from_third_element))
    assert len(values) == 1 and values[0] == 4.0

    # Build target using Dataloader
    loader = DataLoader(ds, batch_size=16, collate_fn=batch_concat)
    single_process_x, single_process_y = next(iter(loader))
    assert torch.equal(single_process_x, x_batch) and torch.equal(single_process_y, y_batch)

    # Build target using multiprocessing dataloader, which uses an extra branch for shared memory allocation
    # Does not yet check if all workers really do good stuff but only the first
    loader = DataLoader(ds, batch_size=16, num_workers=2, collate_fn=batch_concat)
    multi_process_x, multi_process_y = next(iter(loader))
    assert torch.equal(multi_process_x, x_batch) and torch.equal(multi_process_y, y_batch)


def test_batch_concat_with_multiple_batches():
    dslarge = BatchMockupDataset(32)
    ite = iter(DataLoader(dslarge, batch_size=16, collate_fn=batch_concat))
    multi_iter = iter(DataLoader(dslarge, batch_size=16, num_workers=2, collate_fn=batch_concat))
    first_single = ite.__next__()
    first_multi = multi_iter.__next__()
    assert torch.equal(first_single[0], first_multi[0])
    second_single = next(ite)
    second_multi = next(multi_iter)
    assert torch.equal(second_single[0], second_multi[0])
    assert not torch.equal(second_multi[0], first_multi[0])
