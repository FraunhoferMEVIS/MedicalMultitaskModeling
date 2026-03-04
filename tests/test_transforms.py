from typing import Any
import pytest
import torch
import numpy as np

from mmm.transforms import batchify, UnifySizes, Alb
from mmm.augmentations import get_weak_default_augs


def test_unify_sizes_noneedgelen():
    unifysizes = UnifySizes(max_edge_len=None, divisable_by=8)
    testinputs = [{"image": torch.rand(3, e, e)} for e in [31, 32, 28]]
    testoutputs = unifysizes(testinputs)
    for output in testoutputs:
        assert output["image"].shape == (3, 32, 32)


@pytest.mark.parametrize(
    "imageshape, expected, maxedge",
    [
        ((3, 31, 31), (3, 32, 32), 32),
        ((3, 31, 63), (3, 16, 32), 32),
        ((3, 33, 33), (3, 32, 32), 32),
        ((3, 31, 28), (3, 32, 32), 128),  # maxedge larger than image size, should not upscale
    ],
)
def test_unify_sizes(imageshape: tuple[int, int, int], expected: tuple[int, int, int], maxedge: int):
    unifysizes = UnifySizes(max_edge_len=maxedge, divisable_by=8)

    testinputs = [{"image": torch.rand(*imageshape)}]
    testoutputs = unifysizes(testinputs)
    assert testoutputs[0]["image"].shape == expected


def test_batchify_shape():
    testinput = [np.random.random((3, 32, 32)) for _ in range(10)]
    test_batch = batchify(testinput)
    assert test_batch.shape == (10, 3, 32, 32)


def test_batchify_empty():
    testinput = []
    test_batch = batchify(testinput)
    assert np.array_equal(np.array(testinput), test_batch)


@pytest.mark.parametrize(
    "mtl_image",
    [
        {"image": torch.rand(3, 64, 64)},
        {"image": torch.rand(3, 64, 64), "label": torch.randint(0, 2, (64, 64))},
        {"image": torch.rand(3, 64, 64), "masks": torch.randint(0, 2, (5, 64, 64))},
    ],
    ids=["img", "seg", "mseg"],
)
def test_alb_conversion(mtl_image: dict[str, Any]):
    mtl_image = Alb(transforms=get_weak_default_augs())(mtl_image)
