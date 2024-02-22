import numpy as np

from mmm.transforms import batchify


def test_batchify_shape():
    testinput = [np.random.random((3, 32, 32)) for _ in range(10)]
    test_batch = batchify(testinput)
    assert test_batch.shape == (10, 3, 32, 32)


def test_batchify_empty():
    testinput = []
    test_batch = batchify(testinput)
    assert np.array_equal(np.array(testinput), test_batch)
