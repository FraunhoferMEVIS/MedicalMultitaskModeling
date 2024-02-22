from pathlib import Path
import numpy as np
import torch
import hashlib
import os
import time

from mmm.utils import recursive_equality, disk_cacher
from mmm.data_loading.utils import train_val_split_class_dependent


def test_recursive_equality():
    assert not recursive_equality(4, "4")
    assert recursive_equality({"t": torch.Tensor([1, 2])}, {"t": torch.Tensor([1, 2])})
    assert not recursive_equality({"t": torch.Tensor([1, 2])}, {"b": torch.Tensor([1, 2])})
    assert not recursive_equality({"t": torch.Tensor([1, 2])}, {"t": torch.Tensor([2, 1])})


def test_train_val_split_class_dependent():
    indices = np.arange(100)
    classes = np.random.randint(1, 7 + 1, 100)

    train_idx, val_idx = train_val_split_class_dependent(
        indices=indices, classes=classes, perc=0.5, allow_imbalance=True
    )

    # No val indices in train and lengths plausible:
    assert len(set(train_idx) - set(val_idx)) == len(set(train_idx)) == 50 == len(np.unique(train_idx))

    train_idx, val_idx = train_val_split_class_dependent(indices=indices, classes=classes, perc=0.5)
    unique_train_classes, counts = np.unique(classes[train_idx], return_counts=True)
    overall_classes = np.unique(classes)

    assert (unique_train_classes == overall_classes).all()
    assert all(element == counts[0] for element in counts)

    # Test deterministic split
    train_idx_one, val_idx_one = train_val_split_class_dependent(
        indices=indices,
        classes=classes,
        perc=0.5,
        allow_imbalance=False,
        deterministic_seed=42,
    )

    train_idx_two, val_idx_two = train_val_split_class_dependent(
        indices=indices,
        classes=classes,
        perc=0.5,
        allow_imbalance=False,
        deterministic_seed=42,
    )

    assert all(train_idx_one[i] == train_idx_two[i] for i in range(len(train_idx_one)))
    assert all(val_idx_one[i] == val_idx_two[i] for i in range(len(val_idx_one)))

    train_idx_two, val_idx_two = train_val_split_class_dependent(
        indices=indices,
        classes=classes,
        perc=0.5,
        allow_imbalance=False,
        deterministic_seed=9001,
    )

    assert not all(train_idx_one[i] == train_idx_two[i] for i in range(len(train_idx_one)))
    assert not all(val_idx_one[i] == val_idx_two[i] for i in range(len(val_idx_one)))


def test_disk_cacher_emptyfuncs(tmp_path: Path):
    # show that tmp_path is a fresh directory for readability of the test:
    assert tmp_path.is_dir() and len(os.listdir(tmp_path)) == 0

    @disk_cacher(tmp_path)
    def helper_function1():
        return 1

    helper_function1()

    @disk_cacher(tmp_path)
    def helper_function2():
        return 2

    helper_function2()

    assert helper_function1() != helper_function2()


def test_disk_cacher_uniqueness(tmp_path):
    # show that tmp_path is a fresh directory for readability of the test:
    assert tmp_path.is_dir() and len(os.listdir(tmp_path)) == 0

    @disk_cacher(tmp_path)
    def helper_function(number1, number2):
        return number1**number2

    o3 = helper_function(3, 4)
    o4 = helper_function(4, 3)
    assert o3 != o4
    helper_function(3, 4)
    helper_function(4, 3)
    assert len(list(tmp_path.glob("**/*.pkl"))) == 2
    assert helper_function(3, 4) != helper_function(4, 3)


def test_disk_cacher_by_loadingtimes(tmp_path):
    assert tmp_path.is_dir() and len(os.listdir(tmp_path)) == 0

    @disk_cacher(tmp_path)
    def helper_function(number):
        time.sleep(3)
        return number**number

    inp = 5
    start = time.time()
    _ = helper_function(inp)
    end = time.time()
    first_run = end - start

    start = time.time()
    _ = helper_function(inp)
    end = time.time()
    second_run = end - start

    assert second_run < first_run
    assert first_run > 2.0
