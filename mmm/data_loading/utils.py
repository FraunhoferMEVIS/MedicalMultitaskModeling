import logging
import math
import random
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from skimage.draw import ellipse
from torch.utils.data import Dataset

np_str_obj_array_pattern = re.compile(r"[SaUO]")


class TransformedSubset(Dataset):
    """
    Transforms an existing dataset and filters using the given indices.

    Only applies indices or transform if not None.

    Known problem:
    This will use the methods of the supertype instead of your actual child type
    """

    def __init__(
        self,
        source_ds: Dataset,
        indices: Optional[List[int]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.source_ds: Dataset = source_ds
        self.transform = transform
        if not indices:
            indices = list(range(len(self.source_ds)))  # type: ignore
        self.indices = indices

    # Only invoked if the attribute was not found on the actual object!
    def __getattr__(self, attr):
        return getattr(self.source_ds, attr)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x = self.source_ds.__getitem__(self.indices[index])
        if self.transform is not None:
            x = self.transform(x)
        return x


def train_val_split_class_dependent(
    indices: List[int],
    classes: List[int],
    perc: float = 0.7,
    amount: int = 0,
    allow_imbalance: bool = False,
    deterministic_seed: int = -1,
) -> Tuple[List[int], List[int]]:
    """
    Given a list of indices and class indices, split the list into train indices and val indices.
    For train indices, make sure that all classes are represented at least once.

    Not a good use if there are 2 or 3 examples of one class and 100 of a different class.
    """
    class_names, count = np.unique(classes, return_counts=True)

    if amount > 0:
        train_len = amount
        val_len = 0
        desired_train_len = amount * len(class_names)
    else:
        min_fraction = min(count) / len(indices)
        desired_train_len = int(perc * len(indices))

        train_len = int((perc * min_fraction) * len(indices))
        val_len = int(((1 - perc) * min_fraction) * len(indices))

    train = []
    val = []
    used_idxs = []

    if deterministic_seed > 0:
        state = np.random.get_state()
        np.random.seed(deterministic_seed)

    for i in class_names:
        class_idx = [idx for idx in indices if classes[idx] == i]
        train_class_tmp = np.random.choice(class_idx, train_len, replace=False)
        for idx in train_class_tmp:
            train.append(idx)
            used_idxs.append(idx)
            class_idx.remove(idx)

        val_class_tmp = np.random.choice(class_idx, val_len, replace=False)

        for idx in val_class_tmp:
            used_idxs.append(idx)
            val.append(idx)

    unused_idx = [x for x in indices if x not in used_idxs]
    if allow_imbalance:
        while len(train) < desired_train_len:
            random_idx = np.random.choice(unused_idx, replace=False)
            train.append(random_idx)
            used_idxs.append(random_idx)
            unused_idx = [x for x in indices if x not in used_idxs]

    for rest in unused_idx:
        val.append(rest)

    if deterministic_seed > 0:
        np.random.set_state(state)

    return train, val


def train_val_split(
    indices: List[int],
    perc: float = 0.7,
    seed: Optional[int] = None,
    ensure_representativenumber_for_classes: Optional[List[int]] = None,
    min_representatives_per_class=1,
) -> Tuple[List[int], List[int]]:
    """
    Given a list of indices, splits the indices into (train_indices, val_indices).

    min_representatives_per_class is not respected perfectly.

    >>> from mmm.data_loading.utils import train_val_split
    >>> [len(split_indices) for split_indices in train_val_split(range(10), 0.51)]
    [6, 4]
    >>> train_classes = [0, 1, 2, 3, 3, 3, 3, 3]
    >>> a, b = train_val_split(list(range(len(train_classes))), perc=0.4, seed=0, ensure_representativenumber_for_classes=train_classes)
    >>> a
    [3, 0, 1, 2]
    >>> b
    [4, 5, 6, 7]
    """
    temp_random = random.Random(seed) if seed is not None else random
    unique = np.unique(np.asarray(ensure_representativenumber_for_classes))
    train_indices = temp_random.sample(indices, max(math.ceil(len(indices) * perc), len(unique)))

    if ensure_representativenumber_for_classes is not None:
        classes_pushed = []
        train_classes = [ensure_representativenumber_for_classes[i] for i in train_indices]

        for class_id in set(ensure_representativenumber_for_classes):
            # Make sure class_id is represented by train_indices
            while train_classes.count(class_id) < min_representatives_per_class:
                train_indices.append(ensure_representativenumber_for_classes.index(class_id))
                train_classes.append(class_id)
                classes_pushed.append(class_id)

                # Remove one of the overrepresented class
                overrepresented_class_id = max(set(train_classes), key=train_classes.count)
                assert (
                    overrepresented_class_id not in classes_pushed
                ), f"Impossible! Increase train perc {classes_pushed}"
                index_of_overrepresented_class_id = train_classes.index(overrepresented_class_id)
                train_indices.pop(index_of_overrepresented_class_id)
                train_classes.pop(index_of_overrepresented_class_id)

        if classes_pushed:
            logging.debug(f"Pushed validation classes into train after random sampling: {classes_pushed}")

    val_indices = set(indices) - set(train_indices)
    if not val_indices:
        val_indices = [train_indices.pop(0)]
        logging.warn(f"Not enough cases for train_val_split, reshuffled to {val_indices=}, {train_indices=}")
    return list(train_indices), list(val_indices)


def convert_detectcase_to_semseg(
    detectcase: Dict[str, Any], mask_type: Literal["box", "ellipse"] = "box"
) -> Dict[str, Any]:
    assert "image" in detectcase and "boxes" in detectcase and "labels" in detectcase

    mask = np.zeros(detectcase["image"].shape[1:], dtype=np.int64)  # H, W

    for idx, box in enumerate(detectcase["boxes"].long().tolist()):
        if mask_type == "box":
            mask[box[1] : box[3], box[0] : box[2]] = detectcase["labels"][idx] + 1
        elif mask_type == "ellipse":
            x0, y0, x1, y1 = box
            rr, cc = ellipse((y0 + y1) // 2, (x0 + x1) // 2, (y1 - y0) // 2, (x1 - x0) // 2, shape=mask.shape)
            mask[rr, cc] = detectcase["labels"][idx] + 1
        else:
            raise ValueError(f"Unknown mask type {mask_type}")

    detectcase["label"] = torch.from_numpy(mask).long()

    return detectcase
