import logging
import math
import random
from typing import (
    Literal,
    Optional,
    Callable,
    List,
    Tuple,
    Dict,
    Any,
)
import os
from zipfile import ZipFile
from shutil import unpack_archive
import requests
from pathlib import Path

from tqdm import tqdm
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import torch
from torch.utils.data import get_worker_info, Dataset
import collections
import re
import numpy as np

from .DetectionDataset import DetectionDataset
from .SemSegDataset import SemSegDataset


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def batch_concat_with_meta(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Build the list of metainfo
    meta_dicts: List[Dict] = [x for item in batch for x in item["meta"]]
    for case_dict in batch:
        if "meta" in case_dict:
            case_dict.pop("meta")

    # Build the training-relevant batch
    res = batch_concat(batch)

    # Reappend the meta info as list
    res["meta"] = meta_dicts  # type: ignore
    return res  # type: ignore


def batch_concat(batch):
    """
    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py.

    For datasets which return batches of variable size the function can be used to overwrite collate_fn
    in a dataloader.
    """
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        if elem.numel() == 0:
            tensornums = [bool(t.numel()) for t in batch]
            # assert True in tensornums, "All tensors empty for batch concat"
            if not True in tensornums:
                logging.info("All Tensors empty for batch concat :(")
            else:
                elem = batch[tensornums.index(True)]
                elem_type = type(elem)

        out = None
        if get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(sum([t.size(0) for t in batch]), *list(elem.size()[1:]))
        return torch.concat(batch, out=out)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise Exception(
                    f"default_collate: batch must contain tensors, "
                    + "numpy arrays, numbers, dicts or lists; found: {elem.dtype}"
                )

            return batch_concat([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: batch_concat([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: batch_concat([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(batch_concat(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size.")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [batch_concat(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([batch_concat(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [batch_concat(samples) for samples in transposed]

    raise TypeError(batch_concat.format(elem_type))


def download_and_extract_archive(
    target: Path,
    ds_download_link: str,
    archive_type: Literal["infer", "zip", "targz"] = "infer",
    delete_archive_after=True,
    streaming_chunk_size=8192,
) -> Path:
    """
    Given a parent folder such as root / "your_dataset" and a download link, downloads and extracts archived data.

    Returns the path to the root directory in the archive.
    """
    logging.warn(f"Deprecated, use `from torchvision.datasets.utils import download_and_extract_archive`")
    if not target.exists():
        target.mkdir()

    if archive_type == "infer":
        if ds_download_link.lower().endswith(".zip"):
            archive_type = "zip"
        elif ds_download_link.lower().endswith(".tar.gz"):
            archive_type = "targz"
        else:
            raise Exception(f"Cannot infer archive type from {ds_download_link}")

    assert target.is_dir(), f"target {target} should be a directory"
    archive_path = target / "ds_archive"
    if not archive_path.exists():
        print(f"Downloading {ds_download_link} into {archive_path}")
        with open(archive_path, "wb") as f:
            resp = requests.get(ds_download_link, stream=True)
            for chunk in tqdm(resp.iter_content(chunk_size=streaming_chunk_size)):
                f.write(chunk)

    if archive_type == "zip":
        with ZipFile(archive_path, "r") as zip_ref:
            dirs = [target / d.filename for d in zip_ref.infolist() if d.is_dir()]
            if not dirs[0].exists():
                print(f"Unzipping {archive_path} into {target}")
                zip_ref.extractall(target)
    elif archive_type == "targz":
        if len(os.listdir(target)) == 1:
            unpack_archive(archive_path, extract_dir=target, format="gztar")
    else:
        raise Exception(f"Cannot work with archive type {archive_type}")

    if delete_archive_after:
        print(f"Removing the archive {archive_path}")
        os.remove(archive_path)

    print(f"Preparing data done")
    return target


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


def semseg_from_detect_ds(ds: DetectionDataset):
    """
    Converts detection bboxed to a segmentation map of size HxW
    """

    def convert_detectcase_to_semseg(detectcase: Dict[str, Any]) -> Dict[str, Any]:
        assert "image" in detectcase and "boxes" in detectcase and "labels" in detectcase

        mask = torch.zeros(detectcase["image"].shape)

        for idx, box in enumerate(detectcase["boxes"].tolist()):
            mask[box[0] : box[2], box[1] : box[3]] = detectcase["label"][idx]

        return {"image": detectcase["image"], "label": mask.long()}

    return SemSegDataset(
        src_ds=ds,
        src_transforms=convert_detectcase_to_semseg,
        class_names=ds.vis_classes,
    )
