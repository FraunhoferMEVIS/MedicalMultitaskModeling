"""
Utilities that extend the PyTorch types, not relying on our MTL extensions such as the MTLDataset
"""

from __future__ import annotations

import logging
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Generic, Iterator, List, Optional, Sized, Tuple, TypeVar, Union, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ChainDataset, DataLoader, Dataset, IterableDataset, Subset, get_worker_info
from tqdm.auto import tqdm

from mmm.BaseModel import BaseModel
from mmm.data_loading.MTLDataset import MTLDataset
from mmm.utils import get_default_cachepath

T = TypeVar("T")


def replace_childen_recursive(m: nn.Module, layertype_to_replace, newlayer_constructor):
    """
    newlayer_constructor gets an instance of the layer to be replaced.

    Use cases might be replacing all 2D layers by their respective 3D versions.
    """
    for k, layer in m.named_children():
        # Replace the object in question
        if isinstance(layer, layertype_to_replace):
            newlayer = newlayer_constructor(layer)
            # Does not work for children's children, which is why this is a recursive function
            setattr(m, k, newlayer)

        replace_childen_recursive(layer, layertype_to_replace, newlayer_constructor)


def get_random_ds_sample(ds: Dataset[T], subset_size: Union[float, int]) -> Dataset[T]:
    """
    Generates a random subset of a given dataset.

    On the new object, only the default methods will work.
    """
    original_dataset_len = len(ds)  # type: ignore

    if isinstance(subset_size, float):
        subset_size = math.floor(original_dataset_len * subset_size)

    new_indices = random.sample(range(original_dataset_len), subset_size)

    return Subset(ds, new_indices)


SuperCaseType = TypeVar("SuperCaseType")
SubCaseType = TypeVar("SubCaseType")


class CachingSubCaseDSSampler:
    def __init__(self):
        # Once this sampler is assigned to a CachingSubCaseDS, it will set this attribute
        # In consequence, one sampler can only be used for one CachingSubCaseDS
        self.cacheds: CachingSubCaseDS = None

    def prepare_supercase_indices(self, supercase_indices: list[int], worker_id: int | None) -> list[int]:
        """
        By default, the supercases are shuffled
        """
        random.shuffle(supercase_indices)
        return supercase_indices

    def decide_removal(self, popped_case: SubCaseType, draining_phase: bool, index) -> bool:
        """
        By default, a case is removed whenever it is yielded
        """
        return True

    def sample_from_cache(self, draining_phase: bool) -> int:
        """
        By default, a random case is sampled from the cache
        """
        return random.randint(0, len(self.cacheds.subcases) - 1)

    def hook_new_subcases(self, subcases: list[SubCaseType]):
        """
        If the sampler keeps track of the subcases, it can update its internal state here.

        It also needs to return the subcases that should be added to the cache.
        """
        return subcases

    def postprocess_subcase(self, subcase: SubCaseType) -> SubCaseType:
        """
        If the sampler changes the expected type of the subcase, here is the place to change it back.
        """
        return subcase


class DeterministicSampler(CachingSubCaseDSSampler):
    """
    This disables randomization of the data.
    """

    def prepare_supercase_indices(self, idxs: list[int], worker_id) -> list[int]:
        return idxs

    def sample_from_cache(self, draining_phase: bool) -> int:
        return 0  # Always use the first case


class DeterministicSamplerSignalLast(DeterministicSampler):
    def hook_new_subcases(self, subcases: list):
        subcases = list(subcases)
        assert "sampler_last_subcase" not in subcases[-1]["meta"]
        subcases[-1]["meta"]["sampler_last_subcase"] = True
        return super().hook_new_subcases(subcases)


class CachingSubCaseDS(IterableDataset, Generic[SubCaseType]):
    """
    Holds `cache_size` subcases in a cache for each worker.
    The cache is refilled with subcases once a new supercase fits into cache.

    Each supercase is assigned one worker, try to keep the number of workers low.
    In consequence, there is a sampling bias.
    For example, if you have 12 supercases and 4 workers, each worker can construct batches from at most 3 supercases.

    At the end of the loop all subcases are drained, resulting in a loading time at the start of each epoch.
    """

    class Config(BaseModel):
        drain_each_epoch: bool = True
        subcase_cache_size: int = 100
        split_across_workers: bool = True

    def __init__(
        self,
        supercase_ds: Dataset[SuperCaseType],
        supercase_loader: Callable[[SuperCaseType], List[SubCaseType]],
        cfg: Config,
        cache_sampler: CachingSubCaseDSSampler | None = None,
    ) -> None:
        self.supercase_ds, self.cfg, self.supercase_loader = (
            supercase_ds,
            cfg,
            supercase_loader,
        )
        self.subcases = []

        if cache_sampler is None:
            self.cache_sampler: CachingSubCaseDSSampler = CachingSubCaseDSSampler()
        else:
            self.cache_sampler: CachingSubCaseDSSampler = cache_sampler
        assert self.cache_sampler.cacheds is None, "The sampler is already assigned to a CachingSubCaseDS"
        self.cache_sampler.cacheds = self

    def _yield_sample(self, draining_phase: bool):
        # index = random.randint(0, len(self.subcases) - 1)
        index = self.cache_sampler.sample_from_cache(draining_phase)
        subcase = self.subcases[index]
        if self.cache_sampler.decide_removal(subcase, draining_phase, index):
            self.subcases.pop(index)
        else:
            subcase = deepcopy(subcase)  # Prevent consumers from modifying the original
        return self.cache_sampler.postprocess_subcase(subcase)

    def add_subcases(self, subcases: List[SubCaseType]):
        self.subcases.extend(self.cache_sampler.hook_new_subcases(subcases))

    def __iter__(self) -> Iterator[SubCaseType]:
        worker_info = get_worker_info()

        # First move: find out which supercases this worker should process
        if worker_info is None or not self.cfg.split_across_workers:
            supercase_indices: List[int] = list(range(len(self.supercase_ds)))  # type: ignore
            num_workers = 1  # used for cache size calculation
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            supercase_indices: List[int] = list(
                range(
                    math.ceil(len(self.supercase_ds) / num_workers) * worker_id,  # type: ignore
                    min(
                        len(self.supercase_ds),  # type: ignore
                        math.ceil(len(self.supercase_ds) / num_workers) * (worker_id + 1),  # type: ignore
                    ),
                )
            )
            logging.debug(f"Worker {worker_id} got {len(supercase_indices)} supercases.")

            if not supercase_indices:
                logging.warning(f"Worker {worker_info} had no supercases in {self}")
                return
        self.cache_sampler.prepare_supercase_indices(
            supercase_indices, worker_info.id if worker_info is not None else None
        )
        filling_phase = True

        while filling_phase:
            for supercase_index in supercase_indices:
                supercase = self.supercase_ds[supercase_index]
                self.add_subcases(self.supercase_loader(supercase))
                if len(self.subcases) >= self.cfg.subcase_cache_size:
                    filling_phase = False

                if not filling_phase:
                    # Only yield samples if the cache is pretty full to increase diversity
                    while len(self.subcases) >= max(1, (self.cfg.subcase_cache_size // num_workers)):
                        yield self._yield_sample(draining_phase=False)
            if filling_phase and len(self.subcases) < self.cfg.subcase_cache_size:
                logging.warning(f"Dataset {self=} smaller than cache size {self.cfg.subcase_cache_size}")

        if self.cfg.drain_each_epoch:
            # No more supercases to load, yield the remaining cases:
            # We might also skip this to keep the cache full to reduce the next epoch's startup time
            while self.subcases:
                yield self._yield_sample(draining_phase=True)


class SubCaseDataset(Dataset, Generic[SubCaseType]):
    """
    Create a new dataset from a dataset which holds cases which itself hold cases.

    Common use-case: creating a 2D dataset from slices from a 3D dataset.
    In this case, fn_length_of_case might be used to determine the number of slices.
    fn_extract_case_by_index gets the slice index and returns a 2D slice.

    For determining the length of the dataset,
    the user needs to provide a function which this object applies to each supercase.
    """

    def __init__(
        self,
        src_ds: Dataset[SuperCaseType],
        fn_length_of_case: Callable[[SuperCaseType], int],
        fn_extract_case_by_index: Callable[[SuperCaseType, int], SubCaseType],
        cache_foldername: str,
        subcase_transform: Optional[Callable[[SubCaseType], SubCaseType]] = None,
    ) -> None:
        super().__init__()
        self.src_ds = src_ds
        self.fn_length_of_case = fn_length_of_case
        self.fn_extract_case_by_index = fn_extract_case_by_index
        self.cache_path = get_default_cachepath(folder_name=cache_foldername) / "sizes.pkl"
        self.transform = subcase_transform

        self.src_case_to_target_case_map: Dict[int, int] = {}
        self.first_index_of_case = {}

        if not Path(os.getenv("ML_DATA_CACHE", default="/dl_cache/")).exists():
            os.mkdir(Path(os.getenv("ML_DATA_CACHE", default="/dl_cache/")))

        if self.cache_path is not None and not get_default_cachepath(folder_name=cache_foldername).exists():
            os.mkdir(get_default_cachepath(folder_name=cache_foldername))

        if self.cache_path is not None and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self.src_case_to_target_case_map, self.first_index_of_case = torch.load(f)

        else:
            # Supercase stats cache needs to be recomputed
            for case_id in tqdm(range(len(cast(Sized, src_ds)))):
                case = src_ds[case_id]
                self.first_index_of_case[case_id] = len(self.src_case_to_target_case_map)
                for _ in range(fn_length_of_case(case)):
                    self.src_case_to_target_case_map[len(self.src_case_to_target_case_map)] = case_id

            if self.cache_path is not None:
                with open(self.cache_path, "wb") as f:
                    torch.save((self.src_case_to_target_case_map, self.first_index_of_case), f)

    def __len__(self) -> int:
        return len(self.src_case_to_target_case_map)

    def __getitem__(self, index: int) -> SubCaseType:
        case_id = self.src_case_to_target_case_map[index]
        case = self.src_ds[case_id]
        case_index = index - self.first_index_of_case[case_id]
        res: SubCaseType = self.fn_extract_case_by_index(case, case_index)

        if self.transform is not None:
            res = self.transform(res)

        return res


def transform_dataloader(dataloader: DataLoader, transform: Callable):
    transformed_batches = []
    for batch in dataloader:
        # The user might want to use shared blocks to process the raw data
        with torch.inference_mode():
            batch_transformed = transform(batch)
        transformed_batches.append(batch_transformed)
    return transformed_batches


def infer_stride_channels_from_features(
    features: List[torch.Tensor],
) -> Tuple[List[int], List[int]]:
    """
    Assumes the first feature map to be the raw input
    """
    channels = [v.shape[1] for v in features]
    strides = [features[0].shape[2] // v.shape[2] for v in features]
    return channels, strides


class IterableDatasetWrapper(IterableDataset):
    def __init__(self, ds: MTLDataset) -> None:
        super().__init__()
        self.ds: MTLDataset = ds

    def __iter__(self):
        """
        Yield next item from ds
        """
        if hasattr(self.ds, "_indices"):
            for idx in np.random.choice(np.arange(len(self.ds)), len(self.ds), replace=False):
                yield self.ds[idx]
        else:
            for item in self.ds:
                yield item


class CombinedDataset(IterableDataset):
    def __init__(self, mtl_datasets: List[MTLDataset]) -> None:
        super().__init__()
        self.mtl_datasets = mtl_datasets
        logging.info(f"received {len(mtl_datasets)=} and will concatenate them now")
        self.data_set = ChainDataset(self._prepare_datasets([ds for ds in self.mtl_datasets]))

    def _prepare_datasets(self, datasets):
        collection = []
        for data in datasets:
            data.batch_transform = None
            data = IterableDatasetWrapper(data)
            collection.append(data)
        return collection

    def _get_rand_ds_idx(self, num):
        return np.random.choice(np.arange(num))

    def __iter__(self):
        for item in self.data_set:
            yield item
        # tmp_ds = self._prepare_datasets([ds for ds in self.mtl_datasets])
        # while tmp_ds:
        #     idx = self._get_rand_ds_idx(len(tmp_ds))
        #     try:
        #         item = tmp_ds[idx]
        #         yield item
        #     except StopIteration:
        #         tmp_ds.pop(idx)


def combine_datasets(datasets: List[MTLDataset]) -> CombinedDataset:
    return CombinedDataset(datasets)
