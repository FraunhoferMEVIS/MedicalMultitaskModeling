from __future__ import annotations

import logging
import os
import random
from abc import abstractmethod
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple, TypeVar

import logfire
import numpy as np
import torch
from m3_sdk.types import CompressType
from pydantic import Field
from torch.multiprocessing import Value
from torch.utils.data import DataLoader, Dataset, IterableDataset, _utils, get_worker_info
from torch.utils.data._utils.collate import default_collate

from mmm.BaseModel import BaseModel
from mmm.transforms import KeepOnlyKeysInDict
from mmm.utils import recursive_equality

# Our caching mechanism requires access to the data queue from within the dataset
_utils.worker._original_worker_loop = _utils.worker._worker_loop


def _mtl_worker_loop(dataset_kind, dataset, index_queue, data_queue, *args):
    if isinstance(dataset, MTLDataset):
        dataset._data_queue = data_queue
    elif isinstance(dataset, IterableDSWrapper) and isinstance(dataset.wrapped, MTLDataset):
        dataset.wrapped._data_queue = data_queue
    return _utils.worker._original_worker_loop(dataset_kind, dataset, index_queue, data_queue, *args)


_utils.worker._worker_loop = _mtl_worker_loop


class M3LoaderInfo:
    CACHE_PREFIX = "pretraintask"

    def __init__(self, task_name: str, batch_size: int | None, split_name: str):
        self.task_name, self.batch_size, self.split_name = task_name, batch_size, split_name

    def get_cache_key(self, suffix: str | None = None) -> str:
        return ":".join(
            [M3LoaderInfo.CACHE_PREFIX, self.task_name, self.split_name] + ([suffix] if suffix is not None else [])
        )


class InvalidCaseError(Exception):
    def __init__(self, src_ds: MTLDataset, case_id: int, user_msgs: List[str]) -> None:
        self.src_ds, self.case_id = src_ds, case_id

        findcase = "Case" if case_id == -1 else f"Case {case_id}"
        s = f"{findcase} of {src_ds} is invalid:\n"
        for user_msg in user_msgs:
            s += f"{user_msg}\n"
        super().__init__(s)


SrcCaseType = dict[str, Any] | List[dict[str, Any]]


def mtl_collate(batch: List[SrcCaseType], ignore_keys=["meta"]) -> SrcCaseType:
    """Ignores the some keys when collating, usually the meta key."""
    ignored = {
        ignore_key: [item.pop(ignore_key) if ignore_key in item else {} for item in batch] for ignore_key in ignore_keys
    }
    res = default_collate(batch)
    for ignore_key in ignore_keys:
        res[ignore_key] = ignored[ignore_key]
    return res  # type: ignore


T = TypeVar("T")


def flatten_list(l: list[list[T]]) -> list[T]:
    return [item for sublist in l for item in sublist]


def mtl_batch_collate(batch: List[List[SrcCaseType]]) -> SrcCaseType:
    """Ignores the meta key when collating datasets that already return batches"""
    return mtl_collate(flatten_list(batch))


class DatasetStyle(Enum):
    MapStyle = auto()
    IterStyle = auto()


class IterableDSWrapper(IterableDataset):
    """
    Dataloader make an isinstance check.
    IterableDSWrapper is used to pass this check for MTLDatasets that should be iterable-style.
    """

    def __init__(self, fn) -> None:
        self.wrapped = fn

    def __iter__(self):
        return self.wrapped.__iter__()

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class MakeIterable(IterableDataset[SrcCaseType]):
    """
    If caching is enabled, this will make map-style datasets iterable-style datasets
    """

    def __init__(self, ds: Dataset) -> None:
        self.ds = ds

    def __iter__(self):
        indices = list(range(len(self.ds)))  # type: ignore
        random.shuffle(indices)
        for index in indices:
            yield self.ds[index]


class CacheSettings(BaseModel):
    cache_size: int = Field(default=..., gt=0, description="Number of items kept in shared memory.")
    prepare_new_threshold: int = Field(
        default=3,
        description="If more items are already in the queue, new items will be prepared.",
    )
    prefetch_items: int = Field(
        default=5,
        description="Number of items in the prefetch queue of items that are next added to the cache.",
    )
    max_hits: None | int = Field(default=None, description="Minimum and maximum number of cache hits.")


class MTLDataset(Dataset[SrcCaseType]):
    """
    MTLDataset is a wrapper around a torch dataset which can verify cases depending on the dataset type.

    - Map-style datasets have a length and are integer-indexable.
    - src_ds -> should contain the full data for a case
    - src_transform -> converts the data into the training format and the result should be small
    - If you want to apply random transformations, use the batch_transform.
    - Avoid random transforms as part of the `src_ds` or the optional src_transform (caching would be circumvented)

    It enforces cases to be based on dictionaries where certain keys have a fixed place and meaning.
    Like "image" is usually image data scaled to [0, 1].

    - Caching works by storing cache_size cases in RAM
      - make_cachable -> sets the cache size and makes sure the dataset is iterable-style

    Implementation:

    - mainproc_queuesize communicates the number of prepared items to workers
    - mainproc_queuesize is set from outside by the Loop, which needs to have a reference to the dataloader

    """

    def set_cohort_settings(self, batch_size: int, task_name: str, split_name: str):
        self.loader_info = M3LoaderInfo(batch_size=batch_size, task_name=task_name, split_name=split_name)
        self.src_ds.mmm_info = self.loader_info  # type: ignore

    @staticmethod
    def assert_image_data_assumptions(image_data: torch.Tensor) -> None:
        assert torch.is_tensor(image_data) and type(image_data) is torch.Tensor, "image should be of type tensor"
        assert image_data.shape[0] >= 1, "Image needs to have at least one channel"
        assert isinstance(image_data, torch.FloatTensor)

    def __init__(
        self,
        src_ds: Dataset[SrcCaseType],
        src_transform: Optional[Callable[[SrcCaseType], SrcCaseType]] = None,
        batch_transform: Optional[Callable[[SrcCaseType], Any]] = None,
        collate_fn: Optional[Callable[[List[Any]], SrcCaseType]] = None,
        verify_cases: int = 0,
        cache_cfg: CacheSettings | None = None,
        **dataloader_kwargs,
    ) -> None:
        self.src_ds = src_ds
        self.loader_info: M3LoaderInfo | None = None

        # Caching
        self.cache_cfg: CacheSettings | None = cache_cfg
        self.cache = []
        self._data_queue: Queue

        self.src_transform = src_transform
        self.batch_transform: Optional[Callable] = batch_transform
        self.collate_fn = collate_fn if collate_fn is not None else mtl_collate
        self.dataloader_kwargs = dataloader_kwargs

        self.data_stripper = KeepOnlyKeysInDict(
            keys=set(self.get_mandatory_keys() + self.get_optional_keys()),
        )

        self.reduced_size: float = 1.0
        # Represented by a list because it is valid to repeat indices if the user wants so by setting a fraction > 1.
        if self.get_dataset_style() is DatasetStyle.MapStyle:
            self._indices: np.ndarray
            self.reset_indices()

            for case_id in random.sample(list(self._indices), k=min(len(self._indices), verify_cases)):
                self.verify_case_by_index(case_id)

    @staticmethod
    def get_mandatory_keys() -> list[str]:
        """
        Every case is expected to have these keys. Validation will fail if they are missing.
        """
        return []

    @staticmethod
    def get_optional_keys() -> list[str]:
        return ["meta", "tabulars"]

    @staticmethod
    def case_is_compressed(case: dict[str, Any]):
        if "meta" in case and "compress_type" in case["meta"]:
            return case["meta"]["compress_type"] != CompressType.rgbimage.value
        else:
            return len(case["image"].shape) < 3

    @staticmethod
    def batch_is_compressed(batched_image: torch.Tensor, meta: None | list[dict[str, Any]] = None) -> bool:
        """
        Returns True if the image is compressed, False otherwise.
        """
        # If the image is compressed, it should be a 1D tensor [B, C], otherwise [B, C, H, W]
        return len(batched_image.shape) < 4

    def get_dataset_style(self) -> DatasetStyle:
        # Caching makes the dataset iter-style because it loses its fixed length
        if isinstance(self.src_ds, IterableDataset):
            return DatasetStyle.IterStyle
        else:
            return DatasetStyle.MapStyle

    def worker_init_fn(self, worker_id):
        assert self.loader_info is not None, "Cohort settings must be set before initializing workers."
        logfire.DEFAULT_LOGFIRE_INSTANCE._tags += (f"ds_{self.loader_info.task_name}_{worker_id}",)
        logfire.info(
            "Initialized worker {worker_name} with ID {worker_id}, PID: {pid}",
            worker_name=self.loader_info.task_name,
            worker_id=worker_id,
            pid=os.getpid(),
        )

    def get_dataloader(self, **kwargs) -> DataLoader:
        with logfire.span("Building dataloader for {cohort_settings}", cohort_settings=self.loader_info):
            dl_kwargs = dict(
                collate_fn=self.collate_fn,
                # a whole day in seconds
                # without multiprocessing, a timeout>0 is not allowed
                timeout=(86400 if "num_workers" in kwargs and kwargs["num_workers"] > 0 else 0),
                **kwargs,
                **self.dataloader_kwargs,
            )
            if self.get_dataset_style() is DatasetStyle.MapStyle and self.cache_cfg is None:
                if "sampler" in self.dataloader_kwargs and "shuffle" in kwargs:
                    dl_kwargs.pop("shuffle")
                dl_ds = self
            else:
                if "shuffle" in dl_kwargs:
                    dl_kwargs.pop("shuffle")  # Shuffle is not a valid option for iterablestyle datasets
                dl_ds = IterableDSWrapper(self)  # The dataloader checks for isinstance(dataset, IterableDataset)
            if self.cache_cfg is not None:
                assert "prefetch_factor" not in kwargs and "prefetch_factor" not in self.dataloader_kwargs
                dl_kwargs["prefetch_factor"] = self.cache_cfg.prefetch_items

            logfire.info("Dataloader settings: {dl_kwargs}", dl_kwargs=dl_kwargs)

            # Increase the default prefetch factor for multiprocessing
            try:
                src_ds_dataloader_defaults = dl_ds.src_ds.get_dataloader_defaults()  # type: ignore
                logfire.info(
                    "{dataset_type} has dataloader defaults: {x}, applying them",
                    dataset_type=type(dl_ds.src_ds),
                    x=src_ds_dataloader_defaults,
                )
                for k, v in src_ds_dataloader_defaults.items():
                    assert (
                        k not in dl_kwargs
                    ), f"Dataset tries to set {k} to {v}, but the value is already {dl_kwargs[k]=}"
                    dl_kwargs[k] = v
            except AttributeError:
                logfire.debug("{dataset_type} has no dataloader defaults", dataset_type=type(dl_ds.src_ds))

        return DataLoader(
            dl_ds,
            worker_init_fn=partial(
                self.worker_init_fn,
            ),
            **dl_kwargs,  # type: ignore
        )

    def get_mp_batchiterator(self, get_untransformed_cases=False, batch_size=1, shuffle=True, **kwargs):
        """
        Iterates through this dataset's batches with all available workers.
        Use cases are for caching and verifying cases.
        """

        class UntransformedDS(Dataset):
            def __init__(self, src_ds: MTLDataset) -> None:
                self.src = src_ds

            def __len__(self) -> int:
                return self.src.__len__()

            def __getitem__(self, index) -> Any:
                return self.src.get_untransformed_case(index)

        workernum_available = len(os.sched_getaffinity(0))
        # Use all workers for big datasets, use only few workers for small datasets
        used_workers = min(workernum_available, max(1, len(self) // (batch_size * 5)))
        dataloader_settings = dict(
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=used_workers,
            collate_fn=self.collate_fn,
            timeout=86400,
            # Batch size might be in kwargs!
            **kwargs,
        )
        logging.info(f"Preparing batchiterator with {dataloader_settings=}")
        prepare_dataloader = DataLoader(
            UntransformedDS(self) if get_untransformed_cases else self,
            **dataloader_settings,
        )
        return prepare_dataloader

    def reset_indices(self) -> None:
        if self.reduced_size < 1.0:
            logging.warning(f"Resetting indices of {self} to full size")
        self.reduced_size = 1.0
        if not isinstance(self.src_ds, IterableDataset):
            self._indices = np.array(list(range(len(self.src_ds))))  # type: ignore
        else:
            from mmm.torch_ext import CachingSubCaseDS

            if hasattr(self.src_ds, "supercase_ds"):
                self.src_ds.supercase_ds = self.fullcachingsubcasedssrc
            else:
                raise NotImplementedError

    def set_indices_by_fraction(self, fraction: float, seed: int = 13) -> None:
        """
        Starts by adding one example per class, then draws random samples until the fraction criterion is met.
        """
        assert fraction > 0.0 and fraction <= 1.0
        from mmm.data_loading.utils import TransformedSubset, train_val_split
        from mmm.torch_ext import CachingSubCaseDS

        self.reduced_size = fraction

        if self.get_dataset_style() is DatasetStyle.MapStyle:
            self.reset_indices()
            new_indices, _ = train_val_split(list(range(len(self.src_ds))), perc=fraction, seed=seed)  # type: ignore
            self._indices = np.array(new_indices)
        else:
            # In general, this is not possible. However, for some special case we can do it
            if hasattr(self.src_ds, "supercase_ds"):
                self.src_ds: CachingSubCaseDS
                self.fullcachingsubcasedssrc = self.src_ds.supercase_ds
                new_indices, _ = train_val_split(list(range(len(self.src_ds.supercase_ds))), perc=fraction, seed=seed)  # type: ignore
                self.src_ds.supercase_ds = TransformedSubset(self.src_ds.supercase_ds, indices=new_indices)
            else:
                raise NotImplementedError

    def verify_all_cases(self):
        from tqdm.auto import tqdm

        for case_index in tqdm(range(len(self))):
            self.verify_case_by_index(case_index)

    def verify_case_by_index(self, index: int) -> SrcCaseType:
        d = self.get_untransformed_case(index)
        self.verify_case(d) if isinstance(d, dict) else [self.verify_case(x) for x in d]
        return d

    def verify_case(self, d: SrcCaseType) -> None:
        missing_keys = [k for k in self.get_mandatory_keys() if k not in d]
        if missing_keys:
            raise InvalidCaseError(self, -1, [f"{k} missing in case {d}" for k in missing_keys])

    def __len__(self) -> int:
        return len(self._indices)

    @staticmethod
    def apply_to_list_or_dict(f, inps):
        if isinstance(inps, dict):
            return f(inps)
        else:
            return [f(inp) for inp in inps]

    def __iter__(self, apply_batchtransform: bool = True) -> Iterator[SrcCaseType]:
        if self.cache_cfg is not None:
            return self.iter_caching(apply_batchtransform=apply_batchtransform)
        else:
            return self.iter_no_caching(apply_batchtransform=apply_batchtransform)

    def iter_no_caching(self, apply_batchtransform: bool = True) -> Iterator[SrcCaseType]:
        for d in iter(self.src_ds):
            if self.src_transform:
                d = self.src_transform(d)

            d = self.apply_to_list_or_dict(self.data_stripper, d)

            if self.batch_transform is not None and apply_batchtransform:
                d = self.batch_transform(d)

            yield d

    def iter_caching(self, apply_batchtransform: bool = True) -> Iterator[SrcCaseType]:
        assert self.cache_cfg is not None, "Iteration is only supported for datasets with caching enabled"

        def return_item(cache_item: dict):
            d = cache_item["original"]
            d = deepcopy(d)

            # Rewrite the group ids to be unique in case of multiple cache hits of the same item in one batch
            if isinstance(d, dict) and "meta" in d and "group_id" in d["meta"]:
                d["meta"]["group_id"] = f'{d["meta"]["group_id"]}_{cache_item["cachehits"]}'
            elif isinstance(d, list):
                for single_case in d:
                    assert "meta" in single_case and "group_id" in single_case["meta"]
                    single_case["meta"]["group_id"] = f'{single_case["meta"]["group_id"]}_{cache_item["cachehits"]}'
            # Otherwise, probably no Grouper is used

            if self.batch_transform is not None and apply_batchtransform:
                d = self.batch_transform(d)
            return d

        for d in MakeIterable(self.src_ds):
            with logfire.span(
                "Loading item from source, cache size {cachesize}/{cachesize_max}, mainproc-queuesize {mp_queuesize}",
                cachesize=len(self.cache),
                cachesize_max=self.cache_cfg.cache_size,
                mp_queuesize=self._data_queue.qsize(),
            ):
                if self.src_transform is not None:
                    d = self.src_transform(d)

                d = self.apply_to_list_or_dict(self.data_stripper, d)

                if len(self.cache) >= self.cache_cfg.cache_size:
                    deleted_item = self.cache.pop(0)
                    # Log how many times this item was used for training
                    logfire.debug(
                        "Deleting item with {cachehits} hits",
                        cachehits=deleted_item["cachehits"],
                    )

                self.cache.append(new_cache_item := {"original": d, "cachehits": 0})

                yield return_item(new_cache_item)

                # As long as the main process quickly needs new items, only apply batch transform on cache items
                while len(self.cache) > 0 and self._data_queue.qsize() <= self.cache_cfg.prepare_new_threshold:
                    cache_item = random.choice(self.cache)  # Take from cache
                    cache_item["cachehits"] += 1
                    logfire.debug(
                        "Yielding cached item with {cachehits} hits because Qsize={mp_queuesize} <= T={threshold}",
                        cachehits=cache_item["cachehits"],
                        mp_queuesize=self._data_queue.qsize(),
                        threshold=self.cache_cfg.prepare_new_threshold,
                    )
                    yield return_item(cache_item)
                    if self.cache_cfg.max_hits is not None and cache_item["cachehits"] >= self.cache_cfg.max_hits:
                        self.cache.remove(cache_item)
                        logfire.warning(
                            "Item has reached {maxhits} (max) cache hits, cache-size: {cache_size}",
                            maxhits=self.cache_cfg.max_hits,
                            cache_size=len(self.cache),
                        )

    def get_untransformed_case(self, index: int) -> Any:
        case = self.src_ds[self._indices[index]]
        if self.src_transform is not None:
            case = self.src_transform(case)
        return self.apply_to_list_or_dict(self.data_stripper, case)

    def __getitem__(self, index: int) -> SrcCaseType:
        src_case = self.get_untransformed_case(index)

        # All non deterministic transforms should be in the batch_transform
        if self.batch_transform is not None:
            src_case = self.batch_transform(src_case)

        return src_case

    def get_input_output_tuple(self, batch: SrcCaseType) -> Tuple[Any, ...]:
        """
        Relevant for ONNX and transforming cohorts to sklearn formats
        """
        raise NotImplementedError

    @abstractmethod
    def _visualize_batch_case(self, batch: SrcCaseType, i: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def _compute_batchsize_from_batch(self, batch: SrcCaseType) -> int:
        raise NotImplementedError

    def visualize_batch(self, batch: SrcCaseType) -> None:
        import streamlit as st

        batch_size = self._compute_batchsize_from_batch(batch)
        n_examples = min(8, batch_size)
        meta = batch["meta"] if isinstance(batch, dict) else [m.get("meta", {}) for m in batch]
        unique_groups = set([m["group_id"] if "group_id" in m else f"nogroup{i}" for i, m in enumerate(meta)])
        st.title(
            f"Drawing {n_examples} random examples from a batch with size {batch_size} with {len(unique_groups)} unique groups"
        )

        for i in random.sample(list(range(batch_size)), n_examples):
            self._visualize_batch_case(batch, i)

    def __repr__(self) -> str:
        res = f"MTLDataset of style {self.get_dataset_style()}"
        res += f" with mandatory keys for each case: {self.get_mandatory_keys()}"
        if self.get_dataset_style is DatasetStyle.MapStyle:
            res += f" with {self.__len__()} cases"

        # If a SubCaseCachingDataset is used, the supercase_ds has a length which we can use
        elif hasattr(self.src_ds, "supercase_ds"):
            try:
                res += f" with {len(self.src_ds.supercase_ds)} supercases"
            except Exception as e:
                logging.debug(f"Could not get length of supercase_ds due to {e}")

        return res

    def st_find_invalid_cases(self) -> None:
        from mmm.logging.st_ext import st, stw

        with st.form("validator settings"):
            shuffle = st.checkbox("shuffle", value=True)

            submit = st.form_submit_button("Find invalid cases")

        if submit:
            mp_iterator = self.get_mp_batchiterator(shuffle=shuffle)
            mp_iter, i = iter(mp_iterator), 0
            progbar = st.progress(value=0.0)
            error_cases = []
            while True:
                try:
                    batch = next(mp_iter)
                    progbar.progress(
                        i / len(mp_iterator),
                        text=f"Checking step {i} of {len(mp_iterator)}",
                    )
                    i += 1
                except StopIteration:
                    st.balloons()
                    if error_cases:
                        st.error(f"Found {[x[0] for x in error_cases]} invalid cases")
                    return error_cases
                except Exception as e:
                    error_cases.append((i, e))
                    st.error(f"Could not load batch {i} due to {e}")
                    logging.error(f"Could not load batch {i} due to {e}")
                    stw(batch)

    def _st_repr_(self, st_prefix: str = "") -> None:
        from mmm.logging.st_ext import st, stw

        stw(f"### {self.__class__.__name__} of style {self.get_dataset_style()}")

        def dataset_explorer():
            if self.get_dataset_style() is DatasetStyle.MapStyle:
                case_index = st.number_input(
                    f"Select case between 0 and {len(self)}",
                    step=1,
                    value="min",
                    min_value=0,
                    max_value=len(self) - 1,
                    key=f"case_index{st_prefix}",
                )
                try:
                    cases = self.verify_case_by_index(case_index)
                except InvalidCaseError as e:
                    st.error(e)
                    cases = self.get_untransformed_case(case_index)

                cases = cases if isinstance(cases, list) else [cases]

                self.st_case_viewer(cases, case_index)
            else:
                with st.form("iterator demo"):
                    max_items = int(st.number_input("Max iterations", step=1, value=10))
                    display_every = int(st.number_input("Display every N case", step=1, value=2))
                    submitted = st.form_submit_button("Show cases")

                    if submitted:
                        ls = []
                        for i, cases in enumerate(self.__iter__(apply_batchtransform=False)):
                            if i % display_every == 0:
                                cases = cases if isinstance(cases, list) else [cases]
                                for case in cases:
                                    self.verify_case(case)
                                ls.extend(cases)
                            if i >= max_items - 1:
                                break
                        self.st_case_viewer(ls, -1)

        def batch_explorer():
            with st.form("Dataloader settings"):
                seed = st.number_input("random_seed", value=0)
                batchsize = st.number_input("batchsize", value=1, min_value=1, max_value=100)
                shuffle: bool = st.toggle("shuffle", value=True)
                num_workers: int = st.number_input("num_workers", value=0, min_value=0, max_value=10)
                pin_memory: bool = st.toggle("pin_memory", value=False)
                submitted = st.form_submit_button("Load batches")
            if submitted:
                random.seed(seed)
                torch.manual_seed(seed)
                dataloader = self.get_dataloader(
                    batch_size=batchsize,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                batch = next(iter(dataloader))
                self.visualize_batch(batch)

        pages = {
            "Dataset explorer": dataset_explorer,
            "Batch explorer": batch_explorer,
            "Validator": self.st_find_invalid_cases,
        }

        demo_name = st.sidebar.selectbox("Choose", list(pages.keys()))
        pages[demo_name]()

    def st_case_viewer(self, ls: list[dict[str, Any]], i: int = -1) -> None:
        """
        Can be called by GUIs to visualize a specific dataset item.
        """
        from mmm.logging.st_ext import stw

        stw("Overwrite this method to visualize a case")
        stw(ls, st_prefix=f"case{i}")
