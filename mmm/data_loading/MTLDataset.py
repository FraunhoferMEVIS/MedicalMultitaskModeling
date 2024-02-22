from __future__ import annotations
import warnings
from enum import Enum, auto
import os
import numpy as np
import logging
from pathlib import Path
from abc import abstractmethod
import random
from typing import Callable, Any, List, Tuple, Optional, TypeVar, Set, NewType, Iterator
from typing import Generic, TypeVar
from torch.utils.data import Dataset


from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate
from mmm.BaseModel import BaseModel
from mmm.transforms import KeepOnlyKeysInDict
from mmm.utils import disk_cacher, unique_str_hash, recursive_equality


class InvalidCaseError(Exception):
    def __init__(self, src_ds: MTLDataset, case_id: int, user_msgs: List[str]) -> None:
        self.src_ds, self.case_id = src_ds, case_id

        findcase = "Case" if case_id == -1 else f"Case {case_id}"
        s = f"{findcase} of {src_ds} is invalid:\n"
        for user_msg in user_msgs:
            s += f"{user_msg}\n"
        super().__init__(s)


SrcCaseType = dict[str, Any]


def mtl_collate(batch: List[SrcCaseType]) -> SrcCaseType:
    """Ignores the meta key when collating"""
    meta_dicts = [item["meta"] if "meta" in item else {} for item in batch]
    for case_dict in batch:
        if "meta" in case_dict:
            case_dict.pop("meta")
    res = default_collate(batch)
    res["meta"] = meta_dicts  # type: ignore
    return res  # type: ignore


def mtl_batch_collate(batch: List[List[SrcCaseType]]) -> SrcCaseType:
    """Ignores the meta key when collating datasets that already return batches"""
    # Flatten list
    batch_single_list = [item for sublist in batch for item in sublist]
    return mtl_collate(batch_single_list)


class DatasetStyle(Enum):
    MapStyle = auto()
    IterStyle = auto()


class IterableDSWrapper(IterableDataset):
    def __init__(self, fn) -> None:
        self.fn = fn

    def __iter__(self):
        return self.fn.__iter__()


class MTLDataset(Dataset[SrcCaseType]):
    """Wrapper for the PyTorch implementation of Datasets.
    Map-style datasets have a length and are integer-indexable.

    It enforces cases to be based on dictionaries where certain keys have a fixed place and meaning.
    Like "image" is usually the image data scaled to [0, 1].
    Custom types might make the code more modular but would break compatibility
    with MONAI transforms and PyTorch dataloaders.

    MTLDataset can be a wrapper around a vanilla torch dataset which can verify cases depending on the dataset type.
    Additionally, we recommend not using random transforms as part of the `src_ds` or the optional src_transform.

    Building steps for a case:

    1. Source dataset -> should contain the full data for a case
    2. src_transform -> converts the data into the right formats and the result should be small if cache_folder is set

    If cache_folder is set, the result after src_transform and stripping non-relevant keys is saved to `cache_folder`.
    Make sure `cache_folder` is unique to this dataset, including its config!
    If you are using our cohort abstraction it will try to make your cache unique w.r.t. its config.

    If you want to apply random transformations, use the cohort's batch-transforms
    """

    @staticmethod
    def assert_image_data_assumptions(image_data: torch.Tensor) -> None:
        assert torch.is_tensor(image_data) and type(image_data) is torch.Tensor, "image should be of type tensor"
        assert image_data.shape[0] >= 1, "Image needs to have at least one channel"
        assert isinstance(image_data, torch.FloatTensor)

    def __init__(
        self,
        src_ds: Dataset[SrcCaseType],
        mandatory_case_keys: List[str],
        optional_case_keys: List[str],
        src_transform: Optional[Callable[[SrcCaseType], SrcCaseType]] = None,
        batch_transform: Optional[Callable[[SrcCaseType], Any]] = None,
        collate_fn: Optional[Callable[[List[Any]], SrcCaseType]] = None,
        verify_cases: int = 0,
        **dataloader_kwargs,
    ) -> None:
        """Generates and prepares the dataset.

        Args:
            data_root (Path): Should be the path to a folder where the data for this dataset can be stored
            data_cache (Path): Path to cache directory. If it doesn't exist, the dataset will create it.
            mandatory_case_keys (List[str]): To be a valid case, these keys need to be in every case
            optional_case_keys: (List[str]): These keys will not be deleted from the case
            needs_preparation: (bool): If true, will try to call `self.prepare()`
        """
        self.src_ds = src_ds
        self.cache_folder: Optional[Path] = None
        self.src_transform = src_transform
        self.batch_transform: Optional[Callable] = batch_transform
        self.collate_fn = collate_fn if collate_fn is not None else mtl_collate
        self.dataloader_kwargs = dataloader_kwargs

        self.mandatory_case_keys: List[str] = mandatory_case_keys
        self.optional_case_keys: List[str] = optional_case_keys
        self.data_stripper = KeepOnlyKeysInDict(
            keys=set(self.mandatory_case_keys + self.optional_case_keys),
        )

        self.reduced_size: float = 1.0
        # Represented by a list because it is valid to repeat indices if the user wants so by setting a fraction > 1.
        if self.get_dataset_style() is DatasetStyle.MapStyle:
            self._indices: np.ndarray
            self.reset_indices()

            for case_id in random.sample(list(self._indices), k=min(len(self._indices), verify_cases)):
                self.verify_case_by_index(case_id)

    def get_dataset_style(self) -> DatasetStyle:
        if isinstance(self.src_ds, IterableDataset):
            return DatasetStyle.IterStyle
        else:
            return DatasetStyle.MapStyle

    def enable_caching(self, cache_folder: Path, test_cache_validity: bool = True):
        """
        Makes sure that the cache folder exists.

        If it already exists, optionally check for validity.
        """
        self.cache_folder = cache_folder
        if self.cache_folder.exists():
            if test_cache_validity:
                for case_path in self.cache_folder.glob("*.pkl"):
                    case_id: int = int(case_path.stem)
                    logging.info(f"Checking cache of case {case_id} for correctness")
                    shouldbe = self.get_untransformed_case(case_id)
                    cached_case = self.get_case_from_cache(case_id)
                    for key in self.mandatory_case_keys + self.optional_case_keys:
                        if key in shouldbe and key != "meta":
                            if not recursive_equality(shouldbe[key], cached_case[key]):
                                logging.warn(f"{key} shows difference between original and cached:")
                                logging.warn(f"{shouldbe[key]} and {cached_case[key]}")
                                assert False
                    break
        else:
            self.cache_folder.mkdir(parents=True)
            logging.info(f"Starting a new cache at {self.cache_folder}")
        return self

    def get_dataloader(self, **kwargs):
        if self.get_dataset_style() is DatasetStyle.MapStyle:
            if "sampler" in self.dataloader_kwargs and "shuffle" in kwargs:
                kwargs.pop("shuffle")
            return DataLoader(
                self,
                collate_fn=self.collate_fn,
                # a whole day in seconds
                # without multiprocessing, a timeout>0 is not allowed
                timeout=(86400 if "num_workers" in kwargs and kwargs["num_workers"] > 0 else 0),
                **kwargs,
                **self.dataloader_kwargs,
            )
        else:
            # Shuffle is not a valid option for iterablestyle datasets
            # assert isinstance(self, IterableDataset)
            if "shuffle" in kwargs:
                kwargs.pop("shuffle")
            return DataLoader(
                IterableDSWrapper(self),
                collate_fn=self.collate_fn,
                timeout=(86400 if "num_workers" in kwargs and kwargs["num_workers"] > 0 else 0),
                **kwargs,
                **self.dataloader_kwargs,
            )

    def get_mp_batchiterator(self, get_untransformed_cases=False, batch_size=1, shuffle=True, **kwargs):
        """
        Iterates through this dataset's batches with all available workers.
        """

        class UntransformedDS(Dataset):
            def __init__(self, src_ds: MTLDataset) -> None:
                self.src = src_ds

            def __len__(self) -> int:
                return self.src.__len__()

            def __getitem__(self, index) -> Any:
                return (
                    self.src.get_untransformed_case(index)
                    if self.src.cache_folder is None
                    else self.src.get_case_from_cache(index)
                )

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

    def prefill_cache(self):
        missing = self.get_free_cache_case_slots()
        if missing > 0:
            logging.info(f"Preparing cache of {self.cache_folder}")
            prepare_dataloader = self.get_mp_batchiterator(persistent_workers=False)
            pbar = tqdm(prepare_dataloader)
            for _ in pbar:
                missing = self.get_free_cache_case_slots()
                pbar.set_description(f"{missing} cases still missing for caching")
                if missing <= 0:
                    break

            if prepare_dataloader._iterator is not None:
                # Force kill all workers of the dataloader, just to be sure
                prepare_dataloader._iterator._shutdown_workers()  # type: ignore
                prepare_dataloader._iterator = None

    def get_free_cache_case_slots(self) -> int:
        if self.cache_folder is not None:
            assert self.cache_folder.exists()
            cached_num = len(list(self.cache_folder.glob("*.pkl")))
            # For now the number of free cache slots is exactly
            return len(self) - cached_num
        else:
            return 0

    def reset_indices(self) -> None:
        if self.reduced_size < 1.0:
            logging.warning(f"Resetting indices of {self} to full size")
        self.reduced_size = 1.0
        if self.get_dataset_style() is DatasetStyle.MapStyle:
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
        from mmm.data_loading.utils import train_val_split, TransformedSubset
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
                new_indices, _ = train_val_split(
                    list(range(len(self.src_ds.supercase_ds))), perc=fraction, seed=seed  # type: ignore
                )
                self.src_ds.supercase_ds = TransformedSubset(self.src_ds.supercase_ds, indices=new_indices)
            else:
                raise NotImplementedError

    def verify_all_cases(self):
        from tqdm.auto import tqdm

        for case_index in tqdm(range(len(self))):
            self.verify_case_by_index(case_index)

    def verify_case_by_index(self, index: int) -> SrcCaseType:
        d = self.get_untransformed_case(index) if self.cache_folder is None else self.get_case_from_cache(index)
        self.verify_case(d)
        return d

    def verify_case(self, d: SrcCaseType) -> None:
        missing_keys = [k for k in self.mandatory_case_keys if k not in d]
        if missing_keys:
            raise InvalidCaseError(self, -1, [f"{k} missing in case {d}" for k in missing_keys])

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self, apply_batchtransform: bool = True) -> Iterator[SrcCaseType]:
        for d in iter(self.src_ds):
            if self.src_transform:
                d = self.src_transform(d)
            if self.batch_transform is not None and apply_batchtransform:
                d = self.batch_transform(d)
            d = self.data_stripper(d)  # type: ignore
            yield d

    def get_untransformed_case(self, index: int) -> Any:
        case = self.src_ds[self._indices[index]]
        if self.src_transform is not None:
            case = self.src_transform(case)
        return self.data_stripper(case)  # type: ignore

    def get_case_from_cache(self, index: int) -> Any:
        assert self.cache_folder is not None
        filepath = self.cache_folder / f"{self._indices[index]}.pkl"
        if filepath.exists():
            with open(filepath, "rb") as f:
                return torch.load(f)
        else:
            src_case = self.get_untransformed_case(index)
            with open(filepath, "wb") as f:
                torch.save(src_case, f)
            return src_case

    def __getitem__(self, index: int) -> SrcCaseType:
        src_case = self.get_untransformed_case(index) if self.cache_folder is None else self.get_case_from_cache(index)

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
        st.title(f"Drawing {n_examples} random examples from a batch with size {batch_size}")

        for i in random.sample(list(range(batch_size)), n_examples):
            self._visualize_batch_case(batch, i)

    def __repr__(self) -> str:
        res = f"MTLDataset of style {self.get_dataset_style()}"
        res += f" with mandatory keys for each case: {self.mandatory_case_keys}"
        if self.get_dataset_style is DatasetStyle.MapStyle:
            res += f" with {self.__len__()} cases"
        return res

    def st_find_invalid_cases(self) -> None:
        from mmm.logging.st_ext import stw, st

        with st.form("validator settings"):
            shuffle = st.checkbox("shuffle", value=True)

            submit = st.form_submit_button("Find invalid cases")

        if submit:
            mp_iterator = self.get_mp_batchiterator(shuffle=shuffle)
            mp_iter, i = iter(mp_iterator), 0
            progbar = st.progress(value=0.0)
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
                    return
                except Exception as e:
                    st.error(f"Could not load batch {i} due to {e}")
                    logging.error(f"Could not load batch {i} due to {e}")
                    stw(batch)

    def _st_repr_(self, st_prefix: str = "") -> None:
        from mmm.logging.st_ext import stw, st

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
                    case = self.verify_case_by_index(case_index)
                except InvalidCaseError as e:
                    st.error(e)
                    case = self.get_untransformed_case(case_index)
                self.st_case_viewer(case, case_index)
            else:
                with st.form("iterator demo"):
                    max_items = int(st.number_input("Max iterations", step=1, value=10))
                    display_every = int(st.number_input("Display every N case", step=1, value=2))
                    submitted = st.form_submit_button("Reload iterator")

                    if submitted:
                        for i, case in enumerate(self.__iter__(apply_batchtransform=False)):
                            self.verify_case(case)
                            # case = next(iter(self))
                            if i % display_every == 0:
                                self.st_case_viewer(case, i)
                            if i > max_items:
                                break

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

    def st_case_viewer(self, case: SrcCaseType, i: int = -1) -> None:
        """
        Can be called by GUIs to visualize a specific case
        """
        from mmm.logging.st_ext import stw

        stw("Overwrite this method to visualize a case")
        stw(case)
