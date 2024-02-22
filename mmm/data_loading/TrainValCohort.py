from __future__ import annotations
import warnings
from tqdm.auto import tqdm
import numpy as np
from pydantic import Field
from typing import Optional, Literal, Iterable
from mmm.BaseModel import BaseModel
from typing import Optional, Tuple, Dict, Any, Generic, TypeVar, Callable
import logging

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from mmm.utils import unique_str_hash, get_default_cachepath
from mmm.DataSplit import DataSplit
from .MTLDataset import MTLDataset, DatasetStyle

DatasetType = TypeVar("DatasetType", bound=MTLDataset, covariant=True)


class TrainValCohort(Generic[DatasetType]):
    """
    Can be used to define a training cohort with a training and a validation dataset.
    It is used to create dataloaders.

    By setting cache-foldername you allow the cohort to cache its datasets.
    The cached values will appear in ML_DATA_CACHE/cache_key/*.pkl
    """

    class Config(BaseModel):
        batch_size: tuple[int | None, int | None] = Field(default=(None, 1))
        shuffle_loaders: tuple[bool, bool] = (True, True)
        num_workers: int = Field(
            default=1,
            description="""
Number of workers that prepare batches for this task.
Training loader and validation loader will ask for num_workers worker.
Should be chosen such that the task can run alone without any bottlenecks.
In case too many workers are requested the trainer will reduce the number of workers across all tasks
while keeping a minimum of one worker per task.
""",
        )
        pin_memory: bool = False
        prefill_cache: bool = True

    def __init__(
        self,
        args: Config,
        train_ds: DatasetType,
        val_ds: DatasetType,
        cache_foldername="",
    ) -> None:
        self.args: TrainValCohort.Config = args
        self.datasets: Tuple[DatasetType, DatasetType] = (train_ds, val_ds)
        self.data_loaders: Tuple[Optional[DataLoader], Optional[DataLoader]] = (
            None,
            None,
        )

        if cache_foldername:
            ignore_keys_for_discoverability = [
                "batch_size",
                "pin_memory",
                "shuffle_loaders",
                "num_workers",
                "prefill_cache",
            ]  # , "cross_val_split_size"]
            reidentification_args = self.args.dict().copy()
            for key in ignore_keys_for_discoverability:
                reidentification_args.pop(key)
            logging.info(f"I think {cache_foldername} cache needs to be unique w.r.t. \n {reidentification_args}")
            unique_config_hash = unique_str_hash(**reidentification_args)
            self.cache_path = get_default_cachepath(folder_name=cache_foldername) / unique_config_hash
            self.datasets[0].enable_caching(self.cache_path / "train")
            self.datasets[1].enable_caching(self.cache_path / "val")
        else:
            self.cache_path = None

    def __repr_html__(self) -> str:
        return f"""
        <pre><code>{self.__repr__()}</pre></code>
        """

    def _st_repr_(self) -> None:
        from mmm.logging.st_ext import stw, st

        # Streamlit would otherwise resample cross validation splits
        import streamlit as st

        # with st.():
        if st.sidebar.button(f"Run `prepare_epoch(epoch=0)`"):
            self.prepare_epoch(epoch=0)

        if split_name_selection := st.sidebar.selectbox("TrainOrVal", ["Training", "Validation"]):
            split_name: str = split_name_selection
        else:
            split_name: str = "Training"
        train_val_index = 0 if split_name == "Training" else 1

        stw(self.datasets[train_val_index])

    def __repr__(self) -> str:
        return (
            f"Cohort with args: {self.args}"
            f"\nTrain dataset: {self.datasets[0]}\nValidation dataset: {self.datasets[1]}"
        )

    def get_random_batch(self, split: DataSplit) -> Dict[str, Any]:
        warnings.warn(
            "get_random_batch is deprecated. Use ds.get_random_batch instead.",
            DeprecationWarning,
        )
        if None in self.data_loaders:
            self.prepare_epoch(epoch=0)

        return next(iter(self.data_loaders[split.value]))  # type: ignore

    def get_dataloader(self, data_split: DataSplit) -> DataLoader:
        dl = self.data_loaders[data_split.value]
        assert dl is not None, f"Prepare dataloader for {data_split} first using cohort.prepare_epoch"
        return dl

    def build_iterator(self, data_split: DataSplit) -> Iterable:
        # Map-style datasets might have a different length every time
        if self.datasets[data_split.value].get_dataset_style() == DatasetStyle.MapStyle:
            cur_len = len(self.datasets[data_split.value].src_ds)
            if cur_len > len(self.datasets[data_split.value]) * self.datasets[data_split.value].reduced_size:
                logging.debug(
                    f"Dataset {data_split} of {self} has grown from {len(self.datasets[data_split.value])} to {cur_len}."
                )
                # Due to multiprocessing the worker keeps a copy of the old, shorter dataset. It needs to be terminated.
                self.terminate_datasplit_workers(data_split)
                # Let the MTLDataset know about the new length
                self.datasets[data_split.value].reset_indices()

        return iter(self.data_loaders[data_split.value])

    def get_dataset(self, data_split: DataSplit) -> MTLDataset:
        ds = self.datasets[data_split.value]
        assert ds is not None
        return ds

    def get_active_workers(self, group: DataSplit):
        if self.data_loaders[group.value] is not None and self.data_loaders[group.value]._iterator is not None:
            return self.args.num_workers
        else:
            return 0

    def terminate_datasplit_workers(self, group: DataSplit):
        if self.data_loaders[group.value] is not None and self.data_loaders[group.value]._iterator is not None:
            self.data_loaders[group.value]._iterator._shutdown_workers()  # type: ignore
            self.data_loaders[group.value]._iterator = None

    def terminate_workers(self):
        """
        Terminating workers loses the state of the current iterator.
        As a result, you should never do this during usage (e.g. a loop).
        """
        self.terminate_datasplit_workers(DataSplit.train)
        self.terminate_datasplit_workers(DataSplit.val)

    def prepare_epoch(self, epoch: int):
        """
        The epoch might be used by child classes to seed splits for cross validation.
        """
        # Caches do not need to be specific to temporary cross validation datasets
        # because augmentations are not cached anyways
        if (self.cache_path is not None) and self.args.prefill_cache:
            self.datasets[0].prefill_cache()
            self.datasets[1].prefill_cache()

        # If deterministic datasets are used, then dataloaders do not need to be renewed:
        if None in self.data_loaders:
            assert None not in self.args.batch_size, f"Batch size must be set for creating dataloaders, {epoch=}"
            train_loader = self.datasets[0].get_dataloader(
                shuffle=self.args.shuffle_loaders[0],
                batch_size=self.args.batch_size[0],
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                persistent_workers=self.args.num_workers > 0,
            )
            val_loader = self.datasets[1].get_dataloader(
                shuffle=self.args.shuffle_loaders[1],
                batch_size=self.args.batch_size[1],
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                persistent_workers=self.args.num_workers > 0,
            )

            self.data_loaders = train_loader, val_loader

    def get_onnx_input(self, device: str):
        """
        Exporting to ONNX requires an example batch.

        Currently, it only works for the classification task.
        """
        if self.data_loaders:
            self.prepare_epoch(epoch=0)

        loader_index = 0
        assert self.data_loaders[loader_index] is not None, "Dataloader shouldn't be None here"
        example_batch = next(iter(self.data_loaders[loader_index]))  # type: ignore

        # Specific to classification
        example_input = example_batch["image"].to(device)

        return example_input

    def transform_cohort_to_sklearn(
        self,
        feature_encoder: Callable,
        get_untransformed_cases: bool = True,
        include_meta: bool = False,
        batch_size: int = 1,
        num_samples: int = -1,
    ):
        """
        If your feature encoder function is a shared block, you can use the `block.torch_device` property
        for finding out the device of the shared block.

        The dataset's method `get_input_output_tuple` is used for transforming the batch to sklearn's format.
        """

        def _transform_dataloader(ds: MTLDataset):
            x_batches = []
            y_batches = []
            meta = []

            active_children_before = len(mp.active_children())
            dl = ds.get_mp_batchiterator(
                get_untransformed_cases=get_untransformed_cases,
                batch_size=batch_size,
                persistent_workers=True,
            )
            for i, one_batch in enumerate(tqdm(dl)):
                if include_meta:
                    meta.append(torch.Tensor(one_batch["meta"]))
                x_raw, y = self.datasets[0].get_input_output_tuple(one_batch)
                x = feature_encoder(x_raw)
                x_batches.append(x)
                y_batches.append(y)
                if num_samples > 0 and i > num_samples:
                    break

            dl._iterator._shutdown_workers()  # type: ignore
            dl._iterator = None

            if len(mp.active_children()) > active_children_before:
                logging.warn(f"After killing workers, there are still {len(mp.active_children())} workers")

            if dl._iterator is not None:
                logging.warn(f"had to manually shutdown workers")
                dl._iterator._shutdown_workers()  # type: ignore
                dl._iterator = None
            if not meta:
                meta = torch.Tensor([])
            else:
                meta = torch.concat(meta)
            return torch.concat(x_batches).cpu(), torch.concat(y_batches), meta

        X_train, y_train, meta_train = _transform_dataloader(self.datasets[0])
        X_val, y_val, meta_val = _transform_dataloader(self.datasets[1])
        return X_train, y_train, X_val, y_val, meta_train, meta_val
