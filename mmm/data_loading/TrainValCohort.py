from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Generic, Iterable, Literal, Optional, Tuple, TypeVar

import logfire
import torch
import torch.multiprocessing as mp
from pydantic import Field, model_validator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mmm.BaseModel import BaseModel
from mmm.DataSplit import DataSplit

from .MTLDataset import DatasetStyle, MTLDataset

DatasetType = TypeVar("DatasetType", bound=MTLDataset, covariant=True)


class TrainValCohort(Generic[DatasetType]):
    """
    Can be used to define a training cohort with a training and a validation dataset.
    It is used to create dataloaders.
    """

    class Config(BaseModel):
        batch_size: tuple[int | None, int | None] | int = Field(default=(None, 1))
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
        pin_memory: bool = True

        @model_validator(mode="before")
        @classmethod
        def batchsize_to_tuple(cls, data):
            if isinstance(data, int):
                data = (data, data)
            return data

    def __init__(
        self,
        args: Config,
        train_ds: DatasetType,
        val_ds: DatasetType,
        for_task_name: str | None = None,
    ) -> None:
        self.args: TrainValCohort.Config = args
        self.for_task_name: str | None = for_task_name
        self.datasets: Tuple[DatasetType, DatasetType] = (train_ds, val_ds)
        self.data_loaders: Tuple[Optional[DataLoader], Optional[DataLoader]] = (
            None,
            None,
        )

    def __repr_html__(self) -> str:
        return f"""
        <pre><code>{self.__repr__()}</pre></code>
        """

    def _st_repr_(self, st_prefix: str = "") -> None:
        # Streamlit would otherwise resample cross validation splits
        import streamlit as st

        from mmm.logging.st_ext import st, stw

        if st.sidebar.button(f"Run `prepare_epoch(epoch=0)`", key=f"{st_prefix}prepare_epoch"):
            self.prepare_epoch(epoch=0)
        else:
            if not self.for_task_name:
                self.for_task_name = "streamlit_visualization"
            self.push_batchsize_to_datasets()

        if split_name_selection := st.sidebar.selectbox(
            "TrainOrVal", ["Training", "Validation"], key=f"{st_prefix}split"
        ):
            split_name: str = split_name_selection
        else:
            split_name: str = "Training"
        train_val_index = 0 if split_name == "Training" else 1

        stw(self.datasets[train_val_index], st_prefix=f"{st_prefix}dataset_{split_name.lower()}")

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
            cur_len = len(self.datasets[data_split.value].src_ds)  # type: ignore
            if cur_len > len(self.datasets[data_split.value]) * self.datasets[data_split.value].reduced_size:
                logfire.debug(
                    "Dataset {data_split} of {self} has grown from {previous_len} to {cur_len}.",
                    data_split=data_split,
                    self=self,
                    previous_len=len(self.datasets[data_split.value]),
                    cur_len=cur_len,
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

    def push_batchsize_to_datasets(self):
        assert self.for_task_name is not None, "Set name for logging before pushing batch size to datasets"
        train_batchsize, val_batchsize = self.args.batch_size  # type: ignore
        assert train_batchsize is not None, "Train batch size must be set for pushing batch size to datasets"
        assert val_batchsize is not None, "Validation batch size must be set for pushing batch size to datasets"
        self.datasets[0].set_cohort_settings(
            batch_size=train_batchsize, task_name=self.for_task_name, split_name="train"
        )
        if self.datasets[1] is not None:
            self.datasets[1].set_cohort_settings(
                batch_size=val_batchsize, task_name=self.for_task_name, split_name="val"
            )

    def prepare_epoch(self, epoch: int):
        """
        The epoch might be used by child classes to seed splits for cross validation.
        """
        self.push_batchsize_to_datasets()

        # If deterministic datasets are used, then dataloaders do not need to be renewed:
        if None in self.data_loaders:
            assert None not in self.args.batch_size, f"Batch size must be set for creating dataloaders, {epoch=}"
            train_loader = self.datasets[0].get_dataloader(
                shuffle=self.args.shuffle_loaders[0],
                batch_size=self.args.batch_size[0],
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                # Usually, this only improves performance, but it is important for federated learning.
                persistent_workers=self.args.num_workers > 0,
            )
            val_loader = self.datasets[1].get_dataloader(
                shuffle=self.args.shuffle_loaders[1],
                batch_size=self.args.batch_size[1],
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                # Usually, this only improves performance, but it is important for federated learning.
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
