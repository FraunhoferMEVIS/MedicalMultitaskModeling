from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.torch_ext import CachingSubCaseDS

from .MTLDataset import MTLDataset, SrcCaseType, mtl_batch_collate

CaseDict = TypeVar("CaseDict", bound=Dict)


class ApproximationDataset(MTLDataset):
    def __init__(
        self,
        mtl_ds: Dataset[SrcCaseType],
        src_transform: Optional[Callable[[SrcCaseType], Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            mtl_ds,
            ["image", "target"],
            ["meta"],
            src_transform=src_transform,
            *args,
            **kwargs,
        )

    def verify_case(self, case):
        self.assert_image_data_assumptions(case["image"])
        self.assert_image_data_assumptions(case["target"])
        return case

    def get_input_output_tuple(self, batch: Dict[str, Any]) -> Tuple[Any, ...]:
        return batch["image"], batch["target"]

    def st_case_viewer(self, case: Dict[str, Any], X) -> None:
        import streamlit as st

        from mmm.logging.st_ext import blend_with_mask

        im = case["image"]
        blend_with_mask(im, None, caption_suffix=f"Shape: {im.shape}")
        targetim = case["target"]
        blend_with_mask(targetim, None, caption_suffix=f"Shape: {targetim.shape}")
        st.write(case)

    def _compute_batchsize_from_batch(self, batch: Dict[str, Any]) -> int:
        return batch["image"].shape[0]

    def _visualize_batch_case(self, batch: Dict[str, Any], i: int) -> None:
        import streamlit as st

        from mmm.logging.st_ext import blend_with_mask

        blend_with_mask(
            batch["image"][i],
            None,
            caption_suffix=f"{i}/{self._compute_batchsize_from_batch(batch)}: {batch['image'][i].shape}",
            st_key=f"b{i}",
        )
        blend_with_mask(
            batch["target"][i],
            None,
            caption_suffix=f"{i}/{self._compute_batchsize_from_batch(batch)}: {batch['target'][i].shape}",
            st_key=f"b{i}",
        )


def adaptive_batch_transform(case: dict[str, torch.Tensor]):
    r = [{"image": subX, "target": subY, "meta": case["meta"]} for subX, subY in zip(case["image"], case["target"])]
    del case
    return r


def transform_cohort_to_approximative(cohort: TrainValCohort, transform: Callable):
    """
    Transforms any given TrainValCohort into a TrainValCohort with an ApproximationDataset.
    transform: Callable should be a  mmm/augmentations.py --> GenerativeTransform,
    which generates a targegt image from the image in the dataset. Will return a dictionary with image and target keys.
    """

    if cohort.datasets[1].src_transform is None:
        cohort.datasets[1].src_transform = nn.Identity()

    # Selecting src and batch transform from validation cohort to avoid augmentations
    new_src_transform = transforms.Compose([cohort.datasets[1].src_transform, transform])

    return TrainValCohort(
        args=cohort.args,
        train_ds=ApproximationDataset(
            mtl_ds=cohort.datasets[0].src_ds,
            src_transform=new_src_transform,
            batch_transform=lambda x: (
                [x]
                if len(x["image"].shape) < 4
                else [
                    {"image": subX, "target": subY, "meta": subM}
                    for subX, subY, subM in zip(x["image"], x["target"], x["meta"])
                ]
            ),
            collate_fn=mtl_batch_collate,
        ),
        val_ds=ApproximationDataset(
            mtl_ds=cohort.datasets[1].src_ds,
            src_transform=new_src_transform,
            batch_transform=lambda x: (
                [x]
                if len(x["image"].shape) < 4
                else [
                    {"image": subX, "target": subY, "meta": subM}
                    for subX, subY, subM in zip(x["image"], x["target"], x["meta"])
                ]
            ),
            collate_fn=mtl_batch_collate,
        ),
    )
