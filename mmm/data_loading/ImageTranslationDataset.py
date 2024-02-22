from __future__ import annotations
from typing import Any, List, Optional, Callable, Dict, Tuple, TypeVar

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from .MTLDataset import MTLDataset, SrcCaseType

CaseDict = TypeVar("CaseDict", bound=Dict)


class ImageTranslationDataset(MTLDataset):
    def __init__(
        self,
        src_ds: Dataset[SrcCaseType],
        src_transform: Optional[Callable[[SrcCaseType], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(src_ds, ["image", "targetimage"], ["meta"], src_transform=src_transform)

    def verify_case_by_index(self, index: int) -> Dict[str, Any]:
        case = super().verify_case_by_index(index)
        self.assert_image_data_assumptions(case["image"])
        self.assert_image_data_assumptions(case["targetimage"])
        return case

    def get_input_output_tuple(self, batch: Dict[str, Any]) -> Tuple[Any, ...]:
        return batch["image"], batch["imagetarget"]

    def st_case_viewer(self, case: Dict[str, Any]) -> None:
        import streamlit as st
        from mmm.logging.st_ext import blend_with_mask

        st.title("Untransformed image:")

        im = case.pop("image")
        blend_with_mask(im, None, caption_suffix=f"Shape: {im.shape}")
        targetim = case.pop("targetimage")
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
            batch["targetimage"][i],
            None,
            caption_suffix=f"{i}/{self._compute_batchsize_from_batch(batch)}: {batch['targetimage'][i].shape}",
            st_key=f"b{i}",
        )
