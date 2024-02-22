from __future__ import annotations
from typing import Any, Dict, Tuple

import torchvision.transforms.functional as F

from torch.utils.data import Dataset

from mmm.logging.st_ext import blend_with_mask
from .MTLDataset import MTLDataset, SrcCaseType


class RegressionDataset(MTLDataset):
    def __init__(self, src_ds: Dataset[SrcCaseType], *args, **kwargs) -> None:
        super().__init__(src_ds, ["image", "target"], ["meta"], *args, **kwargs)

    def verify_case_by_index(self, index: int) -> Dict[str, Any]:
        case = super().verify_case_by_index(index)
        self.assert_image_data_assumptions(case["image"])
        return case

    def get_input_output_tuple(self, batch: Dict[str, Any]) -> Tuple[Any, ...]:
        return batch["image"], batch["target"]

    def st_case_viewer(self, case: Dict[str, Any], index: int = -1) -> None:
        import streamlit as st

        st.write(f"Target: {case['target']}")
        im = case.pop("image")
        blend_with_mask(im, None, caption_suffix=f"Shape: {im.shape}", st_key=f"c{index}")
        st.write(case)

    def _compute_batchsize_from_batch(self, batch: Dict[str, Any]) -> int:
        return batch["image"].shape[0]

    def _visualize_batch_case(self, batch: Dict[str, Any], i: int) -> None:
        import streamlit as st

        patch = batch["image"][i]
        st.write(f"Target: {batch['target'][i]}")
        blend_with_mask(
            patch,
            None,
            caption_suffix=f"{i}/{self._compute_batchsize_from_batch(batch)}: {patch.shape}",
            st_key=f"b{i}",
        )
