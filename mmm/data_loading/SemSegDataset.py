from typing import Any, List, Optional, Callable, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from mmm.logging.st_ext import blend_with_mask
from .MTLDataset import MTLDataset, SrcCaseType


class SemSegDataset(MTLDataset):
    """
    Contains a single pixel-dense mask for one or more images.
    """

    def __init__(
        self,
        src_ds: Dataset[SrcCaseType],
        class_names: List[str] = None,
        *args,
        **kwargs,
    ) -> None:
        assert class_names is not None, "Segmentation dataset needs class names"
        self.class_names: List[str] = class_names
        super().__init__(src_ds, ["image", "label"], ["original_size", "meta"], *args, **kwargs)

    def set_classes_for_visualization(self, classes: List[str]):
        self.class_names = classes

    def verify_case(self, d: Dict[str, Any]) -> None:
        super().verify_case(d)
        self.assert_image_data_assumptions(d["image"])
        # For example, MONAI tends to load some specific subclass of Tensor, exclude it!
        assert torch.is_tensor(d["label"]), "mask should be of type tensor"
        assert isinstance(d["label"], torch.LongTensor)
        assert d["label"].shape == d["image"].shape[1:]
        assert torch.max(d["label"]).item() < len(self.class_names), "There need to be more class names"
        assert len(d["image"].shape) == (len(d["label"].shape) + 1), "all labels should be in a one-dim tensor"

    def st_case_viewer(self, case: Dict[str, Any], i: int = -1) -> None:
        import streamlit as st

        st.title("Untransformed image:")
        blend_with_mask(
            case.pop("image"),
            case.pop("label"),
            classes=self.class_names,
            st_key=f"c{i}",
        )
        st.write(case)

    def _compute_batchsize_from_batch(self, batch: Dict[str, Any]) -> int:
        return batch["image"].shape[0]

    def _visualize_batch_case(self, batch: Dict[str, Any], i: int) -> None:
        import streamlit as st

        patch = batch["image"][i]
        patch_mask = batch["label"][i]
        if "meta" in batch:
            st.write(batch["meta"][i])

        mask_uniques = torch.unique(patch_mask)
        blend_with_mask(
            patch,
            patch_mask,
            caption_suffix=f"{i}/{self._compute_batchsize_from_batch(batch)}\nUniques: {mask_uniques}",
            classes=self.class_names,
            st_key=f"b{i}",
        )
