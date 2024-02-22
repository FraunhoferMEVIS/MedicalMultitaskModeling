from __future__ import annotations
from typing import Any, List, Optional, Callable, Dict, Tuple, TypeVar
from functools import partial

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .MTLDataset import MTLDataset, SrcCaseType
from .SemSegDataset import SemSegDataset
from mmm.transforms import KeepOnlyKeysInDict


class MultilabelClassificationDataset(MTLDataset):
    """
    Expects cases with the keywords:

    - image: (typical image assumptions of mmm)
    - class_labels: torch.FloatTensor with a float between 0. and 1. for the confidence of that class being True

    Optional:

    - loss_weights: Allows to ignore certain class during loss computation. Use this to deal e.g. with NaNs targets
    """

    @staticmethod
    def from_semseg(ds: SemSegDataset, ignore_classes: Optional[List[int]] = None) -> MultilabelClassificationDataset:
        """
        Uses ds in-place to construct a multilabel classification dataset by appending a batch transform.
        """
        if ignore_classes is None:
            ignore_classes = []
        assert len(ignore_classes) == len(set(ignore_classes)), f"Don't ignore a class twice {ignore_classes=}"

        def semsegbatch_to_mclf_batch(num_classes: int, c, semsegbatch: Dict[str, Any]) -> Dict[str, Any]:
            masks = semsegbatch["label"]
            class_labels = []
            for i in range(masks.shape[0]):
                x = torch.zeros(num_classes)
                original_classes = torch.unique(masks[i, :]).tolist()
                x[[c[x] for x in original_classes if x in c]] = 1
                class_labels.append(x)

            semsegbatch["class_labels"] = torch.stack(class_labels)
            return semsegbatch

        class_names = [v for i, v in enumerate(ds.class_names) if i not in ignore_classes]
        conv = {}
        for i, _ in enumerate(ds.class_names):
            if i not in ignore_classes:
                conv[i] = len(conv)

        mclf_ds = MultilabelClassificationDataset(
            ds.src_ds,
            class_names=class_names,
            src_transform=ds.src_transform,
            batch_transform=ds.batch_transform,
            collate_fn=transforms.Compose(
                [
                    ds.collate_fn,
                    partial(semsegbatch_to_mclf_batch, len(class_names), conv),
                ]
            ),
        )
        # For this approach to work the data stripper needs to chill and allow semseg and multilabel keys
        mclf_ds.data_stripper = KeepOnlyKeysInDict(
            keys=set(list(ds.data_stripper.keys) + list(mclf_ds.data_stripper.keys)),
        )

        return mclf_ds

    def __init__(self, src_ds: Dataset[SrcCaseType], class_names: List[str], **kwargs) -> None:
        self.class_names: List[str] = class_names
        super().__init__(src_ds, ["image", "class_labels"], ["loss_weights", "meta"], **kwargs)

    def verify_case_by_index(self, index: int) -> Dict[str, Any]:
        case = super().verify_case_by_index(index)
        self.assert_image_data_assumptions(case["image"])
        assert isinstance(case["class_labels"], torch.FloatTensor), "Labels should be confidences between 0. and 1."
        assert len(case["class_labels"]) == len(self.class_names)
        assert torch.min(case["class_labels"]) >= 0.0 and torch.max(case["class_labels"]) <= 1.0

        if "loss_weights" in case:
            assert isinstance(case["loss_weights"], torch.FloatTensor), "Loss weights should be float between 0. and 1."
            assert len(case["loss_weights"]) == len(self.class_names)
            assert torch.min(case["loss_weights"]) >= 0.0 and torch.max(case["loss_weights"]) <= 1.0
        return case

    def get_input_output_tuple(self, batch: Dict[str, Any]) -> Tuple[Any, ...]:
        return batch["image"], batch["class_labels"]

    def st_case_viewer(self, case: Dict[str, Any], i: int) -> None:
        import streamlit as st
        from mmm.logging.st_ext import blend_with_mask

        st.title("Untransformed image:")
        im = case.pop("image")
        blend_with_mask(im, None, caption_suffix=f"Shape: {im.shape}", st_key=f"c{i}")
        self._print_relevant_classes(
            case["class_labels"],
        )
        st.write(case)

    def _compute_batchsize_from_batch(self, batch: Dict[str, Any]) -> int:
        return batch["image"].shape[0]

    def _visualize_batch_case(self, batch: Dict[str, Any], i: int) -> None:
        import streamlit as st
        from mmm.logging.st_ext import blend_with_mask

        patch, class_labels = batch["image"][i], batch["class_labels"][i]

        self._print_relevant_classes(class_labels, batch["loss_weights"][i] if "loss_weights" in batch else None)
        if "meta" in batch:
            st.json(batch["meta"][i])
        blend_with_mask(
            patch,
            None,
            caption_suffix=f"{i}/{self._compute_batchsize_from_batch(batch)}: {patch.shape}",
            st_key=f"b{i}",
        )

    def _print_relevant_classes(self, class_labels: torch.Tensor, loss_weights: Optional[torch.Tensor] = None):
        import streamlit as st

        ignored_classes = []
        st.write(f"{class_labels} (Labels)")
        if loss_weights is not None:
            st.write(f"{loss_weights} (Loss weights)")
        for i, v in enumerate(class_labels):
            if loss_weights is None or loss_weights[i] > 0:
                if v > 0.0:
                    st.success(f"{self.class_names[i]} ({i=}) -> {v}")
                else:
                    st.error(f"{self.class_names[i]} ({i=}) -> {v}")
            else:
                ignored_classes.append((self.class_names[i], i))

        st.write(f"Ignored classes: {ignored_classes}")
