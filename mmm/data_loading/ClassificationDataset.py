from __future__ import annotations
import logging
import numpy as np
import random
import warnings
from typing import Any, List, Optional, Callable, Dict, Tuple, TypeVar

from PIL.Image import Image

import torchvision.transforms.functional as F

from torch.utils.data import Dataset

from mmm.bucketizing import BucketConfig
from mmm.logging.st_ext import blend_with_mask
from .MTLDataset import MTLDataset, SrcCaseType

CaseDict = TypeVar("CaseDict", bound=Dict)


class ClassificationDataset(MTLDataset):
    @staticmethod
    def bucketize_case(
        buckets: BucketConfig, description_extractor: Callable[[CaseDict], str]
    ) -> Callable[[CaseDict], CaseDict]:
        def f(casedict: CaseDict):
            desc = description_extractor(casedict)
            bucket_name, new_class_id = buckets.get_bucket_name(desc)

            assert "bucket_name" not in casedict
            casedict["bucket_name"] = bucket_name

            casedict["class"] = new_class_id
            # if "class"
            # assert "old_class_id" not in casedict
            # casedict["old_class_id"] = casedict["class"]

            return casedict

        return f

    @staticmethod
    def TorchvisionToMTLClf(t: Tuple[Image, int]):
        return {"image": F.to_tensor(t[0]), "class": t[1]}

    @staticmethod
    def from_torchvision(ds: Dataset, class_names: Optional[List[str]] = None) -> ClassificationDataset:
        """
        A torchvision dataset returns tuples of the form (img: PIL.Image, class_index: int)
        """
        warnings.warn("Use TorchvisionToMTLClf instead as a src_transform", DeprecationWarning)
        r = ClassificationDataset(
            src_ds=ds,
            src_transform=lambda t: {"image": F.to_tensor(t[0]), "class": t[1]},
            class_names=class_names,
        )
        return r

    def __init__(
        self,
        src_ds: Dataset[SrcCaseType],
        class_names: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> None:
        assert class_names is not None, "Classification datasets without class names are deprecated"
        self.class_names: List[str] = class_names
        self.vis_classes = class_names
        super().__init__(src_ds, ["image", "class"], ["meta"], *args, **kwargs)

    def verify_case_by_index(self, index: int) -> Dict[str, Any]:
        case = super().verify_case_by_index(index)
        self.assert_image_data_assumptions(case["image"])
        assert isinstance(case["class"], int), "Class label should be an integer"
        return case

    def set_indices_by_fraction(self, fraction: float, seed: int = 13) -> None:
        """
        Starts by adding one example per class, then draws random samples until the fraction criterion is met.

        As a result, a fraction of zero will result in a subset of one sample per class
        """
        # Can be used for sampling new cases when artifically reducing the dataset's size
        self.seeded_random = random.Random(seed)
        src_dataset_length = len(self.src_ds)  # type: ignore
        # self._indices = list(range(src_dataset_length))
        self.reset_indices()
        shuffled_original_indices = list(self._indices)
        self.seeded_random.shuffle(shuffled_original_indices)
        new_indices = []

        # Add one sample per class
        classes: List[int] = []
        for original_index in shuffled_original_indices:
            case = self.verify_case_by_index(original_index)
            if (case_class := case["class"]) not in classes:
                classes.append(case_class)
                new_indices.append(original_index)

            if len(new_indices) >= len(self.class_names):
                break
        assert len(new_indices) == len(self.class_names)

        number_of_cases_missing = int(src_dataset_length * fraction) - len(new_indices)
        if number_of_cases_missing > 0:
            unused_indices = set(range(src_dataset_length)) - set(new_indices)
            unbalanced_indices = self.seeded_random.sample(unused_indices, number_of_cases_missing)
            new_indices.extend(unbalanced_indices)
        self._indices = np.array(new_indices)

    def set_classes_for_visualization(self, classes: List[str]):
        self.class_names = classes

    def get_classes_for_visualization(self):
        return self.class_names

    def get_input_output_tuple(self, batch: Dict[str, Any]) -> Tuple[Any, ...]:
        return batch["image"], batch["class"]

    def st_case_viewer(self, case: Dict[str, Any], i: int) -> None:
        import streamlit as st

        if self.class_names:
            for i, class_name in enumerate(self.class_names):
                if case["class"] != i:
                    # For large datasets, do not show negative classes
                    if len(self.class_names) < 100:
                        st.write(f"class {i}: {class_name}")
                else:
                    st.write(f"THIS CLASS ({i}): {class_name}")

        im = case.pop("image")
        blend_with_mask(im, None, caption_suffix=f"Shape: {im.shape}", st_key=f"c{i}")
        st.write(case)

    def _compute_batchsize_from_batch(self, batch: Dict[str, Any]) -> int:
        return batch["image"].shape[0]

    def _visualize_batch_case(self, batch: Dict[str, Any], i: int) -> None:
        import streamlit as st

        patch = batch["image"][i]
        class_name = self.class_names[batch["class"][i]]
        st.write(f"Label: {batch['class'][i]}, " + class_name)

        try:
            st.write(batch["meta"][i])
        except:
            logging.debug(f"Batch does not contain meta information at {i}")

        blend_with_mask(
            patch,
            None,
            caption_suffix=f"{i}/{self._compute_batchsize_from_batch(batch)}: {patch.shape}",
            st_key=f"b{i}",
        )
