from __future__ import annotations

import json
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import torchvision.transforms.functional as F
from PIL.Image import Image
from torch.utils.data import Dataset

from mmm.bucketizing import BucketConfig

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

    def __init__(
        self,
        src_ds: Dataset[SrcCaseType],
        class_names: Optional[list[str]] = None,
        *args,
        **kwargs,
    ) -> None:
        assert class_names is not None, "Classification datasets without class names are deprecated"
        self.class_names: list[str] = class_names
        self.vis_classes = class_names
        super().__init__(src_ds, *args, **kwargs)

    @staticmethod
    def get_mandatory_keys() -> list[str]:
        return super(ClassificationDataset, ClassificationDataset).get_mandatory_keys() + ["image", "class"]

    def verify_case(self, d: SrcCaseType) -> None:
        self.assert_image_data_assumptions(d["image"])
        assert isinstance(d["class"], int), "Class label should be an integer"
        return d

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

    def _label_to_html(self, class_label: int, context: tuple = ()) -> str:
        html = f'<span style="color:green">{self.class_names[class_label]}</span>'

        if context:
            html += f"<br>Context: <span style='color:orange'>{context}</span>"
        return html

    def st_case_viewer(self, ls: list[dict[str, Any]], i: int) -> None:
        from mmm.logging.st_ext import Image2D, M3Image, m3_image, st

        m3_image(
            data=M3Image.Data(
                images=[
                    Image2D.from_tensor(
                        img=d["image"],
                        class_names=self.class_names,
                        desc=json.dumps(d["meta"], indent=2, default=str) if "meta" in d else None,
                        caption=self._label_to_html(
                            d["class"],
                            context=d.get("meta", {}).get("context", ()),
                        ),
                    )
                    for d in ls
                ],
            ),
            key=f"img{i}_original",
        )

    def _compute_batchsize_from_batch(self, batch: Dict[str, Any]) -> int:
        return batch["image"].shape[0]

    def _visualize_batch_case(self, batch: Dict[str, Any], i: int) -> None:
        from mmm.logging.st_ext import blend_with_mask, st

        patch = batch["image"][i]
        class_name = self.class_names[batch["class"][i]]
        st.write(f"Label: {batch['class'][i]}, " + class_name)

        try:
            st.write(batch["meta"][i])
        except:
            logging.debug(f"Batch does not contain meta information at {i}")

        if self.batch_is_compressed(batched_image=batch["image"]):
            st.write(f"Compressed image: {patch.shape}")
        else:
            blend_with_mask(
                patch,
                None,
                caption_suffix=f"{i}/{self._compute_batchsize_from_batch(batch)}: {patch.shape}",
                st_key=f"b{i}",
            )
