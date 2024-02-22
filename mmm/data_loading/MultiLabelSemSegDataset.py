from functools import partial

import random
import json
from typing import Any, List, Optional, Callable, Dict
from jinja2 import Template
import re

from pathlib import Path
import tempfile
import imageio.v3 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from mmm.mtl_types import RGBImage, MultiLabelSegmentation, AnnotatedImage
from mmm.typing_utils import get_colors
from mmm.data_loading.MTLDataset import MTLDataset, SrcCaseType
from mmm.typing_utils import rgbnumpy_to_base64
from mmm.labelstudio_ext.utils import (
    binary_mask_to_result,
    brush_annotation_to_npy,
)
import torchvision.transforms.functional as F

LS_TEMPLATE = r"""
<View>
<Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
<HyperText name="Context" value="$htmlmeta" />
<BrushLabels name="segmentations" toName="image">
{% for class_name, color in classes.items() %}
<Label value="{{class_name}}" background="rgba({{color[0]}}, {{color[1]}}, {{color[2]}}, 0.2)"/>
{% endfor %}
</BrushLabels>
</View>
"""


def masks_from_ls(
    results: list[dict],
    shape: tuple[int, int],
    class_names: list[str],
    unlabeled_value=-1,
) -> np.ndarray:
    # Extract dict[classname, mask] from the annotation
    # if "annotations" in ls_task and len(ls_task["annotations"]) > 0:
    # anno = random.choice(ls_task["annotations"])
    masks_dict = {classname: mask for classname, mask in [brush_annotation_to_npy(result) for result in results]}
    if shape is None:
        shape = masks_dict[list(masks_dict.keys())[0]].shape
    masks = np.zeros([len(class_names), *shape], dtype=np.int64)
    labeled_pixels = np.zeros(shape, dtype=np.int64)

    for classname, mask in masks_dict.items():
        class_index = class_names.index(classname)
        labeled_pixels[mask > 0] += 1
        torchmask = (torch.from_numpy(mask) > 0).long()
        # torchmask[torchmask == 0] = unlabeled_value
        # The region is considered labeled if it is labeled with any class
        masks[class_index, ...] = torchmask
        # masks[~class_index, torchmask == 1] = 0

    # The region is considered unlabeled if it is unlabeled for all classes
    masks[:, labeled_pixels == 0] = unlabeled_value

    return masks


def multilabel_case_from_ls(npy_case: dict[str, Any]) -> dict[str, Any]:
    npy_case["image"] = F.to_tensor(npy_case["image"])
    if "masks" in npy_case:
        npy_case["masks"] = torch.from_numpy(npy_case["masks"])
    return npy_case


def ls_create_mlsemseg_template(class_names: list[str]) -> str:
    class_colors = get_colors(len(class_names))
    template = Template(LS_TEMPLATE)
    return template.render(
        classes={
            class_name: [int(r * 255), int(g * 255), int(b * 255)]
            for class_name, (r, g, b) in zip(class_names, class_colors)
        },
    )


def mtl_case_to_ls(mtl_case: Dict[str, Any], class_names: list[str], htmlmeta: str = "") -> Dict[str, Any]:
    """
    Labelstudio likes to get files in the form of links.

    This function converts the image to a base64 string by default.
    If you have a link, use the "imagelink" key to overwrite this behaviour.

    For example:
    {
        "imagelink": "https://picsum.photos/id/1/200/300",
        "masks": np.random.randint(0, 2, [2, 200, 300]),
    }
    """
    # Convert the [C, H, W] image to [H, W, C] numpy
    if "imagelink" not in mtl_case:
        npy_img = np.transpose(mtl_case["image"].numpy() * 255.0, [1, 2, 0]).astype(np.uint8)
        mtl_case["imagelink"] = rgbnumpy_to_base64(npy_img)

    if not htmlmeta and "meta" in mtl_case:
        htmlmeta = json.dumps(mtl_case["meta"], default=lambda o: str(o))

    return {
        "data": {
            "image": mtl_case["imagelink"],
            "htmlmeta": htmlmeta,
        },
        "annotations": [
            {
                "result": [
                    binary_mask_to_result(mtl_case["masks"][i].numpy(), class_names[i], "segmentations")
                    for i in range(len(mtl_case["masks"]))
                    if torch.max(mtl_case["masks"][i]).item() > 0
                ]
            }
        ],
    }


class MultiLabelSemSegDataset(MTLDataset):
    """
    Requires an image and a mask where each pixel can have zero or all classes.

    "image" shape: [C, H, W]
    "masks" shape: [#class_names, H, W]
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
        super().__init__(src_ds, ["image", "masks"], ["original_size", "meta"], *args, **kwargs)

    def set_classes_for_visualization(self, classes: List[str]):
        self.class_names = classes

    def verify_case(self, d: Dict[str, Any]) -> None:
        super().verify_case(d)
        self.assert_image_data_assumptions(d["image"])
        # For example, MONAI tends to load some specific subclass of Tensor, exclude it!
        assert torch.is_tensor(d["masks"]), "mask should be of type tensor"
        assert isinstance(d["masks"], torch.LongTensor)
        assert list(d["masks"].shape) == [len(self.class_names)] + list(d["image"].shape[1:])
        assert torch.max(d["masks"]).item() <= 1, "Multilabel targets should be binary"
        assert len(d["image"].shape) == len(d["masks"].shape), "Each class should have its own channel"

    def st_case_viewer(self, case: Dict[str, Any], i: int = -1) -> None:
        from mmm.logging.st_ext import stw, st

        stw("### Untransformed image:")
        img, masks = case.pop("image"), case.pop("masks")

        anno_img = AnnotatedImage(
            image=RGBImage(data=img),
            annotations=[MultiLabelSegmentation(data=masks, class_names=self.class_names)],
        )
        stw(anno_img, st_prefix=f"img{i}")
        stw(case, st_prefix=f"case{i}")

    def _compute_batchsize_from_batch(self, batch: Dict[str, Any]) -> int:
        return batch["image"].shape[0]

    def _visualize_batch_case(self, batch: Dict[str, Any], i: int) -> None:
        from mmm.logging.st_ext import stw, st

        if "meta" in batch:
            st.write(batch["meta"][i])
        stw(
            AnnotatedImage(
                image=RGBImage(data=batch["image"][i]),
                annotations=[MultiLabelSegmentation(data=batch["masks"][i], class_names=self.class_names)],
            ),
            st_prefix=f"b{i}",
        )
        # mask_uniques = torch.unique(patch_mask)
        # blend_with_mask(
        #     patch,
        #     patch_mask,
        #     caption_suffix=f"{i}/{self._compute_batchsize_from_batch(batch)}\nUniques: {mask_uniques}",
        #     classes=self.class_names,
        #     st_key=f"b{i}",
        # )

    def ls_create_template(self) -> str:
        return ls_create_mlsemseg_template(self.class_names)

    def ls_get_case(self, mtl_case: Dict[str, Any], htmlmeta: str = "") -> Dict[str, Any]:
        return mtl_case_to_ls(mtl_case, self.class_names, htmlmeta)

    @classmethod
    def from_ls(cls, ls_ds, ls_tpl: str, *args, **kwargs):
        """
        Creates a dataset from a labelstudio template.
        """
        class_names: list[str] = re.findall(r'<Label value="([^"]*)"', ls_tpl)
        return cls(
            src_ds=ls_ds,
            src_transform=cls.ls_src_transform(class_names),
            class_names=class_names,
            *args,
            **kwargs,
        )

    @staticmethod
    def ls_src_transform(class_names: list[str], unlabeled_value=-1) -> Dict[str, Any]:
        """
        Transforms a labelstudio case into an MTL case.

        If there are multiple annotations a random annotation is selected.
        This might be a problem if you use caching!
        """

        return partial(
            multilabel_case_from_ls,
            class_names=class_names,
            unlabeled_value=unlabeled_value,
        )
