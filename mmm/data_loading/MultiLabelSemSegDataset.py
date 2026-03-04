import json
import re
from functools import partial
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from mmm.data_loading.MTLDataset import MTLDataset, SrcCaseType


class MultiLabelSemSegDataset(MTLDataset):
    """
    Requires an image and a mask where each pixel can have zero or all classes.

    "image" shape: [C, H, W]
    "masks" shape: [#class_names, H, W]
    """

    def __init__(
        self,
        src_ds: Dataset[SrcCaseType],
        class_names: list[str] = None,
        *args,
        **kwargs,
    ) -> None:
        assert class_names is not None, "Segmentation dataset needs class names"
        self.class_names: list[str] = class_names
        super().__init__(src_ds, *args, **kwargs)

    @staticmethod
    def get_mandatory_keys():
        return super(MultiLabelSemSegDataset, MultiLabelSemSegDataset).get_mandatory_keys() + ["image", "masks"]

    @staticmethod
    def get_optional_keys():
        return super(MultiLabelSemSegDataset, MultiLabelSemSegDataset).get_optional_keys() + ["original_size"]

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

    def st_case_viewer(self, ls: list[dict[str, Any]], i: int = -1) -> None:
        from mmm.logging.st_ext import Image2D, M3Image, m3_image, st

        st.markdown("### Untransformed image:")
        # img, masks = case["image"], case["masks"]

        # m3_image_from_tensor(img, masks, self.class_names, key=f"img{i}_original")

        m3_image(
            data=M3Image.Data(
                images=[
                    Image2D.from_tensor(
                        img=d["image"],
                        masks=d["masks"] > 0,
                        class_names=self.class_names,
                        desc=json.dumps(d["meta"], indent=2, default=str) if "meta" in d else None,
                    )
                    for d in ls
                ]
            ),
            key=f"img{i}_original",
        )

        # stw(case, st_prefix=f"case{i}")

    def _compute_batchsize_from_batch(self, batch: Dict[str, Any]) -> int:
        return batch["image"].shape[0]

    def _visualize_batch_case(self, batch: Dict[str, Any], i: int) -> None:
        from mmm.logging.st_ext import m3_image_from_tensor, st, stw

        if "meta" in batch:
            st.write(batch["meta"][i])

        img, masks = batch["image"][i], batch["masks"][i]
        m3_image_from_tensor(img, masks, self.class_names, key=f"img{i}_batch")
