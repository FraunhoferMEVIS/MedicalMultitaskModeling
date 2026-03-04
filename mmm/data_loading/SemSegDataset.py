import json
from typing import Any, Dict

import torch
from torch.utils.data import Dataset

from mmm.settings import mtl_settings

from .MTLDataset import MTLDataset, SrcCaseType


class SemSegDataset(MTLDataset):
    """
    Contains a single pixel-dense mask for one or more images.
    """

    def __init__(
        self,
        src_ds: Dataset[SrcCaseType],
        class_names: list[str] | None = None,
        *args,
        **kwargs,
    ) -> None:
        assert class_names is not None, "Segmentation dataset needs class names"
        self.class_names: list[str] | None = class_names
        super().__init__(src_ds, *args, **kwargs)

    @staticmethod
    def get_mandatory_keys() -> list[str]:
        return super(SemSegDataset, SemSegDataset).get_mandatory_keys() + ["image", "label"]

    @staticmethod
    def get_optional_keys() -> list[str]:
        # original_shape is optional, but used for correctly computing metrics
        return super(SemSegDataset, SemSegDataset).get_optional_keys() + ["original_shape"]

    def set_classes_for_visualization(self, classes: list[str]):
        self.class_names = classes

    def verify_case(self, d: dict[str, Any]) -> None:
        super().verify_case(d)
        self.assert_image_data_assumptions(d["image"])
        # For example, MONAI tends to load some specific subclass of Tensor, exclude it!
        assert torch.is_tensor(d["label"]), "mask should be of type tensor"
        assert isinstance(d["label"], torch.LongTensor)
        assert d["label"].shape == d["image"].shape[1:], f"{d['label'].shape=} should match {d['image'].shape[1:]=}"
        assert torch.max(d["label"]).item() < len(self.class_names), "There need to be more class names"
        # Labels should be >= 0 or mtl_settings.ignore_class_value
        assert (
            min({x.item() for x in d["label"].unique()} - {mtl_settings.ignore_class_value}) >= 0
        ), "Labels should be >= 0 or mtl_settings.ignore_class_value"
        assert len(d["image"].shape) == (len(d["label"].shape) + 1), "all labels should be in a one-dim tensor"

    def st_case_viewer(self, ls: list[dict[str, Any]], i: int = -1) -> None:
        from mmm.logging.st_ext import Image2D, M3Image, m3_image, st

        assert self.class_names is not None, "Class names need to be set for visualization"
        m3_image(
            data=M3Image.Data(
                images=[
                    Image2D.from_tensor(
                        img=d["image"],
                        masks=[d["label"] == class_idx for class_idx in range(len(self.class_names))],
                        class_names=self.class_names,
                        desc=json.dumps(d["meta"], indent=2, default=str) if "meta" in d else None,
                        caption=f"""<span style='color:orange'>{d.get('meta', {}).get('context', ())}</span>
<span style='color:cyan'>Classes: {', '.join([self.class_names[c] for c in torch.unique(d['label'])])}</span>""",
                    )
                    for d in ls
                ]
            ),
            key=f"img{i}_original",
        )

    def _compute_batchsize_from_batch(self, batch: Dict[str, Any]) -> int:
        return batch["image"].shape[0]

    def _visualize_batch_case(self, batch: Dict[str, Any], i: int) -> None:
        from mmm.logging.st_ext import blend_with_mask, st

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
