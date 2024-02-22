"""
Avoid circular imports by not importing other mmm stuff here.
"""

from __future__ import annotations  # when types cannot be written from top to bottom
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Any, Literal, Generic, TypeVar

from pydantic import BaseModel, Field, model_validator, ConfigDict, field_validator

import torchvision.transforms.functional as TF
from torchvision.datasets.folder import default_loader

from .typing_utils import rgbnumpy_to_base64, get_colors


class MTLType(BaseModel):
    """
    Generic class for all MTL types
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_untyped(cls, legacy_obj):
        raise NotImplementedError()

    def to_untyped(self):
        raise NotImplementedError()

    def _repr_html_(self):
        return f"<p>{self.__class__.__name__} did not implement _repr_html_</p>"

    def _st_repr_(self) -> Any | None:
        """
        Default representation for streamlit
        """
        import streamlit as st

        # Uses the html representation _repr_html_ by default
        st.write(self)


class ImageAnnotation(MTLType):
    annotation_type: Literal["image_caption", "multiclass_segmentation", "multilabel_segmentation"]


class ImageCaption(ImageAnnotation):
    annotation_type: Literal["image_caption"] = "image_caption"
    caption: str = Field(..., description="The content of the image caption")

    @staticmethod
    def build_example():
        return ImageCaption(caption="A smiling female astronaut. She wears a space suit and holds a helmet.")


class MultiClassSegmentation(ImageAnnotation):
    annotation_type: Literal["multiclass_segmentation"] = "multiclass_segmentation"
    data: torch.Tensor = Field(..., description="The mask")

    @classmethod
    def from_untyped(cls, legacy_obj):
        return cls(data=legacy_obj)

    def to_untyped(self):
        return self.data

    @staticmethod
    def build_example(for_image: RGBImage):
        mask = torch.zeros(for_image.data.shape[1:]).long()
        mask[100:400, 100:400] = 1
        mask[200:300, 300:400] = 2
        return MultiClassSegmentation(data=mask)

    @field_validator("data")
    @classmethod
    def check_tensor(cls, v: torch.Tensor):
        assert isinstance(v, torch.LongTensor), "Mask should be a long tensor"
        assert len(v.shape) == 2, "Mask should be (H, W)"
        return v

    def _repr_html_(self):
        # Convert the mask to an RGB representation between 0 and 255.
        np_img = self.data.numpy()
        np_img = np_img.astype(np.float32) * (255 / np.max(np_img))
        # Add three channels
        np_img = np.stack([np_img] * 3, axis=-1)
        img, desc = (
            rgbnumpy_to_base64(np_img),
            f"type: {self.data.dtype}, shape: {self.data.shape}",
        )
        return f"""
        <figure style="text-align: center">
            <img src="{img}" alt="{desc}">
            <figcaption style="text-align: center">{desc}</figcaption>
        </figure>
        """


class MultiLabelSegmentation(ImageAnnotation):
    annotation_type: Literal["multilabel_segmentation"] = "multilabel_segmentation"
    data: torch.Tensor = Field(..., description="The mask")
    class_names: list[str] | None = None

    @classmethod
    def from_untyped(cls, legacy_obj):
        return cls(data=legacy_obj)

    def to_untyped(self):
        return self.data

    @staticmethod
    def build_example(for_image: RGBImage):
        mask = torch.zeros([4] + list(for_image.data.shape[1:])).long()
        # mask[0] is background
        mask[1, 100:400, 100:400] = 1
        mask[2, 200:300, 300:400] = 1
        mask[3, 100:400, 200:350] = 1
        # mask[3, 200:300, 300:400] = 1
        return MultiLabelSegmentation(data=mask, class_names=["background", "class1", "class2", "class3"])

    @field_validator("data")
    @classmethod
    def check_tensor(cls, v: torch.Tensor):
        assert isinstance(v, torch.LongTensor), "Mask should be a long tensor"
        assert len(v.shape) == 3, "Mask should be (#classnames, H, W)"
        return v

    def _repr_html_(self):
        np_mask = self.data.numpy()
        html = ""
        for class_i in range(self.data.shape[0]):
            # Convert the mask to an RGB representation between 0 and 255.
            np_img = np_mask[class_i]
            np_img = np_img.astype(np.float32) * (255 / np.max(np_img))
            # Add three channels
            np_img = np.stack([np_img] * 3, axis=-1)
            img, desc = (
                rgbnumpy_to_base64(np_img),
                f"type: {self.data.dtype}, shape: {self.data.shape}",
            )
            html += f"""
            <figure style="text-align: center">
                <img src="{img}" alt="{desc}">
                <figcaption style="text-align: center">{desc}</figcaption>
            </figure>
            """
        return html

    def st_class_selector(self, st_prefix: str) -> dict[str, bool]:
        import streamlit as st

        assert self.class_names is not None, "Class names should be set for selecting classes"

        colors = get_colors(len(self.class_names))
        res = {}
        for i, c in enumerate(self.class_names):
            indices, nums = torch.unique(self.data[i], return_counts=True)
            # Count 1s in the mask
            num_pixels = nums[indices == 1][0] if 1 in indices else 0
            # num_pixels = class_counts[i] if i in class_indices else 0

            if num_pixels > 0:
                colorrgb = f"{int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)}"
                col1, col2 = st.columns(2)
                with col1:
                    show = st.checkbox(
                        c,
                        key=f"{st_prefix}check{c}",
                        value=(len(self.class_names) == 1) or (i >= 1),
                    )
                with col2:
                    st.markdown(
                        f"<span style='color: rgb({colorrgb})'>{num_pixels} pixels</span>",
                        unsafe_allow_html=True,
                    )
                res[i] = show
            else:
                res[i] = False
        return res

    def blend_with_image(
        self,
        im: np.ndarray,
        classes: dict[int, bool],
    ) -> None:
        im = (im.copy() / 255.0).astype(np.float32)
        mask = self.data.numpy().copy()
        assert len(mask.shape) == 3
        colors = get_colors(len(classes))

        for class_i, overlay_class in classes.items():
            if overlay_class:
                color_overlay = im.copy()
                color_overlay[mask[class_i] == 1] = np.asarray(colors[class_i], dtype=np.float32)
                im = cv2.addWeighted(im, 0.7, color_overlay.astype(np.float32), 0.3, 0)

        return (im * 255).astype(np.uint8)


class RGBImage(MTLType):
    """
    Generic image sample with 3 channels
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    data: torch.Tensor = Field(..., description="The image")

    @classmethod
    def from_path(cls, img_path: Path):
        return cls(data=TF.to_tensor(default_loader(img_path)))

    @staticmethod
    def build_example():
        from skimage.data import astronaut

        return RGBImage(data=TF.to_tensor(astronaut()))

    @field_validator("data")
    @classmethod
    def check_tensor(cls, v: torch.Tensor):
        assert isinstance(v, torch.FloatTensor), "Image should be a float tensor"
        assert v.shape[0] == 3, "Image should have 3 channels"
        assert len(v.shape) == 3, "Image should be (RGB, H, W)"
        return v

    @classmethod
    def from_untyped(cls, legacy_obj):
        return cls(data=legacy_obj["image"])

    def to_untyped(self):
        return self.data

    def to_numpy(self) -> np.ndarray:
        return self.data.numpy().transpose(1, 2, 0) * 255

    def get_description(self) -> str:
        res = f"type: {self.data.dtype}, shape: {self.data.shape}<br />"
        res += f"Min: {torch.min(self.data):.2f}, Max: {torch.max(self.data):.2f}"
        return res

    def _repr_html_(self):
        img, desc = rgbnumpy_to_base64(self.to_numpy()), self.get_description()
        return f"""
        <figure style="text-align: center">
            <img src="{img}" alt="{desc}">
            <figcaption style="text-align: center">{desc}</figcaption>
        </figure>
        """


# Typevar that can be any child of ImageAnnotation
ImageAnnotationType = TypeVar("ImageAnnotationType", bound=ImageAnnotation)


class AnnotatedImage(MTLType, Generic[ImageAnnotationType]):
    """
    Generic classification sample
    """

    image: RGBImage = Field(..., description="The image")
    # Use default_factory just to be sure with the mutable default
    annotations: list[ImageAnnotationType] = Field(default_factory=lambda: [], description="The annotations")

    original_size: tuple[int, int] | None = Field(None, description="The size to be used for metric computation")

    @model_validator(mode="after")
    def check_annotation_compatibility(self):
        for anno in self.annotations:
            if isinstance(anno, MultiClassSegmentation):
                assert self.image.data.shape[1:] == anno.data.shape, "Annotation shape should match image shape"
            elif isinstance(anno, MultiLabelSegmentation):
                assert self.image.data.shape[1:] == anno.data.shape[1:], "Annotation shape should match image shape"
            elif isinstance(anno, ImageCaption):
                assert anno.caption, "Caption should always contain something"
            else:
                raise NotImplementedError(f"Unknown annotation type {anno.annotation_type}")
        # self.mask.data.shape == self.image.data.shape[1:]

    @staticmethod
    def build_example():
        img = RGBImage.build_example()
        return AnnotatedImage(
            image=img,
            annotations=[
                MultiClassSegmentation.build_example(for_image=img),
                ImageCaption.build_example(),
            ],
        )

    @staticmethod
    def blend_with_multiclass_mask(
        im: np.ndarray,
        mask: np.ndarray,
        classes: dict[int, bool],
    ) -> None:
        # Writing annotations on top of the image without copying first can lead to errors
        im = im.copy().astype(np.float32) / 255.0
        assert len(mask.shape) == 2
        colors = get_colors(len(classes))
        color_overlay = im.copy()
        for class_i, overlay_class in classes.items():
            if overlay_class:
                color_overlay[mask == class_i] = np.asarray(colors[class_i], dtype=np.float32)
        im = (cv2.addWeighted(im, 0.7, color_overlay.astype(np.float32), 0.3, 0) * 255).astype(np.uint8)
        return im

    def build_repr(
        self,
        viewer: Literal["bare_html", "streamlit", "jupyter"] = "bare_html",
        st_prefix: str = "",
    ) -> str:
        """
        Reusable functionality for html-based visualization.
        """
        img = self.image.to_numpy().copy().astype(np.uint8)

        for mcmask in filter(lambda x: x.annotation_type == "multiclass_segmentation", self.annotations):
            mask_classes = {i: True for i in range(torch.max(mcmask.data).item() + 1)}
            mask_classes[0] = False
            img = self.blend_with_multiclass_mask(img, mcmask.data.numpy(), mask_classes)

        for i, mlmask in enumerate(
            filter(
                lambda x: x.annotation_type == "multilabel_segmentation",
                self.annotations,
            )
        ):
            mlmask: MultiLabelSegmentation
            if viewer == "streamlit":
                mask_classes = mlmask.st_class_selector(st_prefix=f"{st_prefix}mlmask_{i}")
            else:
                mask_classes = {i: True for i in range(mlmask.data.shape[0])}
                mask_classes[0] = False
            img = mlmask.blend_with_image(img, mask_classes)

        captions = [x.caption for x in self.annotations if x.annotation_type == "image_caption"]

        desc = self.image.get_description()

        img = f"""
        <figure style="text-align: center">
            <img src="{rgbnumpy_to_base64(img)}" style="width: 50%">
            <figcaption style="text-align: center">{desc}</figcaption>
        </figure>
        """
        return img + "".join([f'<p style="text-align: center">Caption: {x}</p>' for x in captions])

    def _repr_html_(self, st_prefix: str = ""):
        return self.build_repr(viewer="bare_html", st_prefix=st_prefix)

    def _st_repr_(self, st_prefix: str = "") -> Any | None:
        return self.build_repr(viewer="streamlit", st_prefix=st_prefix)

    @classmethod
    def from_untyped(cls, legacy_obj: dict[str, Any]) -> AnnotatedImage:
        annotations = []

        if "captions" in legacy_obj:
            annotations.extend([ImageCaption(caption=text) for text in legacy_obj["captions"]])

        return cls(
            image=RGBImage.from_untyped({"image": legacy_obj["image"]}),
            annotations=annotations,
        )


__all__ = [
    "RGBImage",
    "ImageCaption",
    "MultiClassSegmentation",
    "AnnotatedImage",
    "MetaInfo",
]
