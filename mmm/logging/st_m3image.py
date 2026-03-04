"""
Streamlit component for displaying an image with annotations.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np

try:
    import streamlit as st
except ImportError:
    if not TYPE_CHECKING:
        st = None
    else:
        raise  # Happens during type checking and avoids false positives
import torch
from m3_sdk.utils import rgbnumpy_to_base64
from pydantic import Field

from mmm.BaseModel import BaseModel
from mmm.resources import RESOURCE_PATH


class ImageOverlay(BaseModel):
    data: str
    classname: str

    @staticmethod
    def rgba_from_gray(mask: np.ndarray) -> np.ndarray:
        assert len(mask.shape) == 2, "Mask must be a 2D array"
        if mask.dtype == bool:
            alpha_channel = mask.astype(np.uint8) * 255
        else:
            alpha_channel = mask.astype(np.uint8)
        white_channel = np.ones_like(alpha_channel, dtype=np.uint8) * 255
        black_channel = np.zeros_like(alpha_channel, dtype=np.uint8)
        return np.stack([white_channel, black_channel, black_channel, alpha_channel], axis=-1)


class ImageOverlays(BaseModel):
    overlays: list[ImageOverlay]
    overlay_type: str = Field(default="Overlay", description="E.g., 'Predictions', 'Ground truth'")


class Image2D(BaseModel):
    unique_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    img: str
    overlay_groups: list[ImageOverlays] = []
    desc: str | None = None
    caption: str | None = Field(default=None, description="Short caption displayed below each image.")

    @staticmethod
    def from_tensor(
        img: torch.Tensor, masks: list[torch.Tensor] | None = None, class_names: list[str] | None = None, **kwargs
    ):
        return Image2D.from_numpy(
            (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
            [mask.numpy() for mask in masks] if masks is not None else None,
            class_names,
            **kwargs,
        )

    @staticmethod
    def from_numpy(
        img: np.ndarray, masks: list[np.ndarray] | None = None, class_names: list[str] | None = None, **kwargs
    ):
        if "caption" not in kwargs:
            kwargs["caption"] = ""
        else:
            kwargs["caption"] += "<br>"

        kwargs["caption"] += f"<span style='color:DarkSeaGreen'>{img.shape}</span><br>"
        kwargs["caption"] += f"dtype={img.dtype}<br>min={img.min().item()}<br>max={img.max().item()}"

        if class_names is None:
            class_names = []

        if masks is not None:
            assert len(masks) == len(class_names)

            overlays = []
            for mask, class_name in zip(masks, class_names):
                assert mask.shape == img.shape[:2], f"{mask.shape=} must match {img.shape[:2]=}"
                if mask.sum() == 0:
                    continue
                rgba = ImageOverlay.rgba_from_gray(mask)
                overlays.append(ImageOverlay(data=rgbnumpy_to_base64(rgba), classname=class_name))
        else:
            overlays = None

        assert img.dtype == np.uint8, "Image should be 0-255 uint8 numpy array"
        return Image2D(
            img=rgbnumpy_to_base64(img),
            overlay_groups=[ImageOverlays(overlays=overlays)] if overlays is not None else [],
            **kwargs,
        )


class M3Image:
    class Data(BaseModel):
        images: list[Image2D]
        gallery_height: str = "80vh"
        group_meta: dict | None = None

    def __init__(self):
        self.component = None  # Initially set to None to avoid errors during import when Streamlit is not available

    @staticmethod
    def load_component():
        from streamlit.components.v2 import component

        html = RESOURCE_PATH.joinpath("m3image.html").read_text()
        js = RESOURCE_PATH.joinpath("m3image.js").read_text()
        css = RESOURCE_PATH.joinpath("m3image.css").read_text()
        print("Loading M3 Image component")
        return component(
            name="m3_image",
            html=html,
            js=js,
            css=css,
        )

    def __call__(self, key: str, data: M3Image.Data):
        if self.component is None:
            self.component = self.load_component()

        if data.group_meta is not None:
            st.json(data.group_meta, expanded=False)
        return self.component(
            key=key,
            data=data.model_dump(),
        )


m3_image = M3Image()


if __name__ == "__main__":
    import numpy as np
    import requests
    from m3_sdk.interactive import models, utils
    from skimage.data import astronaut, camera, chelsea

    from mmm.logging.st_ext import st, stw

    def label_image(img: str) -> ImageOverlays:
        response = requests.post(
            "http://localhost:8000/peft/predict",
            json=dict(
                subjects=[models.Subject(data={"image": img}).model_dump()],
                label_config={"xml": '<View><Image name="image" /><BrushLabels name="coco" toName="image" /></View>'},
                for_labels={"coco": "cocoseg"},
            ),
        )
        pred = models.Prediction.model_validate(response.json()["predictions"][0])
        mask, classnames = utils.convert_results_to_seglabel(pred.result)
        overlays = [
            ImageOverlay(data=utils.rgbnumpy_to_base64(ImageOverlay.rgba_from_gray(mask == i)), classname=class_name)
            for i, class_name in enumerate(classnames)
        ]
        return ImageOverlays(
            overlays=overlays,
            overlay_type="Predictions",
        )

    def add_random_shape_overlay(img: np.ndarray) -> ImageOverlays:
        from skimage.draw import random_shapes

        mask = random_shapes(
            img.shape[:2], max_shapes=5, min_shapes=3, max_size=100, allow_overlap=True, num_channels=1
        )[0][..., 0]
        overlay_img = ImageOverlay(
            data=utils.rgbnumpy_to_base64(ImageOverlay.rgba_from_gray(mask - 255)),
            classname="RandomShapes",
        )
        return ImageOverlays(
            overlays=[overlay_img],
            overlay_type="RandomShapes",
        )

    def build_image_2d(np_img: np.ndarray, desc: str) -> Image2D:
        res = Image2D(img=utils.rgbnumpy_to_base64(np_img), desc=desc, caption=r"<br>\nlongtext".join(desc.split(" ")))
        res.overlay_groups.append(label_image(res.img))
        res.overlay_groups.append(add_random_shape_overlay(np_img))
        return res

    num_images = st.number_input(label="Number of images", min_value=1, max_value=10, value=2)
    if st.toggle("Toggle to reload component"):
        images = [
            build_image_2d(np_img, desc)
            for np_img, desc in [
                (astronaut(), "Astronaut image"),
                (chelsea(), "Chelsea the cat image"),
                # (camera(), "Camera image"),
            ]
            * num_images
        ]

        st.title("M3 Image Component Demo")

        output1 = m3_image(
            key="demo_image",
            data=M3Image.Data(images=images, group_meta={"purpose": "Demo of M3 Image component with overlays"}),
        )
        st.write(output1)
