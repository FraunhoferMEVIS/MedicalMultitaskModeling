"""
Trainable simulated vision tasks with a low loading time and memory requirements
for automatic testing.
"""

from __future__ import annotations
from enum import Enum
from itertools import product

import random
from typing import Callable, Dict, Tuple, Any, List, Type, Union, Literal
import numpy as np
from PIL.Image import Image, new as pil_new
from PIL.ImageDraw import ImageDraw, Draw
from pydantic import Field

import torchvision.transforms as torch_transforms

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.BaseModel import BaseModel
from mmm.bucketizing import Bucket, BucketConfig
from mmm.data_loading.utils import TransformedSubset
from mmm.transforms import ApplyToKey, KeepOnlyKeysInDict
from mmm.data_loading.SemSegDataset import SemSegDataset
from mmm.data_loading.DetectionDataset import DetectionDataset
from mmm.data_loading.ClassificationDataset import ClassificationDataset
from mmm.data_loading.ImageCaptionDataset import ImageCaptionDataset
from mmm.data_loading.MultilabelClassificationDataset import (
    MultilabelClassificationDataset,
)
from mmm.data_loading.MultiLabelSemSegDataset import MultiLabelSemSegDataset


def get_random_bounds(min_max_size=(0.1, 0.3), canvas_size=(224, 224)) -> Tuple[int, int, int, int]:
    width = random.randint(int(canvas_size[0] * min_max_size[0]), int(canvas_size[0] * min_max_size[1]))
    height = random.randint(int(canvas_size[1] * min_max_size[0]), int(canvas_size[1] * min_max_size[1]))
    x = random.randint(0, canvas_size[0] - width)
    y = random.randint(0, canvas_size[1] - height)
    return x, y, x + width, y + height


def track_difference(img: Image, drawing_function: Callable[[ImageDraw], Any]):
    """
    Given an existing image and the current mask, draws a new object and adds the difference image to the mask.
    """
    original_im = np.array(img)
    draw = Draw(img)
    drawing_function(draw)

    minus_im = np.array(img) != (original_im)

    # If a pixel is changed in any colorchannel, the mask gets that value
    changed_pixels = np.logical_or(*[minus_im[:, :, channel_dim] for channel_dim in range(minus_im.shape[2])])

    return img, changed_pixels


class CanvasConfig(BaseModel):
    canvas_size: Tuple[int, int] = (512, 512)
    bg_color: Tuple[int, int, int] = (128, 128, 128)
    obj_size_range: Tuple[float, float] = (0.1, 0.3)


class ShapeColor(str, Enum):
    Charcoal = "28536B"
    RosyBrown = "C2948A"
    Khaki = "BBB193"
    TurkeyRed = "A30000"
    Asparagus = "659157"
    Moonstone = "69A2B0"

    @staticmethod
    def hex_to_rgb(hex: str) -> Tuple[int, int, int]:
        if hex.startswith("#"):
            hex = hex[1:]
        return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def toRGB(c: ShapeColor) -> Tuple[int, int, int]:
        return ShapeColor.hex_to_rgb(c.value)


class BaseObject(BaseModel):
    description: str
    in_area: Tuple[int, int, int, int]

    @staticmethod
    def sample_object_for_canvas(settings: CanvasConfig) -> BaseObject:
        raise NotImplementedError

    def draw_shape(self, canvas: ImageDraw) -> ImageDraw:
        raise NotImplementedError


class Ellipse(BaseObject):
    obj_type: Literal["ellipse"] = "ellipse"
    fillcolor: ShapeColor

    @staticmethod
    def sample_object_for_canvas(settings: CanvasConfig) -> BaseObject:
        fillcolor = random.choice(list(ShapeColor))
        outer_area = get_random_bounds(min_max_size=settings.obj_size_range, canvas_size=settings.canvas_size)
        return Ellipse(
            description=f"A {fillcolor.name.lower()} ellipse",
            in_area=outer_area,
            fillcolor=fillcolor,
        )

    def draw_shape(self, canvas: ImageDraw) -> ImageDraw:
        canvas.ellipse(self.in_area, fill=ShapeColor.toRGB(self.fillcolor))
        return canvas


class Rectangle(BaseObject):
    obj_type: Literal["rectangle"] = "rectangle"
    fillcolor: ShapeColor

    @staticmethod
    def sample_object_for_canvas(settings: CanvasConfig) -> BaseObject:
        fillcolor = random.choice(list(ShapeColor))
        outer_area = get_random_bounds(min_max_size=settings.obj_size_range, canvas_size=settings.canvas_size)
        return Rectangle(
            description=f"A {fillcolor.name.lower()} rectangle",
            in_area=outer_area,
            fillcolor=fillcolor,
        )

    def draw_shape(self, canvas: ImageDraw) -> ImageDraw:
        canvas.rectangle(self.in_area, fill=ShapeColor.toRGB(self.fillcolor))
        return canvas


ObjectTypes = BaseObject.__subclasses__()


class ShapeDataset(Dataset[Dict[str, Any]]):
    """
    Dataset which uses PIL to draw simple shapes.

    `get_class_names()`'s index corresponds to the value in the segmentation masks.
    """

    class Config(BaseModel):
        N: int = 10
        canvas: CanvasConfig = CanvasConfig()
        num_objects_per_scene_range: Tuple[int, int] = (3, 10)
        # canvas_buckets: BucketConfig
        object_buckets: BucketConfig = BucketConfig(
            exactly_one_match=True,
            buckets=[
                Bucket(expression=".*ell.*", description="ellipse"),
                Bucket(expression=".*rect.*", description="rectangle"),
            ],
        )

    def __init__(self, c: Config) -> None:
        self.config = c

        # Each case stores several objects
        self.case_list = [self.generate_objects(self.config.num_objects_per_scene_range) for _ in range(self.config.N)]

    def get_class_names(self) -> List[str]:
        # return ["Background"] + [" ".join([color.value, shape.value])
        #                          for color, shape in product(ShapeColor, ShapeType)]
        return self.config.object_buckets.get_class_names()

    def generate_objects(self, num_objects_range: Tuple[int, int]) -> List[BaseObject]:
        num_objects = random.randint(num_objects_range[0], num_objects_range[1])
        object_types: List[Type[BaseObject]] = [random.choice(ObjectTypes) for _ in range(num_objects)]

        # coords = [get_random_bounds(
        #     min_max_size=self.config.canvas.obj_size_range, canvas_size=self.config.canvas.canvas_size
        # ) for _ in range(num_objects)]
        objects = [o.sample_object_for_canvas(self.config.canvas) for o in object_types]

        return objects

    @staticmethod
    def cohort_transform():
        return torch_transforms.Compose(
            [
                ApplyToKey(torch_transforms.ToTensor(), "image"),
                ApplyToKey(torch.LongTensor, "label"),
                ApplyToKey(torch.LongTensor, "masks"),
                ApplyToKey(torch.FloatTensor, "boxes"),
                ApplyToKey(torch.LongTensor, "labels"),
            ]
        )

    def __iter__(self):
        for objects in self.case_list:
            yield self._generate_case_for_objects(objects)

    def __len__(self) -> int:
        return self.config.N

    def _generate_case_for_objects(self, scene_elements: List[BaseObject]) -> Dict[str, Any]:
        im: Image = pil_new("RGB", self.config.canvas.canvas_size[::-1], self.config.canvas.bg_color)
        mask = np.zeros(shape=self.config.canvas.canvas_size, dtype=np.int64)
        masks = np.zeros(
            shape=[len(self.config.object_buckets.get_class_names()) + 1] + list(self.config.canvas.canvas_size),
            dtype=np.int64,
        )
        masks[0, ...] = 1
        boxes = []
        box_labels = []
        class_names = []
        for obj in scene_elements:
            # element_label = self._mask_lookup[(shape, color)]
            class_name, element_label = self.config.object_buckets.get_bucket_name(obj.description)

            im, changed_pixels = track_difference(im, obj.draw_shape)
            mask[changed_pixels] = element_label + 1
            masks[element_label + 1, changed_pixels] = 1  # Set object at changed_pixels to 1
            masks[0, changed_pixels] = 0  # Set background at changed_pixels to 0
            boxes.append(obj.in_area)
            box_labels.append(element_label)
            class_names.append(class_name)

        class_labels = [float(class_index in box_labels) for class_index, _ in enumerate(self.get_class_names())]
        return {
            "image": im,
            "label": mask,
            "masks": masks,
            "boxes": boxes,
            "labels": box_labels,
            "class_labels": torch.Tensor(class_labels).float(),
            "obj_class_names": class_names,
            "meta": {
                "elements": scene_elements,
            },
        }

    def __getitem__(self, i: int) -> Dict[str, Any]:
        scene_elements: List[BaseObject] = self.case_list[i]
        return self._generate_case_for_objects(scene_elements)


def build_caption(case: dict[str, Any]):
    """Assumes that case["meta"]["elements"] exists"""
    if case["meta"]["elements"]:
        case["captions"] = [
            " and ".join([e.description for e in case["meta"]["elements"]]),
            "A scene with " + ", ".join([e.description for e in case["meta"]["elements"]]),
        ]
    else:
        case["captions"] = ["No objects in scene", "Empty scene"]
    return case


class IterShapeDataset(IterableDataset[Dict[str, Any]], ShapeDataset):
    def __iter__(self):
        if (worker_info := get_worker_info()) is not None:
            if worker_info.num_workers > 1:
                raise Exception(f"Iterable shape dataset cannot be used with more than 1 worker.")
        for objects in self.case_list:
            yield self._generate_case_for_objects(objects)


if __name__ == "__main__":
    from mmm.interactive import configs as cfs
    import streamlit as st
    from mmm.logging.st_ext import multi_cohort_explorer, stw

    @st.cache_resource()
    def ds(iterable: bool = False):
        ds_type = IterShapeDataset if iterable else ShapeDataset
        return ds_type(
            ShapeDataset.Config(
                N=1000,
                num_objects_per_scene_range=(1, 3),
                canvas=CanvasConfig(canvas_size=(224, 224)),
            )
        ), ShapeDataset(
            ShapeDataset.Config(
                N=500,
                num_objects_per_scene_range=(1, 1),
                canvas=CanvasConfig(canvas_size=(224, 224)),
            )
        )

    train_ds, val_ds = ds()

    def build_clf_cohort() -> TrainValCohort[ClassificationDataset]:
        classnames = ["No rectangle exists", "Rectangle exists"]

        def transform_for_label(case: Dict[str, Any]) -> Dict[str, Any]:
            case["class"] = int("rectangle" in case["obj_class_names"])
            return case

        base_transform = torch_transforms.Compose([ShapeDataset.cohort_transform(), transform_for_label])
        return TrainValCohort(
            cfs.TrainValCohortConfig(batch_size=8, num_workers=0),
            ClassificationDataset(train_ds, src_transform=base_transform, class_names=classnames),
            ClassificationDataset(val_ds, src_transform=base_transform, class_names=classnames),
        )

    def build_ae_cohort():
        from mmm.interactive import pipes, data
        import albumentations as A
        import torchvision.transforms as transforms

        base_t = [
            ShapeDataset.cohort_transform(),
            pipes.CopyKeysInDict({"image": "targetimage"}),
        ]
        augs = pipes.AlbWithBoxes(
            [
                A.RandomFog(p=0.33),
                A.Rotate(limit=45),
                A.RandomRain(p=0.33),
                A.RandomShadow(p=0.33),
                # A.RandomToneCurve(p=0.33),
                A.RandomSnow(p=0.33),
            ]
        )
        return data.TrainValCohort(
            cfs.TrainValCohortConfig(batch_size=8, num_workers=0),
            data.ImageTranslationDataset(train_ds, src_transform=transforms.Compose(base_t + [augs])),
            data.ImageTranslationDataset(val_ds, src_transform=transforms.Compose(base_t + [augs])),
        )

    def build_multilabelclf_cohort() -> TrainValCohort[MultilabelClassificationDataset]:
        base_transform = torch_transforms.Compose([ShapeDataset.cohort_transform()])
        return TrainValCohort(
            cfs.TrainValCohortConfig(batch_size=8, num_workers=0),
            MultilabelClassificationDataset(
                train_ds,
                src_transform=base_transform,
                class_names=train_ds.get_class_names(),
            ),
            MultilabelClassificationDataset(
                val_ds,
                src_transform=base_transform,
                class_names=val_ds.get_class_names(),
            ),
        )

    def build_semseg_cohort() -> TrainValCohort[SemSegDataset]:
        return TrainValCohort(
            cfs.TrainValCohortConfig(batch_size=8, num_workers=0),
            SemSegDataset(
                train_ds,
                src_transform=ShapeDataset.cohort_transform(),
                class_names=["background"] + train_ds.get_class_names(),
            ),
            SemSegDataset(
                val_ds,
                src_transform=ShapeDataset.cohort_transform(),
                class_names=["background"] + val_ds.get_class_names(),
            ),
        )

    def build_det_cohort() -> TrainValCohort[DetectionDataset]:
        return TrainValCohort(
            cfs.TrainValCohortConfig(batch_size=8, num_workers=0),
            DetectionDataset(
                train_ds,
                src_transform=ShapeDataset.cohort_transform(),
                class_names=train_ds.get_class_names(),
            ),
            DetectionDataset(
                val_ds,
                src_transform=ShapeDataset.cohort_transform(),
                class_names=val_ds.get_class_names(),
            ),
        )

    def build_caption_cohort() -> TrainValCohort[ImageCaptionDataset]:
        return TrainValCohort(
            TrainValCohort.Config(batch_size=8, num_workers=0),
            ImageCaptionDataset(
                train_ds,
                src_transform=torch_transforms.Compose([ShapeDataset.cohort_transform(), build_caption]),
            ),
            ImageCaptionDataset(
                val_ds,
                src_transform=torch_transforms.Compose([ShapeDataset.cohort_transform(), build_caption]),
            ),
        )

    def build_multilabelsemseg_cohort() -> TrainValCohort[MultiLabelSemSegDataset]:
        return TrainValCohort(
            cfs.TrainValCohortConfig(batch_size=8, num_workers=0),
            MultiLabelSemSegDataset(
                train_ds,
                src_transform=ShapeDataset.cohort_transform(),
                class_names=["background"] + train_ds.get_class_names(),
            ),
            MultiLabelSemSegDataset(
                val_ds,
                src_transform=ShapeDataset.cohort_transform(),
                class_names=["background"] + val_ds.get_class_names(),
            ),
        )

    multi_cohort_explorer(
        {
            "mlsemseg": build_multilabelsemseg_cohort,
            "caption": build_caption_cohort,
            "mlclf": build_multilabelclf_cohort,
            "clf": build_clf_cohort,
            "ae": build_ae_cohort,
            "shapesemseg": build_semseg_cohort,
            "shapedetection": build_det_cohort,
        }
    )
