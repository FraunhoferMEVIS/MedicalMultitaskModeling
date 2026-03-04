"""
Trainable simulated vision tasks with a low loading time and memory requirements
for automatic testing.

It comes with numerical and categorical tabular properties both visible in the image and invisible.
Visible: Color (categorical), Size (numerical)
Invisible: a configurable list of categorical and numerical properties without a direct representation in the image.
"""

from __future__ import annotations

import logging
import random
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Tuple, Type

import numpy as np
import torch
import torchvision.transforms as torch_transforms
from PIL.Image import Image
from PIL.Image import new as pil_new
from PIL.ImageDraw import Draw, ImageDraw
from pydantic import Field
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from mmm.BaseModel import BaseModel
from mmm.bucketizing import Bucket, BucketConfig
from mmm.data_loading.ClassificationDataset import ClassificationDataset
from mmm.data_loading.DetectionDataset import DetectionDataset
from mmm.data_loading.MTLDataset import flatten_list, mtl_batch_collate
from mmm.data_loading.MultilabelClassificationDataset import MultilabelClassificationDataset
from mmm.data_loading.MultiLabelSemSegDataset import MultiLabelSemSegDataset
from mmm.data_loading.SemSegDataset import SemSegDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.transforms import ApplyToKey, ApplyToList


def get_random_bounds(
    min_max_size=(0.1, 0.3), canvas_size=(224, 224), random_gen: random.Random = random.Random()
) -> Tuple[int, int, int, int]:
    width = random_gen.randint(int(canvas_size[0] * min_max_size[0]), int(canvas_size[0] * min_max_size[1]))
    height = random_gen.randint(int(canvas_size[1] * min_max_size[0]), int(canvas_size[1] * min_max_size[1]))
    x = random_gen.randint(0, canvas_size[0] - width)
    y = random_gen.randint(0, canvas_size[1] - height)
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

    box_around_changed_pixels = np.where(changed_pixels)
    if box_around_changed_pixels[0].size > 0 and box_around_changed_pixels[1].size > 0:
        x1, y1, x2, y2 = (
            min(box_around_changed_pixels[1]),
            min(box_around_changed_pixels[0]),
            max(box_around_changed_pixels[1]),
            max(box_around_changed_pixels[0]),
        )
        box = (x1, y1, x2, y2)
    else:
        logging.warning(f"No changed pixels found in {changed_pixels}")
        box = None

    return img, changed_pixels, box


class CanvasConfig(BaseModel):
    canvas_size: tuple[int, int] = (384, 384)
    bg_color: tuple[int, int, int] = (128, 128, 128)
    obj_size_range: tuple[float, float] = (0.1, 0.3)


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
    def sample_object_for_canvas(settings: CanvasConfig, random_gen: random.Random = random.Random()) -> BaseObject:
        raise NotImplementedError

    def draw_shape(self, canvas: ImageDraw) -> ImageDraw:
        raise NotImplementedError


class Ellipse(BaseObject):
    obj_type: Literal["ellipse"] = "ellipse"
    fillcolor: ShapeColor

    @staticmethod
    def sample_object_for_canvas(settings: CanvasConfig, random_gen: random.Random = random.Random()) -> BaseObject:
        fillcolor = random_gen.choice(list(ShapeColor))
        outer_area = get_random_bounds(
            min_max_size=settings.obj_size_range, canvas_size=settings.canvas_size, random_gen=random_gen
        )
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
    def sample_object_for_canvas(settings: CanvasConfig, random_gen: random.Random = random.Random()) -> BaseObject:
        fillcolor = random_gen.choice(list(ShapeColor))
        outer_area = get_random_bounds(
            min_max_size=settings.obj_size_range, canvas_size=settings.canvas_size, random_gen=random_gen
        )
        return Rectangle(
            description=f"A {fillcolor.name.lower()} rectangle",
            in_area=outer_area,
            fillcolor=fillcolor,
        )

    def draw_shape(self, canvas: ImageDraw) -> ImageDraw:
        canvas.rectangle(self.in_area, fill=ShapeColor.toRGB(self.fillcolor))
        return canvas


class LetterX(BaseObject):
    obj_type: Literal["letterx"] = "letterx"
    fillcolor: ShapeColor

    @staticmethod
    def _letter() -> str:
        return "X"

    @staticmethod
    def sample_object_for_canvas(settings: CanvasConfig, random_gen: random.Random = random.Random()) -> BaseObject:
        fillcolor = random_gen.choice(list(ShapeColor))
        outer_area = get_random_bounds(
            min_max_size=settings.obj_size_range, canvas_size=settings.canvas_size, random_gen=random_gen
        )
        return LetterX(
            description=f"A {fillcolor.name.lower()} letter {LetterX._letter()}",
            in_area=outer_area,
            fillcolor=fillcolor,
        )

    def draw_shape(self, canvas: ImageDraw) -> ImageDraw:
        font_size = int((self.in_area[3] - self.in_area[1]))
        canvas.text(
            self.in_area[:2],
            self._letter(),
            fill=ShapeColor.toRGB(self.fillcolor),
            stroke_width=font_size // 10,
            font_size=font_size,
        )
        return canvas


class LetterAt(LetterX):
    obj_type: Literal["letterat"] = "letterat"

    @staticmethod
    def _letter() -> str:
        return "@"


ObjectTypes = [Ellipse, Rectangle, LetterX, LetterAt]


class ShapeDataset(Dataset[dict[str, Any]]):
    """
    Dataset which uses PIL to draw simple shapes.

    `get_class_names()`'s index corresponds to the value in the segmentation masks.
    """

    class Config(BaseModel):
        N: int = 10
        canvas: CanvasConfig = CanvasConfig()
        num_objects_per_scene_range: tuple[int, int] = (3, 10)
        object_buckets: BucketConfig = BucketConfig(
            exactly_one_match=True,
            buckets=[
                Bucket(expression=".*ell.*", description="ellipse"),
                Bucket(expression=".*rect.*", description="rectangle"),
                Bucket(expression=".*letter.*", description="letter"),
            ],
        )
        invisible_categoricals: List[str] = ["first", "second"]
        invisible_numerical_range: tuple[float, float] = (-1.0, 1.0)
        generation_seed: int = 42

    def __init__(self, c: Config) -> None:
        self.random_gen = random.Random(c.generation_seed)  # Use a fixed seed for reproducibility across runs
        self.config = c

        # Each case stores several objects
        self.case_list = [self.generate_objects(self.config.num_objects_per_scene_range) for _ in range(self.config.N)]

    def get_class_names(self) -> List[str]:
        return self.config.object_buckets.get_class_names()

    def generate_objects(self, num_objects_range: Tuple[int, int]) -> List[BaseObject]:
        num_objects = self.random_gen.randint(num_objects_range[0], num_objects_range[1])
        object_types: List[Type[BaseObject]] = [self.random_gen.choice(ObjectTypes) for _ in range(num_objects)]

        objects = [o.sample_object_for_canvas(self.config.canvas, self.random_gen) for o in object_types]

        tabulars = {
            "invisible_categorical": self.random_gen.choice(self.config.invisible_categoricals),
            "invisible_numerical": self.random_gen.uniform(*self.config.invisible_numerical_range),
        }

        return objects, tabulars

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

    def _generate_case_for_objects(self, scene_elements: List[BaseObject], tabulars: dict) -> Dict[str, Any]:
        # add up to 5 intensity values to each channel in the bg_color
        bg_color_augmented = tuple(
            min(255, max(0, int(c + self.random_gen.randint(-5, 5)))) for c in self.config.canvas.bg_color
        )
        im: Image = pil_new("RGB", self.config.canvas.canvas_size[::-1], bg_color_augmented)
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

            im, changed_pixels, box = track_difference(im, obj.draw_shape)
            mask[changed_pixels] = element_label + 1
            masks[element_label + 1, changed_pixels] = 1  # Set object at changed_pixels to 1
            masks[0, changed_pixels] = 0  # Set background at changed_pixels to 0
            if box is not None:
                boxes.append(list(box))
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
            "tabulars": tabulars,
        }

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self._generate_case_for_objects(*self.case_list[i])


class ShapeDatasetVideo(ShapeDataset):
    """
    - Each case is a list of frames
    - The label of interest is the most common object in the group
    - invisible_categorical can be label of interest or another object
      - makes the instance class label depend on the group
    """

    class Config(ShapeDataset.Config):
        group_size_range: tuple[int, int] = (2, 5)
        num_objects_per_scene_range: tuple[int, int] = (0, 5)
        invisible_categoricals: list[str] = ["labelofinterest", "ellipse", "rectangle", "letter"]
        depend_on_position: bool = False  # If True, objects from the previous frame are copied to the next frame
        depend_on_tabular: bool = False  # If True, copies the invisible_categoricals object from all other frames
        depend_on_group: bool = True  # Makes label of most common object in group switch, and copies rectangles

    def __init__(self, c: ShapeDataset.Config) -> None:
        self.config: ShapeDatasetVideo.Config = c
        self.random_gen = random.Random(c.generation_seed)  # Use a fixed seed for reproducibility across runs
        self.case_list = [self.generate_supercase() for _ in range(c.N)]

    def generate_supercase(self) -> dict[str, Any]:
        group_size = self.random_gen.randint(self.config.group_size_range[0], self.config.group_size_range[1])
        group = []
        for frame_index in range(group_size):
            objects, tabulars = self.generate_objects(self.config.num_objects_per_scene_range)

            # Randomly copy an object from the previous frame
            if self.config.depend_on_position and frame_index > 0 and (previous_objects := group[frame_index - 1][0]):
                objects.append(self.random_gen.choice(previous_objects))

            group.append((objects, tabulars))

        return group

    def get_context_class_name(self):
        return ["no", "yes"]

    def get_class_names(self):
        # Add additional object which is defined by the context
        return super().get_class_names() + ["groupobject"]

    def build_allinstance_objectmask(self, instances, object_class: int = 2):
        # Build a mask which should apply to all instances in the group
        all_obj_mask = np.zeros_like(instances[0]["label"])
        all_obj_boxes = []
        all_obj_labels = []
        for r in instances:
            all_obj_mask[r["label"] == object_class + 1] = object_class + 1
            rect_box_indices = [i for i, x in enumerate(r["labels"]) if x == object_class]
            all_obj_boxes += [r["boxes"][i] for i in rect_box_indices]
            all_obj_labels += [r["labels"][i] for i in rect_box_indices]
        return all_obj_mask, all_obj_boxes, all_obj_labels

    def __getitem__(self, i: int) -> Dict[str, Any]:
        instances = []

        # Generate instances for each frame in the group
        for frame_index, scene_elements in enumerate(self.case_list[i]):
            newitem = self._generate_case_for_objects(*scene_elements)
            newitem["meta"]["group_id"] = f"Group{i}"
            newitem["meta"]["context"] = (frame_index, newitem["tabulars"]["invisible_categorical"])
            instances.append(newitem)

        # For all instances, copy something from the previous frame if depend_on_position is True
        if self.config.depend_on_position:
            for frame_index in range(1, len(instances)):
                instances[frame_index]["label"][instances[frame_index - 1]["label"] == 1] = 1
                # Do the same for boxes
                previous_circles = [i for i, x in enumerate(instances[frame_index - 1]["labels"]) if x == 0]
                instances[frame_index]["boxes"] += [instances[frame_index - 1]["boxes"][i] for i in previous_circles]
                instances[frame_index]["labels"] += [0] * len(previous_circles)

        if self.config.depend_on_group:
            # Replace the label of the most common object in the group with a groupobject class
            if all_labels := [object_label for instance in instances for object_label in instance["labels"]]:
                label_of_interest = max(set(all_labels), key=all_labels.count)  # Get the label which is most common
                for instance in instances:
                    instance["meta"]["label_of_interest_name"] = self.get_class_names()[label_of_interest]
                    instance["class"] = int(len([x for x in instance["labels"] if x == label_of_interest]) > 0)
                    if (label_of_interest + 1) in instance["label"]:
                        instance["label"][instance["label"] == (label_of_interest + 1)] = len(self.get_class_names())
                    if label_of_interest in instance["labels"]:
                        instance["labels"] = [
                            len(self.get_class_names()) - 1 if object_label == label_of_interest else object_label
                            for object_label in instance["labels"]
                        ]
            else:
                for instance in instances:
                    instance["class"] = 0

            # Add all rectangle labels from the unified mask
            all_obj_mask, all_obj_boxes, all_obj_labels = self.build_allinstance_objectmask(instances, 1)
            for instance in instances:
                instance["label"][(all_obj_mask == 2) & (instance["label"] == 0)] = 2
                instance["boxes"] = [
                    box for i, box in enumerate(instance["boxes"]) if instance["labels"][i] != 1
                ] + all_obj_boxes
                instance["labels"] = [label for label in instance["labels"] if label != 1] + all_obj_labels

        if self.config.depend_on_tabular:
            raise NotImplementedError()

        return instances


class GroupShapeDataset(ShapeDataset):
    """
    Toy dataset for survival analysis with complex data which consists of multiple image instances.

    Each time to event starts at 0, and is at least 0.
    Each object then pushes the time to event forward or back by an amount depending of the object's type.
    """

    class Config(ShapeDataset.Config):
        group_size_range: tuple[int, int] = (10, 13)
        num_objects_per_scene_range: tuple[int, int] = (1, 3)

        prob_biased: float = Field(0.1, description="These tiles will oversample some objects.")
        prob_useless: float = Field(0.05, description="These tiles will have no objects.")

        min_time_to_event: float = Field(0.1, description="Minimum time to event for a case.")
        censor_prob: float = Field(0.5, description="Probability of censoring the time to event.")

    def __init__(self, c: Config) -> None:
        self.config: GroupShapeDataset.Config = c
        self.case_list = [self.generate_supercase() for _ in range(self.config.N)]

    def compute_time_to_event(self, objects: list[BaseObject]) -> float:
        """
        - Ellipses will push the time to event forward by 0.2
        - Rectangles with color ShapeColor.TurkeyRed will push the time to event backward by 2.0
        - Other rectangles will push the time to event forward by 0.05
        """
        time_to_event = 0.0
        for obj in objects:
            if isinstance(obj, Ellipse):
                time_to_event += 0.4
            elif isinstance(obj, Rectangle):
                if obj.fillcolor is ShapeColor.TurkeyRed:
                    time_to_event -= 1.0
                else:
                    time_to_event += 0.2
            else:
                raise ValueError(f"Unknown object type {obj.obj_type}")
        return max(self.config.min_time_to_event, time_to_event)

    def generate_supercase(self) -> dict[str, Any]:
        group_size = random.randint(self.config.group_size_range[0], self.config.group_size_range[1])
        objects = self.generate_objects(
            (
                self.config.num_objects_per_scene_range[0] * group_size,
                self.config.num_objects_per_scene_range[1] * group_size,
            ),
        )

        # Decide on the type of object and compute the survival target
        time_to_event = self.compute_time_to_event(objects)

        # Distribute the objects on the image instances
        subcases = []
        for subcase_index in range(group_size):
            if subcase_index == group_size - 1:
                scene_objects = objects
            elif random.random() < self.config.prob_useless:
                scene_objects = []
            elif random.random() < self.config.prob_biased:
                # Only add turkey red rectangles
                red_rects = [o for o in objects if isinstance(o, Rectangle) and o.fillcolor is ShapeColor.TurkeyRed]
                if len(red_rects) == 0:
                    scene_objects = []
                else:
                    scene_objects = random.sample(
                        red_rects,
                        random.randint(
                            self.config.num_objects_per_scene_range[0],
                            min(len(red_rects), self.config.num_objects_per_scene_range[1]),
                        ),
                    )
            else:
                # Add remaining objects such that the distribution of remaining objects is balanced
                windows_left = group_size - subcase_index
                good_number_of_objects = len(objects) // windows_left
                scene_objects = random.sample(objects, good_number_of_objects)
            objects = [o for o in objects if o not in scene_objects]
            subcases.append(self._generate_case_for_objects(scene_objects))

        censored = random.random() < self.config.censor_prob
        full_time_to_event = time_to_event
        if censored:
            # Random number between 0 and the true time to event
            time_to_event = random.random() * time_to_event

        return {
            "time_to_event": time_to_event,
            "event": not censored,
            "num_objects": len(objects),
            "group_size": group_size,
            "subcases": subcases,
            "uncensored_time_to_event": full_time_to_event,
        }

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.case_list[i]

    def get_labeling_xml(self) -> str:
        surv_config = '<TextArea name="survival" toName="image" editable="true" />'
        reg_config = '<Number name="survivaltime" toName="image" min="0" />'
        return f"""
<View>
    <Image name="image" value="$image" zoom="true"/>
    {surv_config}
    {reg_config}
</View>"""

    def to_lstask(self, group: dict[str, Any]):
        from m3_sdk.utils import rgbnumpy_to_base64

        from mmm.api.models import Annotation, MSubject

        anno_results = [
            {
                "type": "textarea",
                "value": {"text": [f"{group['time_to_event']}|{'EVENT' if group['event'] else 'CENSORED'}"]},
                "to_name": "image",
                "from_name": "survival",
            },
            {
                "type": "number",
                "value": {"number": group["time_to_event"]},
                "to_name": "image",
                "from_name": "survivaltime",
            },
        ]

        if self.config.group_size_range[1] == 1:
            assert len(group["subcases"]) == 1
            # If there is only one subcase, directly use the image
            imgs = rgbnumpy_to_base64(np.array(group["subcases"][0]["image"]))
        else:
            imgs = [rgbnumpy_to_base64(np.array(subcase["image"])) for subcase in group["subcases"]]

        return MSubject(
            data={
                # directly embed the image into the task by converting it to base64
                "image": imgs,
            },
            annotations=[{"result": anno_results}],
            meta={"uncensored_time_to_event": group["uncensored_time_to_event"]},
        )


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
    from mmm.logging.st_ext import multi_cohort_explorer, st, stw

    @st.cache_resource()
    def ds(iterable: bool = False):
        ds_type = IterShapeDataset if iterable else ShapeDataset
        return ds_type(
            ShapeDataset.Config(
                N=1000,
                num_objects_per_scene_range=(1, 3),
                canvas=CanvasConfig(canvas_size=(512, 512)),
            )
        ), ShapeDataset(
            ShapeDataset.Config(
                N=500,
                num_objects_per_scene_range=(1, 1),
                canvas=CanvasConfig(canvas_size=(512, 512)),
            )
        )

    def build_clf_cohort() -> TrainValCohort[ClassificationDataset]:
        train_ds, val_ds = ds()
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

    def build_multilabelclf_cohort() -> TrainValCohort[MultilabelClassificationDataset]:
        train_ds, val_ds = ds()
        base_transform = torch_transforms.Compose([ShapeDataset.cohort_transform()])
        return TrainValCohort(
            TrainValCohort.Config(batch_size=8, num_workers=0),
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
        train_ds, val_ds = ds()
        return TrainValCohort(
            TrainValCohort.Config(batch_size=8, num_workers=0),
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
        train_ds, val_ds = ds()
        return TrainValCohort(
            TrainValCohort.Config(batch_size=8, num_workers=0),
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

    def build_multilabelsemseg_cohort() -> TrainValCohort[MultiLabelSemSegDataset]:
        train_ds, val_ds = ds()
        return TrainValCohort(
            TrainValCohort.Config(batch_size=8, num_workers=0),
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

    @st.cache_resource()
    def build_semseg3d_cohort() -> TrainValCohort[SemSegDataset]:
        train3d_ds, val3d_ds = ShapeDatasetVideo(ShapeDatasetVideo.Config()), ShapeDatasetVideo(
            ShapeDatasetVideo.Config()
        )

        return TrainValCohort(
            TrainValCohort.Config(batch_size=1, num_workers=0),
            SemSegDataset(
                train3d_ds,
                src_transform=ApplyToList(ShapeDataset.cohort_transform()),
                class_names=["background"] + train3d_ds.get_class_names(),
                collate_fn=mtl_batch_collate,
            ),
            SemSegDataset(
                val3d_ds,
                src_transform=ApplyToList(ShapeDataset.cohort_transform()),
                class_names=["background"] + val3d_ds.get_class_names(),
                collate_fn=mtl_batch_collate,
            ),
        )

    @st.cache_resource()
    def build_detection3d_cohort() -> TrainValCohort[DetectionDataset]:
        train3d_ds, val3d_ds = ShapeDatasetVideo(ShapeDatasetVideo.Config()), ShapeDatasetVideo(
            ShapeDatasetVideo.Config()
        )

        return TrainValCohort(
            TrainValCohort.Config(batch_size=1, num_workers=0),
            DetectionDataset(
                train3d_ds,
                src_transform=ApplyToList(ShapeDataset.cohort_transform()),
                class_names=train3d_ds.get_class_names(),
                collate_fn=flatten_list,
            ),
            DetectionDataset(
                val3d_ds,
                src_transform=ApplyToList(ShapeDataset.cohort_transform()),
                class_names=val3d_ds.get_class_names(),
                collate_fn=flatten_list,
            ),
        )

    @st.cache_resource()
    def build_clf_video_cohort() -> TrainValCohort[ClassificationDataset]:
        train3d_ds = ShapeDatasetVideo(ShapeDatasetVideo.Config())
        val3d_ds = ShapeDatasetVideo(ShapeDatasetVideo.Config())

        return TrainValCohort(
            TrainValCohort.Config(batch_size=1, num_workers=0),
            ClassificationDataset(
                train3d_ds,
                src_transform=ApplyToList(ShapeDataset.cohort_transform()),
                class_names=train3d_ds.get_context_class_name(),
            ),
            ClassificationDataset(
                val3d_ds,
                src_transform=ApplyToList(ShapeDataset.cohort_transform()),
                class_names=val3d_ds.get_context_class_name(),
            ),
        )

    multi_cohort_explorer(
        {
            "shapesemseg": build_semseg_cohort,
            "shapesemseg3d": build_semseg3d_cohort,
            "mlsemseg": build_multilabelsemseg_cohort,
            # "caption": build_caption_cohort,
            "mlclf": build_multilabelclf_cohort,
            "clf": build_clf_cohort,
            "shapedetection": build_det_cohort,
            "shapedetection3d": build_detection3d_cohort,
            "clfvideo": build_clf_video_cohort,
        }
    )

    # from mmm.api.st_annotate import st_annotate
    # from mmm.api.evaluation import TaskDefinition

    # with st.form(key="target_task"):
    #     target_folder_path = st.text_input(label="Output folder", value="./shape_target_task")
    #     num_cases = st.number_input(label="Number of cases", value=100)
    #     submitted = st.form_submit_button("Write target task to disk")

    # if submitted:
    #     data_folder = Path(target_folder_path)
    #     data_folder.mkdir(parents=True, exist_ok=True)

    #     td = TaskDefinition(categories=["shapeworld", "unittesting"])
    #     (data_folder / "task-definition.json").write_text(td.model_dump_json(indent=2))

    #     ds = GroupShapeDataset(
    #         GroupShapeDataset.Config(
    #             N=num_cases, group_size_range=(3, 10), canvas=CanvasConfig(canvas_size=(128, 128)), censor_prob=0.3
    #         )
    #     )
    #     (data_folder / td.labeling_xml.upath()).write_text(ds.get_labeling_xml())
    #     st_annotate(task=ds.to_lstask(ds[0]), labeling_config=ds.get_labeling_xml())
    #     (data_folder / td.data_json_files[0].upath()).write_text(
    #         json.dumps([ds.to_lstask(ds[i]).model_dump_labelstudio() for i in range(len(ds))])
    #     )

    # ds = Semseg3DDataset(Semseg3DDataset.Config())
    # stw(ds)
    # stw(ds[0])
    # stw(build_semseg3d_cohort())
