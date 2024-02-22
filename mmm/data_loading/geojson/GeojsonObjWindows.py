from typing import Literal, Any
import re
import logging
import random
from pydantic import Field
from copy import deepcopy
from shapely.geometry import Polygon
import numpy as np
from tiffslide import TiffSlide
import torch
import torchvision.transforms.functional as F

from mmm.BaseModel import BaseModel
from mmm.bucketizing import BucketConfig, Bucket

from .GeoAnno import GeoAnno
from .utils import (
    move_anno_to_window,
    rasterize_annotations,
    rasterize_multilabel_annotations,
    extract_detection_labels,
)

ObjectLabelType = Literal["det", "seg", "mlsemseg"]


class GeojsonObjWindows(BaseModel):
    """
    Each WSI is treated as a supercase with windows as subcases.
    This class offers a parameterized function for extracting windows per WSI.

    Enables segmentation, detection and multilabel classification.
    """

    minimum_pixels_level: int = Field(
        default=15 * 15,
        description="An annotation must represent at least this many pixels in the level to be included.",
    )
    filter_larger_than_fraction: float = Field(
        default=0.95,
        description="If an annotation is larger than this fraction of the window, it will be excluded.",
    )
    output_patch_size: tuple[int, int] = 1024, 1024
    minimum_visible_fraction: float = 0.2
    buckets: BucketConfig = BucketConfig(buckets=[Bucket(expression=f".*", description="all")])
    best_area_per_patch: float = 0.5
    target_label: list[ObjectLabelType] = ["det"]
    prefill_semsegmask_value: int = -1

    stuff_matcher: str | None = Field(
        default=None,
        description="Stuff is rasterized with in the back and not filtered by visibility criteria.",
    )

    def _pick_good_level(self, for_anno: GeoAnno, downsample_factors: list[float]) -> int | None:
        wind_wh = {
            level: (
                self.output_patch_size[0] * factor,
                self.output_patch_size[1] * factor,
            )
            for level, factor in enumerate(downsample_factors)
        }
        _, l0_anno_size = for_anno.get_enclosing_region()
        areas_per_level = {
            level: (
                for_anno.shape.area / (downsample_factor**2),
                l0_anno_size[0] / downsample_factor,
                l0_anno_size[1] / downsample_factor,
            )
            for level, downsample_factor in enumerate(downsample_factors)
        }
        # Collect the probability for each level to select it as the best level
        good_levels = {
            level: 1 - np.abs(self.best_area_per_patch - ((w * h) / (wind_wh[level][0] * wind_wh[level][1])))
            for level, (ar, w, h) in areas_per_level.items()
            if (
                (ar > self.minimum_pixels_level)
                and (ar / (wind_wh[level][0] * wind_wh[level][1]) < self.filter_larger_than_fraction)
                and (w < self.output_patch_size[0])
                and (h < self.output_patch_size[1])
            )
        }
        if not good_levels:
            # If there is no really good level for this annotation, skip the window
            logging.debug(f"Found no good window for {for_anno}")
            return None
        # Pick one of the levels that are suitable for this annotation:
        window_level = random.choices(list(good_levels.keys()), weights=list(good_levels.values()), k=1)[0]
        return window_level

    def select_good_window(
        self, for_anno: GeoAnno, downsample_fac: float, wsi_rows: int, wsi_cols: int
    ) -> tuple[Polygon, int, int, Polygon] | None:
        l0_anno_location, l0_anno_size = for_anno.get_enclosing_region()
        l0_window_rows, l0_window_cols = (
            self.output_patch_size[0] * downsample_fac,
            self.output_patch_size[1] * downsample_fac,
        )
        max_x = min(wsi_rows - l0_window_rows, l0_anno_location[0])
        min_x = max(0, l0_anno_location[0] - (l0_window_rows - l0_anno_size[0]))
        max_y = min(wsi_cols - l0_window_cols, l0_anno_location[1])
        min_y = max(0, l0_anno_location[1] - (l0_window_cols - l0_anno_size[1]))
        if min_x > max_x or min_y > max_y:
            logging.debug(f"The annotation {for_anno} is too large for the selected level")
            return None
        l0_x, l0_y = random.randint(int(min_x), int(max_x)), random.randint(int(min_y), int(max_y))
        l0_window = Polygon(
            [
                (l0_x, l0_y),
                (l0_x + l0_window_rows, l0_y),
                (l0_x + l0_window_rows, l0_y + l0_window_cols),
                (l0_x, l0_y + l0_window_cols),
            ]
        )

        l_x, l_y = int(l0_x / downsample_fac), int(l0_y / downsample_fac)
        l_window = Polygon(
            [
                (l_x, l_y),
                (l_x + self.output_patch_size[0], l_y),
                (l_x + self.output_patch_size[0], l_y + self.output_patch_size[1]),
                (l_x, l_y + self.output_patch_size[1]),
            ]
        )
        return l0_window, l0_x, l0_y, l_window

    def transform_wsi_to_windows(self, supercase: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Supercases need to have at least the keys:

        - wsi: TiffSlide
        - annos: List[GeoAnno]

        and should have a meta key containing a custom dictionary.

        This method yields windows with the annotations in them.
        """
        res = []
        for window_dict in self._iterate_windows(supercase):
            slide, l0_x, l0_y, window_level, annos_in_window, buckets, anno_labels = (
                window_dict["slide"],
                window_dict["l0_x"],
                window_dict["l0_y"],
                window_dict["window_level"],
                window_dict["annos_in_window"],
                window_dict["buckets"],
                window_dict["anno_labels"],
            )
            try:
                img_region = slide.read_region(
                    location=(l0_x, l0_y),
                    level=window_level,
                    size=self.output_patch_size,
                )
            except Exception as e:
                logging.error(
                    f"Region read error {e=}\n"
                    + "for {l0_x}, {l0_y}, {window_level}, {self.output_patch_size} in {window_dict['slide']=}"
                )
                continue
            subcase_dict: dict[str, Any] = {
                "image": F.to_tensor(img_region),
                "meta": {
                    "level": window_level,
                    "window_location": (l0_x, l0_y),
                    "classnames": anno_labels,
                },
            }

            for label_type in self.target_label:
                self._extract_labels(subcase_dict, label_type, annos_in_window, buckets)

            if "meta" in supercase:
                subcase_dict["meta"]["supercase"] = supercase["meta"]
            res.append(subcase_dict)

        return res

    def _is_window_perfect_for_anno(self, l0_window: Polygon, l0_anno: GeoAnno, downsample_fac: float) -> bool:
        # Not too big or only partly visible:
        by_overlap = l0_anno.overlaps_with(l0_window, 0.99)
        # Not too small:
        by_pixel_num = (l0_anno.shape.area / (downsample_fac**2)) > self.minimum_pixels_level
        return by_overlap and by_pixel_num

    def is_anno_useful_for_window(self, l0_window: Polygon, l0_anno: GeoAnno, downsample_fac: float) -> bool:
        return l0_anno.overlaps_with(l0_window, self.minimum_visible_fraction)

    def _is_stuff(self, anno: GeoAnno) -> bool:
        # Match using regex
        if self.stuff_matcher is None:
            return False

        return re.match(self.stuff_matcher, anno.get_class_name()) is not None

    def _iterate_windows(self, supercase: dict[str, Any]) -> list[dict[str, Any]]:
        slide: TiffSlide = TiffSlide(supercase["wsi"])

        wsi_rows, wsi_cols = slide.level_dimensions[0][0], slide.level_dimensions[0][1]
        # This buffer will be used to create windows
        all_annos: list[GeoAnno] = [anno for anno in supercase["annos"]]
        # Only use index in this list
        annos_left: list[int] = [i for i, _ in enumerate(all_annos)]
        random.shuffle(annos_left)

        if self.stuff_matcher is not None:
            # Put stuff in the back of the list to give objects the chance for a good window first
            stuff_annos = filter(lambda i: self._is_stuff(all_annos[i]), annos_left)
            thing_annos = filter(lambda i: not self._is_stuff(all_annos[i]), annos_left)
            annos_left = list(thing_annos) + list(stuff_annos)

        while annos_left:
            # The perfect window for this annotation is selected
            main_annotation_index = annos_left.pop(0)
            main_annotation: GeoAnno = all_annos[main_annotation_index]

            # Find the level where the annotation makes up a maximum of 50% of the area
            window_level = self._pick_good_level(main_annotation, slide.level_downsamples)
            if window_level is None:
                continue
            downsample_fac = slide.level_downsamples[window_level]

            l0_window, l0_x, l0_y, l_window = self.select_good_window(
                main_annotation, downsample_fac, wsi_rows, wsi_cols
            )
            if l0_window is None:
                continue

            other_annos = []
            for all_anno_index, l0_anno in enumerate(all_annos):
                if all_anno_index != main_annotation_index:
                    # Determine if anno is well represented, partially represented or not represented in this window
                    if self._is_window_perfect_for_anno(l0_window, l0_anno, downsample_fac):
                        other_annos.append(deepcopy(l0_anno))
                        if all_anno_index in annos_left:
                            annos_left.remove(all_anno_index)
                    elif self.is_anno_useful_for_window(l0_window, l0_anno, downsample_fac):
                        # This annotation is not yet well represented, it should get another chance in its own window
                        # In consequence, it is not removed from the list of annotations left
                        other_annos.append(deepcopy(l0_anno))
                    elif self._is_stuff(l0_anno) and l0_anno.shape.intersection(l0_window).area > 0:
                        other_annos.append(deepcopy(l0_anno))

            other_annos.append(deepcopy(main_annotation))

            annos_in_window: list[GeoAnno] = list(
                map(
                    lambda anno: move_anno_to_window(anno, l0_window, l0_x, l0_y, downsample_fac),
                    other_annos,
                )
            )
            # Filter those that are too small or too big in this window
            annos_in_window = [
                anno
                for anno in annos_in_window
                if self._is_stuff(anno)
                or (
                    (anno.shape.area > self.minimum_pixels_level)
                    and (anno.shape.area / l_window.area < self.filter_larger_than_fraction)
                )
            ]
            if not annos_in_window:
                # This should never happen because the window is extracted to fit at least one annotation
                logging.warning("Somehow I got a window without useful annotations")

            # Sort the stuff annotations into the beginning of the list to ensure they do not overwrite things
            annos_in_window.sort(key=lambda anno: not self._is_stuff(anno))

            anno_labels = [anno.get_class_name() for anno in annos_in_window]
            buckets: list[tuple[str, int]] = [self.buckets.get_bucket_name(l.lower()) for l in anno_labels]
            yield dict(
                annos_in_window=annos_in_window,
                buckets=buckets,
                l0_window=l0_window,
                l_window=l_window,
                slide=slide,
                l0_x=l0_x,
                l0_y=l0_y,
                window_level=window_level,
                anno_labels=anno_labels,
            )

    def _extract_labels(
        self,
        subcase_dict: dict[str, Any],
        t: ObjectLabelType,
        annos_in_window: list[GeoAnno],
        buckets: list[tuple[str, int]],
    ) -> None:
        if t == "det":
            subcase_dict = extract_detection_labels(subcase_dict, annos_in_window, buckets, self.output_patch_size)
        elif t == "seg":
            subcase_dict["label"] = torch.from_numpy(
                rasterize_annotations(
                    self.output_patch_size[0],
                    self.output_patch_size[1],
                    annos_in_window,
                    torch.Tensor([x[1] for x in buckets]).long(),
                    unlabeled_value=self.prefill_semsegmask_value,
                )
            )
        elif t == "mlsemseg":
            subcase_dict["masks"] = torch.from_numpy(
                rasterize_multilabel_annotations(
                    self.output_patch_size[0],
                    self.output_patch_size[1],
                    annos_in_window,
                    torch.Tensor([x[1] for x in buckets]).long(),
                    len(self.buckets.buckets),
                    unlabeled_value=self.prefill_semsegmask_value,
                )
            )
        else:
            raise NotImplementedError
