from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from m3_sdk.geojson import GeoAnno, annotations_from_mask, create_featurecollection
from rasterio.features import rasterize
from shapely.affinity import scale, translate
from shapely.geometry import Polygon, shape
from tiffslide import TiffSlide


def extract_detection_labels(
    subcase_dict: dict[str, Any],
    annos_in_window: list[GeoAnno],
    buckets: list[tuple[str, int]],
    patch_size: tuple[int, int],
):
    anno_boxes = [anno.get_enclosing_region() for anno in annos_in_window]
    anno_detection_boxes_level = [
        (
            max(0, int(anno_x)),  # / downsample_fac) - x_level),
            max(0, int(anno_y)),  # / downsample_fac) - y_level),
            min(patch_size[0] - 1, int((anno_x + anno_width))),  # / downsample_fac) - x_level),
            min(patch_size[1] - 1, int((anno_y + anno_height))),  # / downsample_fac) - y_level),
        )
        for ((anno_x, anno_y), (anno_width, anno_height)) in anno_boxes
    ]
    subcase_dict["boxes"] = torch.Tensor(anno_detection_boxes_level).float()
    subcase_dict["labels"] = torch.Tensor([x[1] for x in buckets]).long()
    return subcase_dict


def rasterize_annotations(
    window_height_level,
    window_width_level,
    annotations: list[GeoAnno],
    anno_labels: list[int],
    unlabeled_value=-1,
):
    arr = np.zeros((window_height_level, window_width_level), dtype=np.int64)
    arr.fill(unlabeled_value)
    for anno, class_key in zip(annotations, anno_labels):
        rasterize([anno.shape], out=arr, fill=unlabeled_value, default_value=(class_key))
    return arr


def rasterize_multilabel_annotations(
    window_width_level,
    window_height_level,
    annotations,
    anno_labels,
    num_classes: int,
    unlabeled_value=-1,
):
    arr = np.zeros((num_classes, window_width_level, window_height_level), dtype=np.int64)
    arr.fill(unlabeled_value)
    for anno, class_key in zip(annotations, anno_labels):
        rasterize([anno.shape], out=arr[class_key], fill=unlabeled_value, default_value=1)
    return arr


def move_anno_to_origin(anno: GeoAnno, l0_window: Polygon, downsample_fac: float):
    """
    Given an annotation detected in some window specified by its full-sized (level 0) Polygon and a downsample factor,
    returns the annotation in original l0 coordinates.
    """
    min_x, min_y, max_x, max_y = l0_window.bounds
    funcs = [
        lambda o: scale(o, xfact=downsample_fac, yfact=downsample_fac, origin=(0, 0)),  # type: ignore
        lambda o: translate(o, xoff=min_x, yoff=min_y),
    ]
    anno.shape = transforms.Compose(funcs)(anno.shape)
    return anno


def move_anno_to_window(anno: GeoAnno, l0_window: Polygon, l0_x, l0_y, downsample_fac):
    funcs = [
        lambda o: translate(o, xoff=-1 * l0_x, yoff=-1 * l0_y),
        lambda o: scale(o, xfact=1 / downsample_fac, yfact=1 / downsample_fac, origin=(0, 0)),  # type: ignore
    ]
    anno.shape = anno.shape.intersection(l0_window)
    anno.shape = transforms.Compose(funcs)(anno.shape)
    return anno


def create_geojson_from_tissuemask(slide: TiffSlide, mask_slidepath: Path, coarse=False) -> dict:
    mask_array = np.array(TiffSlide(mask_slidepath).get_thumbnail((768, 768)))
    downsample_fac = slide.level_dimensions[0][0] / mask_array.shape[1]

    assert len(np.unique(mask_array)) > 1
    foreground = annotations_from_mask(
        (mask_array > 0).astype(np.uint8),
        for_values={1: "foreground"},
        downsample_fac=downsample_fac,
        coarse=coarse,
    )

    return create_featurecollection(foreground)
