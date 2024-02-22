from typing import Any
import numpy as np
import torch
import cv2
from shapely.affinity import scale, translate
from shapely.geometry import Polygon, shape
from rasterio.features import rasterize
from rasterio import features
import torchvision.transforms as transforms

from .GeoAnno import GeoAnno


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
    window_width_level,
    window_height_level,
    annotations: list[GeoAnno],
    anno_labels: list[int],
    unlabeled_value=-1,
):
    arr = np.zeros((window_width_level, window_height_level), dtype=np.int64)
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


def create_featurecollection(annos: list[GeoAnno]) -> dict:
    res = {
        "type": "FeatureCollection",
        "features": [anno.to_geojson() for anno in annos],
    }
    return res


def shapes_from_binary_mask(mask: np.ndarray, downsample_fac: float, coarse: bool, min_area: float = 0.0):
    """
    Returns a list of shapely shapes from a binary mask by using connected components.

    min_area always refers to the area in level 0 (computed by your downsample_fac).
    """
    if coarse:
        fine_areas = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)[2][:, cv2.CC_STAT_AREA]
        mask = features.sieve(mask, size=max(1, fine_areas.mean()))
    dicts = list(features.shapes(mask))
    shapes = [(shape(x[0]), x[1]) for x in dicts]
    shapes = [
        x for x, val in shapes if x.bounds[2] - x.bounds[0] > 1.0 and x.bounds[3] - x.bounds[1] > 1.0 and val > 0.0
    ]
    if min_area:
        shapes = filter(lambda x: (x.area * (downsample_fac**2)) > min_area, shapes)
    return [scale(x, xfact=downsample_fac, yfact=downsample_fac, origin=(0, 0)) for x in shapes]  # type:ignore


def annotations_from_mask(
    mask: np.ndarray,
    for_values: dict[int, str],
    downsample_fac: float,
    coarse: bool = False,
    min_area: float = 0.0,
) -> list[GeoAnno]:
    """
    min_area refers to the area in level 0 (computed by your downsample_fac).
    """
    res: list[GeoAnno] = []
    for val, class_name in for_values.items():
        shapes_for_val = shapes_from_binary_mask(
            (mask == val).astype(np.uint8), downsample_fac, coarse, min_area=min_area
        )
        annos_for_val = [GeoAnno.from_shapely(shape) for shape in shapes_for_val]
        for anno in annos_for_val:
            anno.set_class_name(class_name)
        res.extend(annos_for_val)
    return res
