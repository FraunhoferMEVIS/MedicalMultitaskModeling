import pytest
from typing import Callable, Tuple

from mmm.data_loading.geojson import GeoAnno, AnnotationType, GeojsonRegionWindows
from mmm.data_loading.geojson import NoUsefulWindowException

from mmm.data_loading.geojson.utils import move_anno_to_origin
from mmm.data_loading.geojson.AnnoFilterConfig import AnnoFilterConfig


def test_rectangle_builder():
    anno = GeoAnno.rectangle_builder((0, 0), (10, 10))
    assert anno.shape.area == 100.0
    assert anno.get_annotation_type() is AnnotationType.DetectionBox
    assert anno.get_class_name() == "defaultclass"


def test_region_filter():
    filter_region = GeoAnno.rectangle_builder((0, 0), (10, 10))
    smaller_anno = GeoAnno.rectangle_builder((1, 1), (9, 9))
    only_intersecting_anno = GeoAnno.rectangle_builder((5, 5), (15, 15))
    not_intersecting_anno = GeoAnno.rectangle_builder((11, 11), (20, 20))

    filter_config = AnnoFilterConfig()
    assert filter_config.apply(smaller_anno, use_regions=[filter_region])
    assert not filter_config.apply(not_intersecting_anno, use_regions=[filter_region])
    assert filter_config.apply(only_intersecting_anno, use_regions=[filter_region])


def test_moving_to_origin():
    anno = GeoAnno.rectangle_builder((10, 10), (20, 20))
    origin_anno = GeoAnno.rectangle_builder((60, 60), (70, 70))
    window = GeoAnno.rectangle_builder((50, 50), (100, 100))

    moved_anno = move_anno_to_origin(anno, window.shape, 1.0)
    assert (origin_anno.shape.intersection(moved_anno.shape)).area == origin_anno.shape.area


def test_deterministic_geojson_region_windows_singlelevel():
    windower = GeojsonRegionWindows(
        min_area=1.0,
        patch_size=(100, 100),
        coordinate_augmentation="none",
        windowsize_augmentation="none",
        stepsize=1,
    )
    windows = [x for x in windower.iter_valid_windows(GeoAnno.rectangle_builder((0, 0), (1000, 1000)), {0: 1.0})]

    # With deterministic settings there should always be exactly 100 windows
    assert len(windows) == 100


def test_no_useful_window_exception():
    windower = GeojsonRegionWindows(min_area=1.0, patch_size=(100, 100), coordinate_augmentation="none", stepsize=1)
    with pytest.raises(NoUsefulWindowException):
        windows = [x for x in windower.iter_valid_windows(GeoAnno.rectangle_builder((0, 0), (1000, 1000)), {1: 11.0})]


def test_deterministic_geojson_region_windows_multilevel():
    windower = GeojsonRegionWindows(
        min_area=1.0,
        patch_size=(100, 100),
        coordinate_augmentation="none",
        windowsize_augmentation="none",
        stepsize=1,
    )
    windows = [
        x
        for x in windower.iter_valid_windows(
            GeoAnno.rectangle_builder((0, 100), (1000, 1100)),
            {1: 5.0, 2: 10.0},
        )
    ]

    # There are four level 1 windows and 1 level 2 window:
    assert set(windows) == set(
        [
            (2, (0, 100), (100, 100)),
            (1, (500, 600), (100, 100)),
            (1, (500, 100), (100, 100)),
            (1, (0, 600), (100, 100)),
            (1, (0, 100), (100, 100)),
        ]
    )


def test_randomized_geojson_region_windows_multilevel():
    windower = GeojsonRegionWindows(
        min_area=0.5,
        patch_size=(100, 100),
        coordinate_augmentation="random",
        windowsize_augmentation="relative",
        stepsize=2,
    )
    windows = [
        x
        for x in windower.iter_valid_windows(
            GeoAnno.rectangle_builder((0, 100), (1000, 1100)),
            {1: 5.0, 2: 10.0},
        )
    ]

    # There are four level 1 windows and 1 level 2 window:
    assert len(windows) > 5
