"""
GeoJSON annotations are aimed to be compatible with QuPath.
"""

from __future__ import annotations
import math
from typing import Type
from enum import Enum
import numpy as np
import json

import shapely
from shapely.geometry import shape, Polygon


class AnnotationType(Enum):
    # Perfect for a detection task
    DetectionBox = "DetectionBox"
    # Perfect for a semantic segmentation task with close boundaries
    Contour = "Contour"
    # Single object with multiple regions
    MultiContour = "MultiContour"


class GeoAnno:
    """
    Wraps a single QuPath geojson annotation such as

    ```
    {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [
                        7624,
                        9051
                    ],
                    [
                        7624,
                        9227
                    ],
                    [
                        7845,
                        9227
                    ],
                    [
                        7845,
                        9051
                    ],
                    [
                        7624,
                        9051
                    ]
                ]
            ]
        },
        "properties": {
            "object_type": "annotation",
            "classification": {
                "name": "Dust_8.1.1",
                "colorRGB": -4987213
            },
            "isLocked": false
        }
    }
    ```
    """

    @classmethod
    def from_shapely(cls, shape: shapely.Geometry) -> GeoAnno:
        json_dict = {
            "type": "Feature",
            "geometry": shapely.geometry.mapping(shape),
            "properties": {
                "object_type": "annotation",
            },
        }
        return cls(json_dict)

    @classmethod
    def rectangle_builder(
        cls: Type[GeoAnno],
        left_upper: tuple[int, int],
        right_lower: tuple[int, int],
        class_name="defaultclass",
    ) -> GeoAnno:
        return cls(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [left_upper[0], left_upper[1]],
                            [right_lower[0], left_upper[1]],
                            [right_lower[0], right_lower[1]],
                            [left_upper[0], right_lower[1]],
                            [left_upper[0], left_upper[1]],
                        ]
                    ],
                },
                "properties": {
                    "object_type": "annotation",
                    "classification": {"name": class_name, "color": [200, 0, 0]},
                },
            }
        )

    def __init__(self, json_dict: dict, origin_id: str | None = None) -> None:
        self.wsi_id: str | None = origin_id
        self.shape = shape(json_dict["geometry"])

        self.properties: dict = json_dict["properties"] if "properties" in json_dict else {}

        # QuPath needs a color
        try:
            self.color: list[int] = self.properties["classification"]["color"]
            assert len(self.color) == 3
        except KeyError:
            self.color = [200, 0, 0]

        if self.shape.geom_type == "LineString":
            # Convert to Polygon, because a LineString has no area and we depend on that
            self.shape = self.shape.buffer(np.sqrt(self.shape.length))

    def set_class_name(self, class_name: str):
        if "classification" not in self.properties:
            self.properties["classification"] = {
                "name": class_name.lower(),
                "color": self.color,
            }
        else:
            self.properties["classification"]["name"] = class_name.lower()

    def get_class_name(self) -> str | None:
        try:
            return self.properties["classification"]["name"].lower()
        except KeyError:
            return None

    def get_enclosing_region(self) -> tuple[tuple[int, int], tuple[int, int]]:
        bounds: tuple[int, int, int, int] = self.shape.bounds  # type: ignore
        return (int(bounds[0]), int(bounds[1])), (
            int(bounds[2] - bounds[0]),
            int(bounds[3] - bounds[1]),
        )

    def overlaps_with(self, other_shape: Polygon, overlap_threshold: float = 0.0) -> bool:
        if self.shape.intersects(other_shape):
            intersection_shape = self.shape.intersection(other_shape)
            return intersection_shape.area > (self.shape.area * overlap_threshold)
        return False

    def get_annotation_type(self) -> AnnotationType:
        """
        If the annotation looks like a perfect box, it returns AnnotationType.DetectionBox.
        Otherwise: AnnotationType.Contour or AnnotationType.MultiContour
        """
        if math.isclose(self.shape.area, self.shape.minimum_rotated_rectangle.area):
            return AnnotationType.DetectionBox
        elif self.shape.geom_type == "Polygon":
            return AnnotationType.Contour
        elif self.shape.geom_type == "MultiPolygon":
            return AnnotationType.MultiContour
        else:
            raise NotImplementedError(f"Not implemented for: {self.shape.geom_type}")

    def visualize_as_pyplot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        if c := self.get_class_name():
            ax.set_title(f"Class {c}")
        ax.plot(*self.shape.exterior.xy)
        return fig

    def to_geojson(self) -> dict:
        res = {
            "type": "Feature",
            "geometry": shapely.geometry.mapping(self.shape),
        }
        if self.properties:
            res["properties"] = self.properties
        return res

    def __repr__(self) -> str:
        return f"GeoAnno:\n{self.shape.geom_type=};{self.shape.area=}\n{json.dumps(self.properties, indent=2)}"
