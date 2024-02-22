import logging
import re
from typing import Callable
from pydantic import Field
from mmm.BaseModel import BaseModel

from .GeoAnno import GeoAnno


class AnnoFilterConfig(BaseModel):
    """
    Filters annotations by regex. All annotations without a class name are filtered by default.
    """

    delete_regex: str | None = Field(
        default=None,
        description="Only include annotations with class names that do not match",
    )
    keeponly_regex: str | None = Field(
        default=None,  # f".*({'|'.join([x[0] for x in recommended_buckets])}).*",
        description="Only include annotations with class names that match",
    )

    def apply(
        self,
        x: GeoAnno,
        additional_function: Callable[[GeoAnno], bool] | None = None,
        use_regions: list[GeoAnno] | None = None,
        ignore_regions: list[GeoAnno] | None = None,
    ) -> bool:
        """
        wsi_regions filters annotations that do not intersect with any of the wsi_regions
        """
        anno_class_name = x.get_class_name()

        if anno_class_name is None:
            logging.debug(f"omitting {x.shape.bounds}, {x.get_class_name()} because no class")
            return False

        anno_class_name = anno_class_name.lower()

        if self.delete_regex is not None:
            if re.match(self.delete_regex, anno_class_name):
                logging.debug(f"omitting {x.shape.bounds}, {x.get_class_name()} because {self.delete_regex=}")
                return False

        if self.keeponly_regex is not None:
            if not re.match(self.keeponly_regex, anno_class_name):
                logging.debug(f"omitting {x.shape.bounds}, {x.get_class_name()} because {self.keeponly_regex=}")
                return False

        if use_regions is not None:
            if not any([x.shape.intersects(y.shape) for y in use_regions]):
                logging.debug(f"omitting {x.shape.bounds}, {x.get_class_name()} because not in wsi regions")
                return False

        if ignore_regions is not None:
            if any([x.shape.intersects(y.shape) for y in ignore_regions]):
                logging.debug(f"omitting {x.shape.bounds}, {x.get_class_name()} because in ignore regions")
                return False

        # Should be the last statement
        return additional_function is None or additional_function(x)
