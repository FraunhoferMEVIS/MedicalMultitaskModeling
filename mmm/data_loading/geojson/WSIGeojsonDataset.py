import json
from functools import partial
import logging
from typing import Any
from torch.utils.data import Dataset
from mmm.BaseModel import BaseModel
from pathlib import Path

from .GeoAnno import GeoAnno
from .AnnoFilterConfig import AnnoFilterConfig


class WSIGeojsonDataset(Dataset[dict[str, Any]]):
    """
    Encapsulates WSIs and corresponding annotations.

    A WSI path can be anything, a local path, a URL, etc.
    """

    class Config(BaseModel):
        anno_filter: AnnoFilterConfig = AnnoFilterConfig()
        use_regions: AnnoFilterConfig | None = None
        ignore_regions: AnnoFilterConfig | None = None

    @staticmethod
    def load_annotations_from_json(json_path: Path) -> list[GeoAnno]:
        with open(json_path, "r") as f:
            json_dict = json.load(f)
            return [GeoAnno(x) for x in json_dict["features"]]

    def __init__(self, args: Config, wsi_anno_map: dict[Path, str]) -> None:
        self.args = args
        self.wsi_list = []

        outfiltered = []
        for anno_path, wsi_path in wsi_anno_map.items():
            annos: list[GeoAnno] = self.load_annotations_from_json(anno_path)

            original_anno_num = len(annos)
            # pre-filter annotations, e.g. by their class names

            if self.args.use_regions is not None:
                use_regions = list(filter(self.args.use_regions.apply, annos))
            else:
                use_regions = None

            if self.args.ignore_regions is not None:
                ignore_regions = list(filter(self.args.ignore_regions.apply, annos))
            else:
                ignore_regions = None
            anno_filter = partial(
                self.args.anno_filter.apply,
                use_regions=use_regions,
                ignore_regions=ignore_regions,
            )
            annos = list(filter(anno_filter, annos))

            # Only add this WSI if there are annotations left
            if annos:
                self.wsi_list.append((wsi_path, annos))
            else:
                outfiltered.append((wsi_path, original_anno_num))

        if outfiltered:
            logging.warning(f"Dropped {len(outfiltered)} WSI because no labels left after filtering: {outfiltered=}")

    def get_classes(self) -> list[str]:
        all_classes = set()
        for _, annos in self.wsi_list:
            for anno in filter(self.args.anno_filter.apply, annos):
                all_classes.add(anno.get_class_name())
        return list(all_classes)

    def __len__(self) -> int:
        return len(self.wsi_list)

    def __getitem__(self, index) -> dict[str, Any]:
        wsi_path, annos = self.wsi_list[index]

        return {"wsi": wsi_path, "annos": annos, "meta": {"id": str(wsi_path)}}

    def __repr__(self) -> str:
        wsi_stats = "".join([f"- {str(wsi_path)}: {len(annos)} annotations\n" for wsi_path, annos in self.wsi_list])
        return f"""
Length: {len(self)}\n\n
{wsi_stats}

{self.get_classes()}
        """
