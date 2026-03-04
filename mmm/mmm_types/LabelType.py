from __future__ import annotations

from enum import Enum


class LabelType(str, Enum):
    seg = "seg"
    clf = "clf"
    surv = "surv"
    reg = "reg"
    geomask = "geomask"
    volume_seg = "volume_seg"

    @staticmethod
    def from_string(labeltype: str) -> LabelType:
        return {
            "choices": LabelType.clf,
            "brushlabels": LabelType.seg,
            "survival": LabelType.surv,
            "number": LabelType.reg,
            "geomask": LabelType.geomask,
            "volume3dmask": LabelType.volume_seg,
            "textarea": LabelType.surv,  # for backwards compatibility
        }[labeltype.lower()]

    @staticmethod
    def get_segmentation_types():
        return {LabelType.seg, LabelType.volume_seg, LabelType.geomask}
