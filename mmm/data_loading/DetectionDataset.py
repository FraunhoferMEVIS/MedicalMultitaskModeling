import logging
import numpy as np
import torch
import cv2
from typing import List, Optional, Callable, Dict, Any, TypeVar, Tuple

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .MTLDataset import MTLDataset, SrcCaseType
from .SemSegDataset import SemSegDataset

try:
    from mmdet.evaluation.functional import eval_map
except ImportError:
    eval_map = Any


def get_mmdet_annos(gtboxes, gtlabels):
    return [{"bboxes": box, "labels": lbl} for box, lbl in zip(gtboxes, gtlabels)]


def get_mmdet_preds(predboxes, predlabels, class_names: List[str]):
    det_results = []
    for image_pred_boxes, image_pred_labels in zip(predboxes, predlabels):
        # A list of predictions for each class
        class_boxes = [
            [box for i, box in enumerate(image_pred_boxes) if image_pred_labels[i] == cls_index]
            for cls_index in range(len(class_names))
        ]
        det_results.append([np.array(boxes) if boxes else np.empty((0, 5)) for boxes in class_boxes])
    return det_results


def eval_map_batch(predboxes, predlabels, gtboxes, gtlabels, class_names: List[str]):
    preds = get_mmdet_preds(predboxes, predlabels, class_names)
    annos = get_mmdet_annos(gtboxes, gtlabels)

    return eval_map(
        preds,
        annos,
        logger="silent",
    )


class DetectionDataset(MTLDataset):
    """
    A 2D detection dataset should consist of:

    - "image" -> FloatTensor[C, H, W] in the 0-1 range
    - "boxes" -> FloatTensor[N, 4] in [x1, y1, x2, y2] format (pascal_voc format)
    - "labels" -> Int64Tensor[N], the class label for each box
    """

    @staticmethod
    def convert_coco_box_to_pascal_box(x1, y1, w, h):
        return [max(x1, 0.0), max(y1, 0.0), x1 + w, y1 + h]

    def batch_collater(self) -> Callable:
        """
        For detection tasks the common batch format seems to be a list of cases.
        By default, collate_fn overwriters work with a list of cases.
        """
        # No inputs are stacked, detection tasks want lists!
        batch_type = TypeVar("batch_type", bound=List[Dict[str, Any]])

        def f(x: batch_type) -> batch_type:
            return x

        return f

    @staticmethod
    def from_semseg(ds: SemSegDataset, for_class_indices: Tuple[int, ...]):
        """Computes connected components in the mask and each connected component will get a box"""

        def convert_semseg_to_detection_case(semseg_case: Dict[str, Any]) -> Dict[str, Any]:
            assert "image" in semseg_case and "label" in semseg_case

            boxes = []
            box_labels = []

            for class_index in for_class_indices:
                lbl_for_class = (semseg_case["label"].numpy() == class_index).astype(np.uint8)
                contours, _ = cv2.findContours(lbl_for_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    # Suppress boxes from a single pixel
                    # if w > 0 and h > 0:
                    boxes.append([x, y, (x + w), y + h])
                    box_labels.append(class_index)

            semseg_case["boxes"] = torch.FloatTensor(boxes)
            semseg_case["labels"] = torch.LongTensor(box_labels)

            return semseg_case

        return DetectionDataset(
            src_ds=ds,
            src_transform=convert_semseg_to_detection_case,
            class_names=ds.class_names,
        )

    def __init__(
        self,
        src_ds: Dataset[SrcCaseType],
        src_transform: Optional[Callable[[SrcCaseType], Dict[str, Any]]] = None,
        class_names: Optional[List[str]] = None,
        collate_fn: Optional[Callable] = None,
        *args,
        **kwargs,
    ) -> None:
        assert class_names is not None, "Detection datasets without class names are deprecated"
        self.vis_classes: List[str] = class_names
        if collate_fn is not None:
            collate_fn = transforms.Compose([collate_fn, self.batch_collater()])
        else:
            collate_fn = self.batch_collater()
        super().__init__(
            src_ds,
            ["image", "boxes", "labels"],
            ["original_size", "meta"],
            src_transform,
            collate_fn=collate_fn,
            *args,
            **kwargs,
        )

    def verify_case_by_index(self, index: int) -> Dict[str, Any]:
        case = super().verify_case_by_index(index)
        self.assert_image_data_assumptions(case["image"])
        assert (
            case["boxes"].shape[0] == case["labels"].shape[0]
        ), f"Number of boxes and box labels do not match: {case['boxes'].shape[0]=} {case['labels'].shape[0]=}"
        # if len(case['boxes'].shape) != 2:
        #     print("test")
        if case["boxes"].shape[0] > 0:  # there are boxes
            assert len(case["boxes"].shape) == 2, f"{len(case['boxes'].shape)=} should be 2"
            assert case["boxes"].shape[-1] == 4, f"{case['boxes'].shape=}"

            assert isinstance(case["boxes"], torch.FloatTensor)
            for box in case["boxes"]:
                x1, y1, x2, y2 = box
                height, width = case["image"].shape[1:]
                conds = [
                    x1 >= 0,
                    x1 <= x2,
                    x2 >= x1,
                    x2 <= width,
                    y1 >= 0,
                    y1 <= y2,
                    y2 >= y1,
                    y2 <= height,
                ]
                assert False not in conds, f"Box sanity check failed for {box} because {conds=}"

            assert isinstance(case["labels"], torch.LongTensor)
            if self.vis_classes is not None:
                # Make sure the labels make sense
                for label in case["labels"]:
                    assert label >= 0 <= len(self.vis_classes)

        return case

    def set_classes_for_visualization(self, classes: List[str]):
        self.vis_classes = classes

    def get_classes_for_visualization(
        self,
    ):
        assert self.vis_classes is not None
        return self.vis_classes

    def st_case_viewer(self, case: Dict[str, Any], i: int) -> None:
        import streamlit as st
        from mmm.logging.st_ext import blend_with_mask

        st.title("Untransformed image:")
        im = case["image"]
        blend_with_mask(
            im,
            None,
            caption_suffix=f"Shape: {im.shape}",
            classes=self.vis_classes,
            boxes=(case["boxes"], case["labels"]),
            st_key=f"c{i}",
        )
        st.write(case)

    def _visualize_batch_case(self, batch: List[Dict[str, Any]], i: int) -> None:
        import streamlit as st
        from mmm.logging.st_ext import blend_with_mask

        patch = batch[i]["image"]
        patch_boxes = batch[i]["boxes"]
        patch_labels = batch[i]["labels"]
        # class_name = "No class names known" if self.vis_classes is None else self.vis_classes[batch['class'][i]]
        # st.write(f"Label: {batch['class'][i]}, " + class_name)
        blend_with_mask(
            patch,
            None,
            caption_suffix=f"{i}/{self._compute_batchsize_from_batch(batch)}: {patch.shape}",
            classes=self.vis_classes,
            boxes=(patch_boxes, patch_labels),
            st_key=f"b{i}",
        )
        if "meta" in batch[i]:
            st.write(batch[i]["meta"])

    def _compute_batchsize_from_batch(self, batch: List[Dict[str, Any]]) -> int:
        return len(batch)
