import logging
from pathlib import Path
import random
import json

from pydantic import Field
import torch
import torch.nn as nn
from typing import Any, Dict, Tuple, List, cast, Optional

import numpy as np
import wandb
from mmm.logging.type_ext import StepFeedbackDict, StepMetricDict
from .MTLTask import MTLTask
from mmm.data_loading.DetectionDataset import DetectionDataset, eval_map_batch
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules

try:
    from mmdet.registry import MODELS
    from mmdet.models.utils import multi_apply
    from mmdet.models.dense_heads import FCOSHead, AnchorFreeHead
    from mmengine.structures import InstanceData, BaseDataElement
    from mmengine.config import ConfigDict
except ImportError:
    logging.warning("Detection extra dependencies not installed")
    FCOSHead = nn.Module
    InstanceData = Any


class MTLFCOSHead(FCOSHead):
    #     def _init_predictor(self) -> None:
    #         """Initialize predictor layers of the head."""
    #         self.conv_cls = nn.Conv2d(
    #             self.feat_channels, self.cls_out_channels, 3, padding=1)
    #         self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def forward_single(self, cls_feat, reg_feat, scale, stride: int):
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        # cls_score, bbox_pred, cls_feat, reg_feat = AnchorFreeHead.forward_single(self, x)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)

            # THE ONLY CHANGE COMPARED TO THE ORIGINAL FCOS HEAD's implementation
            # if not self.training:
            #     bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness

    def forward(self, cls_feats, reg_feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is \
            num_points * 4.
            - centernesses (list[Tensor]): centerness for each scale level, \
            each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, cls_feats, reg_feats, self.scales, self.strides)

    def loss_by_feat(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        centernesses: List[torch.Tensor],
        batch_gt_instances: list[InstanceData],
    ) -> Dict[str, torch.Tensor]:
        """
        Uses DDP blocking, but we don't share the tasks. Remove the blocking.

        So, remove the calls to "reduce_mean".
        Problematically, this method might diverge from the original implementation in future updates without noticing.
        If you know a better way to remove the reduce_mean calls, please update.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device
        )
        labels, bbox_targets = self.get_targets(all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_centerness = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(num_pos, 1.0)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(pos_centerness_targets.sum().detach(), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm,
            )
            loss_centerness = self.loss_centerness(pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_centerness=loss_centerness)


class MMDetectionTask(MTLTask):
    """
    The confidence score of a box is influenced by the classification output and the centerness output.
    """

    class Config(MTLTask.Config):
        encoder_key: str = "encoder"
        decoder_key: str = "fcosfpn"
        min_threshold_for_metrics: float = Field(
            default=0.05,
            description="Threshold after which confidence score boxes are considered in metric computation",
        )
        max_boxes: int = 100
        norm_on_bbox: bool = True
        centerness_on_reg: bool = True
        center_sampling: bool = False
        conv_bias: bool = True
        stacked_convs: int = 0
        dcn_on_last_conv: bool = False
        # head_kernel_size: int = Field(
        #     default=3,
        #     description="Boxes and classes are predicted with this kernel size."
        # )

    def __init__(
        self,
        args: Config,
        for_strides: List[int],
        in_channels: int,
        cohort: TrainValCohort[DetectionDataset],
    ) -> None:
        super().__init__(args, cohort)
        self.args: MMDetectionTask.Config
        self.class_names = cohort.datasets[0].vis_classes

        self.good_thres: Optional[float] = None

        self.strides = for_strides
        logging.debug(f"Setting up mmDetection task, fpn strides: {for_strides}")

        # I think feature channels are configurable, in_channels need to be the same as the output of the neck
        fcos_head: MTLFCOSHead = MTLFCOSHead(
            num_classes=len(self.class_names),
            in_channels=in_channels,
            stacked_convs=args.stacked_convs,
            feat_channels=in_channels,
            strides=for_strides,
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1 / np.log(len(self.class_names)),
            ),
            loss_bbox=dict(type="IoULoss", loss_weight=1.0),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.2),
            # Training tricks not mentioned in the original paper:
            norm_on_bbox=args.norm_on_bbox,
            centerness_on_reg=args.centerness_on_reg,
            dcn_on_last_conv=args.dcn_on_last_conv,
            center_sampling=args.center_sampling,
            conv_bias=args.conv_bias,
        )
        self.test_cfg = ConfigDict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=args.max_boxes,
        )

        self.task_modules: nn.ModuleDict = nn.ModuleDict(
            {
                # # The same convolution applied to each classification feature map.
                # # In consequence, FPN has same channel everywhere.
                "head": fcos_head
            }
        )
        # The normalization factor might be computable

    def get_head(self) -> MTLFCOSHead:
        return cast(MTLFCOSHead, self.task_modules["head"])

    def forward(self, x: Any, shared_blocks: Dict[str, SharedBlock]):
        pyr = shared_blocks[self.args.encoder_key](x)
        cls_features, reg_features = shared_blocks[self.args.decoder_key](pyr)
        cls_score, bbox_pred, centerness = self.task_modules["head"].forward(cls_features, reg_features)
        return cls_score, bbox_pred, centerness

    def prepare_batch(self, batch: List[Dict[str, Any]]) -> Any:
        for d in batch:
            d["image"] = d["image"].to(self.torch_device)
            d["boxes"] = d["boxes"].to(self.torch_device)
            d["labels"] = d["labels"].to(self.torch_device)
        return batch

    def features_to_boxes(
        self, cls_score, bbox_pred, centerness, img_metas
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        img_metas is a list with the mmdet image metadata for each input image:

        img_shape without channel dimension (e.g. (512, 512))
        scale_factor (e.g. 1.)
        """
        results_list = self.get_head().predict_by_feat(
            cls_score,
            ([box * s for box, s in zip(bbox_pred, self.strides)] if self.get_head().norm_on_bbox else bbox_pred),
            centerness,
            batch_img_metas=img_metas,
            rescale=False,
            cfg=self.test_cfg,
        )
        return results_list  # type: ignore

    def training_step(
        self, batch: List[Dict[str, Any]], shared_blocks: SharedModules
    ) -> Tuple[torch.Tensor, StepFeedbackDict]:
        images, metas = [], []
        targets: List[InstanceData] = []
        for d in batch:
            images.append(d["image"])
            if d["boxes"].shape[0] > 0:
                boxes_tensor = torch.clone(d["boxes"])
            else:
                boxes_tensor = torch.empty((0, 4)).float().to(d["image"].device)
            label_tens = d["labels"] if d["boxes"].shape[0] > 0 else torch.empty((0)).long().to(d["image"].device)
            targets.append(
                InstanceData(
                    bboxes=boxes_tensor,
                    labels=label_tens,
                    metainfo={"img_shape": d["image"].shape[1:], "scale_factor": 1.0},
                )
            )
            metas.append(d["meta"] if "meta" in d else {})

        cls_score, bbox_pred, centerness = shared_blocks.forward(torch.stack(images), self.forward)

        losses: Dict = self.get_head().loss_by_feat(
            cls_scores=cls_score,
            bbox_preds=bbox_pred,
            centernesses=centerness,
            batch_gt_instances=targets,
            # batch_img_metas=[],  # Unused in their implementation of FCOS [d.img_meta for d in targets]
            # batch_gt_instances_ignore=None
        )
        cls_loss, box_loss, centerness_loss = (
            losses["loss_cls"],
            losses["loss_bbox"],
            losses["loss_centerness"],
        )
        final_loss: torch.Tensor = cls_loss + box_loss + centerness_loss

        # Compute user feedback
        real_time_feedback: StepFeedbackDict = {k: lossval.item() for k, lossval in losses.items()}  # type: ignore

        with torch.no_grad():
            boxes_foreach_image = self.features_to_boxes(
                cls_score, bbox_pred, centerness, [d.metainfo for d in targets]
            )  # scores, boxes, labels
            # import pdb
            # pdb.set_trace()
            vis_n = min(self._takeout_vis_budget(), len(images))
            if vis_n > 0:
                img_index = random.randint(0, len(images) - 1)

                # boxes_foreach_image[img_index]
                wandb_img = self.visualize_prediction(
                    images[img_index].cpu(),
                    targets[img_index].bboxes.cpu(),
                    targets[img_index].labels.cpu(),
                    boxes_foreach_image[img_index].bboxes.cpu(),
                    boxes_foreach_image[img_index].scores.cpu(),
                    boxes_foreach_image[img_index].labels.cpu(),
                    metas[img_index],
                )
                real_time_feedback["preds"] = wandb_img

            filter_by_score = [
                img_preds_bboxes.scores > self.args.min_threshold_for_metrics
                for img_preds_bboxes in boxes_foreach_image
            ]

            pred_boxes = [
                img_preds_bboxes.bboxes[boxfilter, ...].clone().cpu().numpy()
                for boxfilter, img_preds_bboxes in zip(filter_by_score, boxes_foreach_image)
            ]
            pred_scores = [
                img_preds_bboxes.scores[boxfilter].clone().cpu().numpy()
                for boxfilter, img_preds_bboxes in zip(filter_by_score, boxes_foreach_image)
            ]
            pred_labels = [
                img_preds_bboxes.labels[boxfilter].clone().cpu().numpy()
                for boxfilter, img_preds_bboxes in zip(filter_by_score, boxes_foreach_image)
            ]

            step_metrics: StepMetricDict = {
                "gtboxes": [d.bboxes.clone().cpu().numpy().astype(np.int64) for d in targets],
                "gtlabels": [d.labels.clone().cpu().numpy() for d in targets],
                "predboxes": pred_boxes,
                "predscores": pred_scores,
                "predlabels": pred_labels,
            }  # type: ignore
        self.add_step_result(final_loss.item(), step_metrics)
        return final_loss, real_time_feedback

    @torch.no_grad()
    def visualize_prediction(
        self,
        img: torch.Tensor,
        img_boxes: List,
        img_labels: List,
        pred_boxes: List,
        pred_scores: List,
        pred_labels: List,
        meta: Dict,
    ) -> wandb.Image:
        gt_boxes = [
            {
                "position": {
                    "minX": int(box[0]),
                    "maxX": int(box[2]),
                    "minY": int(box[1]),
                    "maxY": int(box[3]),
                },
                "domain": "pixel",
                "class_id": int(box_label),
            }
            for box, box_label in zip(img_boxes, img_labels)
        ]

        vis_boxes = [
            {
                "position": {
                    "minX": int(box[0]),
                    "maxX": int(box[2]),
                    "minY": int(box[1]),
                    "maxY": int(box[3]),
                },
                "domain": "pixel",
                "class_id": int(box_label),
                "scores": {"score": box_score.item()},
                # "box_caption": "test_caption"
            }
            for box, box_score, box_label in zip(pred_boxes, pred_scores, pred_labels)
        ]

        metastr = json.dumps(meta, default=lambda o: str(o))
        im = wandb.Image(
            img,
            boxes={
                "predictions": {
                    "class_labels": {i: f"{v}_pred" for i, v in enumerate(self.class_names)},
                    "box_data": vis_boxes,
                },
                "ground_truth": {
                    "class_labels": {i: f"{v}_gt" for i, v in enumerate(self.class_names)},
                    "box_data": gt_boxes,
                },
            },
            caption=f"{np.min(img.numpy()):.3f}, {np.max(img.numpy()):.3f}\n{img.shape}\n{metastr}",
        )
        return im

    def determine_thresholds(self, proposal_thresholds: List[float]):
        # threshold -> mAP
        res: dict[float, tuple[float, Any]] = {}
        for t in proposal_thresholds:
            predboxes = [
                np.hstack([m[score > t], score[score > t].reshape(-1, 1)])
                for metrics in self._step_metrics
                for score, m in zip(metrics["predscores"], metrics["predboxes"])
            ]
            predlabels = [
                m[score > t]
                for metrics in self._step_metrics
                for score, m in zip(metrics["predscores"], metrics["predlabels"])
            ]
            gtboxes = [m for metrics in self._step_metrics for m in metrics["gtboxes"]]
            gtlabels = [m for metrics in self._step_metrics for m in metrics["gtlabels"]]

            mean_ap, details = eval_map_batch(predboxes, predlabels, gtboxes, gtlabels, self.class_names)
            res[t] = mean_ap, details
        return res

    def save_checkpoint(self, folder_path: Path):
        super().save_checkpoint(folder_path)
        meta_dict = {}
        if self.good_thres is not None:
            # Save the best threshold as json to folder_path
            meta_dict["threshold"] = self.good_thres
        with open(folder_path / "meta.json", "w") as f:
            json.dump(meta_dict, f)

    def load_checkpoint(self, folder_path: Path):
        super().load_checkpoint(folder_path)
        with open(folder_path / "meta.json", "r") as f:
            meta = json.load(f)
        if "threshold" in meta:
            self.good_thres = meta["threshold"]

    def log_epoch_metrics(self) -> Tuple[Dict[str, Any], str]:
        # Build one huge "batch" with all results of the epoch and compute metrics:

        metrics, logstring = super().log_epoch_metrics()
        # assert self.args.score_threshold_for_metrics is not None

        # if self.training:
        #     if self.good_thres is None:
        #         thresholds = self.determine_thresholds(list(np.arange(self.args.min_threshold_for_metrics, 0.9, 0.1)))
        #     else:
        #         proposals = list(filter(lambda x: 0.95 >= x >= self.args.min_threshold_for_metrics, [
        #             self.good_thres - 0.05,
        #             self.good_thres,
        #             self.good_thres + 0.05,
        #         ]))
        #         thresholds = self.determine_thresholds(proposals)

        #     best_threshold = max(thresholds, key=lambda k: thresholds[k][0])
        #     self.good_thres = best_threshold
        # else:
        #     if self.good_thres is None:
        #         thresholds = self.determine_thresholds([self.args.min_threshold_for_metrics])
        #     else:
        #         thresholds = self.determine_thresholds([self.good_thres])

        thresholds = self.determine_thresholds([self.args.min_threshold_for_metrics])
        best_threshold = max(thresholds, key=lambda k: thresholds[k][0])
        metrics[f"mAP"] = thresholds[best_threshold][0]
        metrics[f"best_threshold"] = best_threshold
        logstring = f"{logstring} - map{best_threshold:.2f} {thresholds[best_threshold][0]:.3f}"
        for class_ap, class_name in zip(thresholds[best_threshold][1], self.class_names):
            metrics[f"{class_name}AP"] = class_ap["ap"]

        return metrics, logstring
