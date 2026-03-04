import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import wandb
from pydantic import Field

from mmm.data_loading.DetectionDataset import DetectionDataset, eval_map_batch
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.logging.type_ext import StepFeedbackDict, StepMetricDict
from mmm.mmm_types.GroupUsage import GroupingStrategy, GroupUsage, MaskingStrategy
from mmm.mtl_modules.shared_blocks.FCOSDecoder import FCOSDecoder
from mmm.mtl_modules.shared_blocks.Grouper import Grouper, make_grid_for_supercase
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules

from .MTLTask import MTLTask

try:
    from mmdet.models.dense_heads import FCOSHead
    from mmdet.models.utils import multi_apply
    from mmengine.config import ConfigDict
    from mmengine.structures import InstanceData
except ImportError:
    logging.warning(
        "Detection extras are not installed. For detection, install MMCV manually, as it is not available on PyPI."
    )
    FCOSHead = nn.Module
    InstanceData = Any


class MTLFCOSHead(FCOSHead):
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
    Requires to add mmcv and detection extra dependencies.
    Depending on your environment, a statement like could be added to your pyproject.toml:
    `mmcv = { url = "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl" }`

    The confidence score of a box is influenced by the classification output and the centerness output.
    """

    class Config(MTLTask.Config):
        encoder_key: str = "encoder"
        decoder_key: str = "fcosfpn"
        squeezer_key: str = "squeezer"
        grouper_key: GroupUsage = Field(
            default=GroupUsage(grouper_key="", masking=MaskingStrategy.fullattention),
            description="If set, a shared Grouper is used, and the mixer specified by mixer_key will be used.",
        )
        mixer_key: str = Field(
            default="",
            description="If set, a shared FeatureMixer is used. The grouper_key must be set as well.",
        )
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
        multiscale_invariance: bool = Field(
            default=True,
            description="""If no multiscale invariance is required, the detection task can use segmentation features.
Multiscale invariance uses an FPN.""",
        )

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
        assert self.args.grouper_key.grouping is GroupingStrategy.full

        self.good_thres: Optional[float] = None

        self.strides = for_strides if self.args.multiscale_invariance else [for_strides[0]]
        logging.debug(f"Setting up mmDetection task, {self.strides=}")

        # I think feature channels are configurable, in_channels need to be the same as the output of the neck
        min_fac = 5  # reproduces regress ranges of the original FCOS implementation if 5 strides are used
        fpn_regress_ranges = [
            (-1 if i == min_fac else 2**i, 2 ** (i + 1) if i < len(self.strides) + min_fac - 1 else 1e8)
            for i in range(min_fac, min_fac + len(self.strides))
        ]
        fcos_head: MTLFCOSHead = MTLFCOSHead(
            num_classes=len(self.class_names),
            regress_ranges=fpn_regress_ranges,
            in_channels=in_channels,
            stacked_convs=args.stacked_convs,
            feat_channels=in_channels,
            strides=self.strides,
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=(1 / np.log(len(self.class_names))) if len(self.class_names) > 1 else 1.0,
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
        self.flatten = nn.Flatten(1)
        # The normalization factor might be computable

    def get_head(self) -> MTLFCOSHead:
        return cast(MTLFCOSHead, self.task_modules["head"])

    def forward(self, inputs: Any, shared_blocks: Dict[str, SharedBlock]):
        x, supercase_indexes, contexts = inputs if len(inputs) == 3 else (inputs[0], inputs[1], None)

        pyr = shared_blocks[self.args.encoder_key](x)

        # for Backward compatibility
        if hasattr(self.args, "squeezer_key") and self.args.squeezer_key in list(shared_blocks.keys()):
            pyr[-1], hidden_vector = shared_blocks[self.args.squeezer_key](pyr)

        if self.args.grouper_key.grouper_key:
            hidden_vector = self.flatten(hidden_vector)
            if hasattr(self.args, "positions") and self.args.positions is not None:
                positions = [c[self.args.positions[0]] for c in contexts]
            else:
                positions = None
            hidden_vector, self._grouper_meta = shared_blocks[self.args.grouper_key.grouper_key](
                hidden_vector, supercase_indexes, self.args.grouper_key, positions=positions
            )

        if self.args.mixer_key:
            assert self.args.grouper_key.grouper_key
            mixer = shared_blocks[self.args.mixer_key]
            pyr = mixer(hidden_vector, pyr, supercase_indexes)

        if isinstance(decoder := shared_blocks[self.args.decoder_key], FCOSDecoder):
            assert self.args.multiscale_invariance, "FCOSDecoder can only be used with multiscale invariance"
            cls_features, reg_features = decoder.forward(pyr)
        else:
            segmentation_features = (
                decoder.forward_fpn(pyr)[::-1] if self.args.multiscale_invariance else [decoder.forward(pyr)]
            )
            cls_features, reg_features = segmentation_features, segmentation_features
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

        supercase_indices = Grouper.extract_ids_from_batch(
            [x.get("group_id", None) for x in metas], for_task_name=self.get_name()
        ).to(self.torch_device)

        cls_score, bbox_pred, centerness = shared_blocks.forward(
            (torch.stack(images), supercase_indices, [x.get("context") for x in metas]), self.forward
        )

        with torch.cuda.amp.autocast(enabled=False):  # disable autocast to be able to cast types here
            losses: Dict = self.get_head().loss_by_feat(
                cls_scores=[x.float() for x in cls_score],
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
            real_time_feedback.update(
                self.visualize_case(images, supercase_indices, targets, boxes_foreach_image, metas)
            )

            filter_by_score = [
                img_preds_bboxes.scores > self.args.min_threshold_for_metrics
                for img_preds_bboxes in boxes_foreach_image
            ]

            pred_boxes = [
                img_preds_bboxes.bboxes[boxfilter, ...].clone().cpu().float().numpy()
                for boxfilter, img_preds_bboxes in zip(filter_by_score, boxes_foreach_image)
            ]
            pred_scores = [
                img_preds_bboxes.scores[boxfilter].clone().cpu().float().numpy()
                for boxfilter, img_preds_bboxes in zip(filter_by_score, boxes_foreach_image)
            ]
            pred_labels = [
                img_preds_bboxes.labels[boxfilter].clone().cpu().float().numpy()
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
    def visualize_case(self, images, supercase_indices, targets, boxes_foreach_image, metas) -> dict[str, Any]:
        vis_n = min(self.ask_for_visualization(), len(images))
        if vis_n <= 0:
            return {}

        grid_img, weight_str, vis_indices, with_annos = make_grid_for_supercase(
            torch.stack(images).cpu(),
            supercase_indices,
            supercase_index := random.choice(supercase_indices),
            self._grouper_meta if hasattr(self, "_grouper_meta") else None,
            with_boxes=dict(
                gtboxes=[{"bboxes": d.bboxes.cpu().clone(), "labels": d.labels.cpu().clone()} for d in targets],
                predboxes=[
                    {"bboxes": b.bboxes.cpu().clone(), "scores": b.scores.cpu().clone(), "labels": b.labels.cpu()}
                    for b in boxes_foreach_image
                ],
            ),
        )

        wandb_img = self.visualize_prediction_single(
            grid_img,
            torch.cat([x["bboxes"] for x in with_annos["boxes"]["gtboxes"]]),
            torch.cat([x["labels"] for x in with_annos["boxes"]["gtboxes"]]),
            torch.cat([x["bboxes"] for x in with_annos["boxes"]["predboxes"]]),
            torch.cat([x["scores"] for x in with_annos["boxes"]["predboxes"]]),
            torch.cat([x["labels"] for x in with_annos["boxes"]["predboxes"]]),
            [metas[i] for i in vis_indices],
            extra=f"supercase {supercase_index} {weight_str} {[metas[i].get('context') for i in vis_indices]}",
        )
        return {"preds": wandb_img}

    @torch.no_grad()
    def visualize_prediction_single(
        self,
        img: torch.Tensor,
        img_boxes: List,
        img_labels: List,
        pred_boxes: List,
        pred_scores: List,
        pred_labels: List,
        meta: Dict,
        extra: str = "",
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
            caption=f"{np.min(img.numpy()):.3f}, {np.max(img.numpy()):.3f}\n{img.shape}\n{metastr}\n{extra}",
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

        thresholds = self.determine_thresholds([self.args.min_threshold_for_metrics])
        best_threshold = max(thresholds, key=lambda k: thresholds[k][0])
        metrics[f"mAP"] = thresholds[best_threshold][0]
        metrics[f"best_threshold"] = best_threshold
        logstring = f"{logstring} - map{best_threshold:.2f} {thresholds[best_threshold][0]:.3f}"
        for class_ap, class_name in zip(thresholds[best_threshold][1], self.class_names):
            metrics[f"{class_name}AP"] = class_ap["ap"]

        return metrics, logstring
