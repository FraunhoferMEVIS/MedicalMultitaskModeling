import logging
import random
import json
from pydantic import Field
from typing import Dict, Tuple, Any, List, cast, Literal, Union, Optional

import wandb
import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms.functional as F
from torchvision.transforms import Resize, InterpolationMode

from mmm.logging.type_ext import StepMetricDict

from mmm.mtl_modules.shared_blocks.PyramidDecoder import PyramidDecoder

from .MTLTask import MTLTask
from mmm.settings import mtl_settings
from mmm.utils import flatten_list_of_dicts
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.data_loading.SemSegDataset import SemSegDataset
from mmm.neural.module_conversions import convert_2d_to_3d
from mmm.mtl_modules.shared_blocks.MTLDecoder import MTLDecoder
from mmm.neural.losses import SMPDiceFocalLoss2D

import segmentation_models_pytorch.metrics as smp_metrics


@torch.no_grad()
def get_stats_multilabel(
    output: torch.LongTensor,
    target: torch.LongTensor,
    ignore_value: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expects [B, C, *dims] tensors of predictions and targets.
    Not logits or scores!
    Returns tp,fp,fn,tn per sample per class.

    >>> # A batch with one image, with one class
    >>> from mmm.mtl_modules.tasks.SemSegTask import get_stats_multilabel; import torch
    >>> preds11 = torch.Tensor([[[[0, 1], [0, 1]]]]).long()
    >>> trues11 = torch.Tensor([[[[0, -1], [0, 1]]]]).long()
    >>> get_stats_multilabel(preds11, trues11, ignore_value=-1)
    (tensor([[1]]), tensor([[0]]), tensor([[0]]), tensor([[2]]))
    >>> preds = torch.Tensor([[[[0, 1], [0, 1]], [[1, 1], [0, 1]]], [[[0, 0], [1, 1]], [[0, 0], [0, 1]]]]).long()
    >>> trues = torch.Tensor([[[[-1, -1], [1, 1]], [[0, 1], [0, -1]]], [[[0, -1], [-1, 0]], [[0, -1], [0, -1]]]]).long()
    >>> # True negatives of the second image of the second class
    >>> get_stats_multilabel(preds, trues, ignore_value=-1)[3][1, 1]
    tensor(2)
    """
    output, target = output.clone().long(), target.clone().long()

    if ignore_value is not None:
        mask = target != ignore_value
        output, target = output * mask, target * mask

    tp = (output * target).sum(dim=(-1, -2))
    fp = (output * (1 - target)).sum(dim=(-1, -2))
    fn = ((1 - output) * target).sum(dim=(-1, -2))
    # Ignored values will appear in this count
    tn = ((1 - output) * (1 - target)).sum(dim=(-1, -2))

    # Subtract the number of ignored values from tn
    tn -= (mask == False).long().sum(dim=(-1, -2))

    return tp, fp, fn, tn


def get_single_metrics(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    original_shape: Tuple[int, int],
    class_names: List[str],
    ignore_index: int = -1,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
):
    """
    Expects single images, not batches!
    """
    # Sometimes the metrics are desired on the original image size, not the network output size
    if original_shape != pred_mask.shape[-2:]:
        if mode == "multiclass":
            assert len(pred_mask.shape) == len(original_shape)
            assert len(true_mask.shape) == len(original_shape)
        elif mode == "multilabel":
            assert len(pred_mask.shape) == len(original_shape) + 1
            assert len(true_mask.shape) == len(original_shape) + 1
        resize_function = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)
        _pred_mask = resize_function(torch.unsqueeze(pred_mask, dim=0))
        _true_mask = resize_function(torch.unsqueeze(true_mask, dim=0))
    else:
        _pred_mask: torch.LongTensor = torch.unsqueeze(pred_mask, dim=0)  # type: ignore
        _true_mask: torch.LongTensor = torch.unsqueeze(true_mask, dim=0)  # type: ignore

    # Sadly, smp can only compute metrics with ignore_index outside the valid class indices
    num_classes = len(class_names)
    if mode == "multiclass":
        # _pred_mask[_pred_mask == ignore_index] = -1
        # _true_mask[_true_mask == ignore_index] = -1
        # _pred_mask[_pred_mask > ignore_index] -= 1
        # _true_mask[_true_mask > ignore_index] -= 1
        # num_classes -= 1

        return smp_metrics.get_stats(_pred_mask, _true_mask, mode=mode, num_classes=num_classes, ignore_index=-1)
    elif mode == "multilabel":
        return get_stats_multilabel(_pred_mask, _true_mask, ignore_value=ignore_index)


def get_batch_metrics(
    pred_masks: torch.Tensor,
    true_masks: torch.Tensor,
    original_shapes: List[Tuple[int, int]],
    class_names: List[str],
    ignore_index: int = -1,
    mode: Literal["multiclass", "multilabel"] = "multiclass",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Setting original shape is important because the metrics are better on a smaller image size!

    original shapes is a list of the shapes for the whole batch:
    """
    res = []
    # assert isinstance(pred_masks, torch.LongTensor) and isinstance(true_masks, torch.LongTensor)
    assert pred_masks.shape == true_masks.shape

    # if original_shapes is None:
    #     return smp_metrics.get_stats(
    #         pred_masks,
    #         true_masks,
    #         mode="multiclass", num_classes=len(class_names)
    #     )  # type: ignore (They annotated the tuple wrong)
    # self.args.batch_size cannot be used, because the last batch can be smaller than batch_size
    for index_in_batch in range(pred_masks.shape[0]):
        single_image_metrics = get_single_metrics(
            pred_masks[index_in_batch],
            true_masks[index_in_batch],
            tuple(original_shapes[index_in_batch]),
            class_names,
            ignore_index=ignore_index,
            mode=mode,
        )
        res.append(single_image_metrics)
    tp = torch.concat([x[0] for x in res])
    fp = torch.concat([x[1] for x in res])
    fn = torch.concat([x[2] for x in res])
    tn = torch.concat([x[3] for x in res])
    return tp, fp, fn, tn


class SemSegTask(MTLTask):
    """
    Semantic segmentation task. Uses a fixed loss function by adding focalloss and dice loss together.

    Values in the mask that are equal to -1 will be ignored in the loss and metrics.
    This value can also be predicted when the best predicted class is below the confidence threshold.

    For building the task-specific heads it needs the number of output channels of the decoder it should use.

    Expects a dataset with tuples (image, mask). For example:

    >>> from mmm.interactive import *
    >>> from mmm.data_loading.synthetic.shape_dataset import ShapeDataset, CanvasConfig
    >>> ds = ShapeDataset(ShapeDataset.Config())
    >>> ssds = data.SemSegDataset(
    ...     src_ds=ds,
    ...     src_transform=ShapeDataset.cohort_transform(),
    ...     class_names=["background"] + ds.get_class_names())
    >>> t = tasks.SemSegTask(
    ...     args=configs.SemSegTaskConfig(module_name="TestSegmentation"),
    ...     cohort=data.TrainValCohort(configs.TrainValCohortConfig(), ssds, val_ds=None),
    ...     for_decoder=blocks.PyramidDecoder(PyramidDecoder.Config(), [32, 64, 128], 32),
    ...     class_names=ssds.class_names
    ... )
    """

    class Config(MTLTask.Config):
        encoder_key: str = "encoder"
        decoder_key: str = "decoder"
        squeezer_key: str = "squeezer"
        decoder_type: Literal["pyramid", "mtldecoder"] = "pyramid"
        loss: Literal["dicefocal"] = "dicefocal"
        dropout: float = 0.0
        headdropout: float = 0.0
        head_kernel_size: int = Field(default=3, description="Controls the receptive field of the head.")
        mask_key: str = "label"
        crit: SMPDiceFocalLoss2D.Config = SMPDiceFocalLoss2D.Config()
        confidence_threshold: float = 0.5

    def __init__(
        self,
        class_names: List[str],
        for_decoder: Union[MTLDecoder, PyramidDecoder],
        args: Config,
        cohort: TrainValCohort[SemSegDataset],
    ):
        super().__init__(args, cohort)
        self.args: SemSegTask.Config
        self.class_names = class_names
        if cohort and self.class_names != cohort.datasets[0].class_names:
            logging.warning(
                f"Class names {self.class_names} do not match dataset class names {cohort.datasets[0].class_names}"
            )

        self.task_modules = self.build_head(for_decoder)
        self.crit = self.build_loss()

    def build_head(self, for_decoder: Optional[Union[MTLDecoder, PyramidDecoder]] = None):
        return nn.ModuleDict(
            {
                "segmentation_head": nn.Conv2d(
                    for_decoder.get_output_dim_per_pixel(),
                    len(self.class_names),
                    kernel_size=self.args.head_kernel_size,
                    padding=self.args.head_kernel_size // 2,
                ),
                "dropout": nn.ModuleList(
                    [(nn.Dropout2d(p=self.args.dropout) if self.args.dropout > 0 else nn.Identity()) for _ in range(4)]
                ),
                "headdropout": (nn.Dropout2d(p=self.args.headdropout) if self.args.headdropout > 0 else nn.Identity()),
            }
        )

    def build_loss(self):
        return SMPDiceFocalLoss2D(
            self.args.crit,
            mode="multiclass",
            ignore_index=mtl_settings.ignore_class_value,
        )

    def prepare_batch(self, batch: Dict[str, Any]) -> Any:
        batch["image"] = batch["image"].to(self.torch_device)
        spatial_dims = batch["image"].size()[2:]  # type: ignore
        assert False not in [
            dim % 32 == 0 for dim in spatial_dims
        ], f"All spatial dimensions need to be divisable by 32, but you have {spatial_dims=}"

        batch[self.args.mask_key] = batch[self.args.mask_key].to(self.torch_device)
        return batch

    def forward(self, x: Dict[str, Any], shared_blocks: Dict[str, SharedBlock]):
        feat = shared_blocks[self.args.encoder_key](x)

        # for Backward compatibility
        if hasattr(self.args, "squeezer_key") and self.args.squeezer_key in list(shared_blocks.keys()):
            feat[-1], _ = shared_blocks[self.args.squeezer_key](feat)

        if self.args.dropout > 0.0:
            feat = [feat[0]] + [self.task_modules["dropout"][i](f) for i, f in enumerate(feat[1:])]  # type: ignore

        # The decoder computes from the feature pyramid and the image's hidden dimension an embedding per pixel
        if self.args.decoder_type == "mtldecoder":
            decoder: MTLDecoder = shared_blocks[self.args.decoder_key]  # type: ignore
            raise Exception("MTLDecoder requires conditioning")
            decoder_output = decoder.forward(feat, z)  # type: ignore
        elif self.args.decoder_type == "pyramid":
            decoder: PyramidDecoder = cast(PyramidDecoder, shared_blocks[self.args.decoder_key])
            decoder_output = decoder.forward(feat)
        else:
            raise Exception(f"Unknown decoder type: {self.args.decoder_type}")

        # task specific computation
        if self.args.headdropout > 0.0:
            decoder_output = self.task_modules["headdropout"](decoder_output)
        masks_logits = self.task_modules["segmentation_head"](decoder_output)
        return masks_logits

    def check_batch_for_skippability(self, batch: Dict[str, Any]) -> bool:
        mask = batch[self.args.mask_key]
        if mtl_settings.ignore_class_value is not None:
            # If there is an ignore_index, the batch can be skipped if everything should be ignored
            uniques = torch.unique(mask)
            if uniques.shape[0] == 1 and (uniques[0] == mtl_settings.ignore_class_value).item():
                logging.warn(
                    f"In {self.args.module_name}, a whole batch with size {mask.shape} had only ignore index {mtl_settings.ignore_class_value}."
                )
                return True
        return False

    def training_step(self, batch: Dict[str, Any], shared_blocks: SharedModules) -> Optional[Tuple[torch.Tensor, Dict]]:
        im = batch["image"]
        mask = batch[self.args.mask_key]
        # Skip the batch, if there is only the ignore index in the mask
        if self.check_batch_for_skippability(batch):
            return None

        metas = batch["meta"] if "meta" in batch else [{} for _ in range(im.shape[0])]

        y_hat = shared_blocks.forward(im, self.forward)

        if y_hat.shape != mask.shape:
            mask = F.resize(mask, y_hat.shape[-2:], interpolation=InterpolationMode.NEAREST)

        realtime_log = {}

        losses = self.crit(y_hat, mask)
        dice_loss = losses["dice"]
        focal_loss = losses["focal"]
        realtime_log.update({"dice": dice_loss.item(), "focal": focal_loss.item()})
        batch_loss = self.normalize_loss(dice_loss, focal_loss)

        # Keep on GPU because metrics can be computed on GPU
        im_detached, mask_detached, y_hat_detached = (
            im.detach(),
            mask.detach(),
            y_hat.detach(),
        )
        # y_hat_detached = torch.argmax(y_hat_detached.sigmoid(), dim=1)
        # y_hat_preds = self.probas_to_preds(
        #     out_probas=self.logits_to_probas(y_hat_detached),
        #     confidence_threshold=self.args.confidence_threshold if mtl_settings.ignore_class_value is not None else None,
        #     uncertainty_class=mtl_settings.ignore_class_value,
        # )
        y_hat_probas = self.logits_to_probas(y_hat_detached)
        y_hat_preds = self.probas_to_preds(out_probas=y_hat_probas)
        step_metrics = self._collect_stats_for_metrics(
            mask_detached,
            y_hat_preds,
            batch.get("original_shape", [im.shape[-2:] for _ in range(im.shape[0])]),
        )

        self.add_step_result(batch_loss.item(), step_metrics)

        realtime_log.update(self._visualize_preds(im_detached, mask_detached, y_hat_preds, step_metrics, metas))

        return batch_loss, realtime_log

    def normalize_loss(self, dice_loss: torch.Tensor, focal_loss: torch.Tensor) -> torch.Tensor:
        # For now, normalization like normal cross entropy seems to be sufficient
        return dice_loss + (focal_loss / np.log(len(self.class_names)))

    @torch.no_grad()
    def _collect_stats_for_metrics(self, mask_detached, y_hat_preds, for_shape):
        tp, fp, fn, tn = get_batch_metrics(
            y_hat_preds.long(),
            mask_detached.long(),
            for_shape,
            self.class_names,
            ignore_index=mtl_settings.ignore_class_value,
        )
        step_metrics: StepMetricDict = {
            "tp": tp.numpy(),
            "fp": fp.numpy(),
            "fn": fn.numpy(),
            "tn": tn.numpy(),
        }  # type: ignore
        return step_metrics

    @torch.no_grad()
    def _visualize_pred(self, img, mask, pred, im_iou, meta: Dict) -> wandb.Image:
        if pred.shape[-2:] != img.shape[-2:]:
            # Add channel dim, resize, remove channel dim:
            mask = F.resize(
                mask.unsqueeze(0),
                img.shape[-2:],
                interpolation=InterpolationMode.NEAREST,
            ).squeeze()
            pred = F.resize(
                pred.unsqueeze(0),
                img.shape[-2:],
                interpolation=InterpolationMode.NEAREST,
            ).squeeze()

        metastr = json.dumps(meta, default=str)
        classes = {i + 1: self.class_names[i] for i in range(len(self.class_names))}
        assert mtl_settings.ignore_class_value not in classes, "Unexpected ignore class value in class names!"
        classes.update({mtl_settings.ignore_class_value + 1: "ignored"})
        res = wandb.Image(
            img,
            masks={
                "predictions": {
                    "mask_data": pred.numpy() + 1,
                    "class_labels": classes,
                },
                "ground_truth": {
                    "mask_data": mask.numpy() + 1,
                    "class_labels": classes,
                },
            },
            caption=f"{self.class_names}\nIOU: {im_iou.numpy()}\n[{torch.min(img):.2f}, {torch.max(img):.2f}]\n{img.shape}\n{metastr}]",
        )
        return res

    @torch.no_grad()
    def _visualize_preds(self, train_ims, train_masks, preds, step_metrics: Dict, metas: List[Dict]) -> Dict:
        res = {}

        vis_n = min(self._takeout_vis_budget(), train_ims.size(0))
        if vis_n >= 1:
            batch_size = train_ims.size(0)
            wandb_ims = []
            for rand_index in random.sample(list(range(batch_size)), vis_n):
                im, mask, pred = (
                    train_ims[rand_index].cpu(),
                    train_masks[rand_index].cpu(),
                    preds[rand_index].cpu(),
                )
                im_iou = smp_metrics.iou_score(
                    torch.LongTensor(step_metrics["tp"][rand_index]),
                    torch.LongTensor(step_metrics["fp"][rand_index]),
                    torch.LongTensor(step_metrics["fn"][rand_index]),
                    torch.LongTensor(step_metrics["tn"][rand_index]),
                )
                wandb_ims.append(self._visualize_pred(im, mask, pred, im_iou, metas[rand_index]))

            res["preds"] = wandb_ims
        return res

    def log_epoch_metrics(self) -> Tuple[Dict[str, Any], str]:
        flat_metrics = flatten_list_of_dicts(self._step_metrics)

        task_log_dict = {}

        by_class_ious = smp_metrics.iou_score(
            torch.LongTensor(flat_metrics["tp"]),
            torch.LongTensor(flat_metrics["fp"]),
            torch.LongTensor(flat_metrics["fn"]),
            torch.LongTensor(flat_metrics["tn"]),
        ).mean(
            dim=0
        )  # iou_score returns iou of all images by all classes [IMAGES, CLASSES]
        if len(self.class_names) < mtl_settings.max_classes_detailed_logging:
            for i, class_name in enumerate(self.class_names):
                # if mtl_settings.ignore_class_value is not None and i == mtl_settings.ignore_class_value:
                #     continue
                # class_idx = (
                #     i - 1 if mtl_settings.ignore_class_value is not None and i > mtl_settings.ignore_class_value else i
                # )
                task_log_dict[f"{class_name}iou"] = by_class_ious[i]
        # if mtl_settings.ignore_class_value is not None:
        #     by_class_ious = np.delete(by_class_ious, mtl_settings.ignore_class_value, None)
        task_log_dict["meanclassiou"] = by_class_ious.mean()

        _, print_str = super().log_epoch_metrics()
        return (
            task_log_dict,
            f"{print_str} - mean-class-iou {task_log_dict['meanclassiou']}",
        )

    @torch.no_grad()
    def get_logits_for_inputs(
        self,
        input_batch: torch.Tensor,
        shared_blocks: Dict[str, SharedBlock],
    ) -> torch.Tensor:
        """
        Returns multiclass probabilities for each pixel in the input image using softmax.
        """
        input_batch = input_batch.to(shared_blocks[self.args.decoder_key].torch_device)
        logits = self.forward(input_batch, shared_blocks)  # type: ignore
        return logits

    @torch.no_grad()
    def logits_to_probas(self, logits: torch.Tensor, output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Converts logits to probabilities using softmax.

        If output_size is set, the probabilities will be resized to that size.

        It is not merged with get_logits_for_inputs because then the logits can be computed using ONNX.
        """
        probas = logits.softmax(dim=1)
        if output_size is not None:
            probas = F.resize(probas, list(output_size), interpolation=InterpolationMode.NEAREST_EXACT)
        return probas

    @torch.no_grad()
    def probas_to_preds(self, out_probas: torch.Tensor, pixel_threshold: float | None = None):
        """
        Computes predictions for softmax probabilities, e.g. computed using SemSegTask.static_inference.

        If confidence_threshold is set, all pixels that do not have a certain prediction above that threshold
        will be set to uncertainty_class.
        """
        if pixel_threshold is None:
            if self.args.confidence_threshold is None:
                pixel_threshold = None
            else:
                pixel_threshold = self.args.confidence_threshold

        max_values, prediction = torch.max(out_probas, dim=1)
        if pixel_threshold is not None:
            prediction[max_values < pixel_threshold] = mtl_settings.ignore_class_value
        return prediction


class MultiLabelSemSegTask(SemSegTask):
    """
    At the ignore index (value in this case), the training process will assume that the area is unlabeled.
    Both metrics and loss will ignore those pixels.
    """

    class Config(SemSegTask.Config):
        mask_key: str = "masks"

    def build_loss(self):
        return SMPDiceFocalLoss2D(
            self.args.crit,
            mode="multilabel",
            ignore_index=mtl_settings.ignore_class_value,
        )

    def check_batch_for_skippability(self, batch: Dict[str, Any]) -> bool:
        mask = batch[self.args.mask_key]
        if mtl_settings.ignore_class_value is not None:
            # If there is an ignore_index, the batch can be skipped if the whole ignore dimension is set to 1
            if torch.all(mask[...] == mtl_settings.ignore_class_value):
                logging.warn(
                    f"In {self.args.module_name}, a whole batch with size {mask.shape} had only ignore value {mtl_settings.ignore_class_value}."
                )
                return True
        return False

    @torch.no_grad()
    def _collect_stats_for_metrics(self, mask_detached, y_hat_detached, for_shape):
        tp, fp, fn, tn = get_batch_metrics(
            y_hat_detached.long(),
            mask_detached.long(),
            for_shape,
            self.class_names,
            ignore_index=mtl_settings.ignore_class_value,
            mode="multilabel",
        )
        step_metrics: StepMetricDict = {
            "tp": tp.cpu().numpy(),
            "fp": fp.cpu().numpy(),
            "fn": fn.cpu().numpy(),
            "tn": tn.cpu().numpy(),
        }  # type: ignore
        return step_metrics

    @torch.no_grad()
    def logits_to_probas(self, logits: torch.Tensor, output_size: Tuple[int, int] | None = None) -> torch.Tensor:
        probas = logits.sigmoid()
        if output_size is not None:
            probas = F.resize(probas, list(output_size), interpolation=InterpolationMode.NEAREST)
        return probas

    @torch.no_grad()
    def probas_to_preds(self, out_probas: torch.Tensor, overwrite_threshold: float | None = None):
        """
        Computes predictions for sigmoid probabilities, e.g. computed using SemSegTask.static_inference.

        If confidence_threshold is set, all pixels that do not have a certain prediction above that threshold
        will be set to uncertainty_class.
        Confidence threshold will also be used for the value where a class is considered true.
        """
        prediction = torch.zeros_like(out_probas)

        threshold = self.args.confidence_threshold if overwrite_threshold is None else overwrite_threshold

        prediction[out_probas > threshold] = 1
        return prediction

    @torch.no_grad()
    def _visualize_pred(self, img, mask, pred, im_iou, meta: Dict) -> wandb.Image:
        if pred.shape[-2:] != img.shape[-2:]:
            # Add channel dim, resize, remove channel dim:
            mask = F.resize(
                mask.unsqueeze(0),
                img.shape[-2:],
                interpolation=InterpolationMode.NEAREST,
            ).squeeze()
            pred = F.resize(
                pred.unsqueeze(0),
                img.shape[-2:],
                interpolation=InterpolationMode.NEAREST,
            ).squeeze()

        metastr = json.dumps(meta, default=str)
        masks = {}
        undefined_display_value = len(self.class_names) + 1

        if len(self.class_names) <= mtl_settings.max_classes_detailed_logging:
            for i, class_name in enumerate(self.class_names):
                # predictions
                pred_class = pred[i].clone()
                pred_class[pred_class == 1] = pred_class[pred_class == 1] * (i + 1)
                if mtl_settings.ignore_class_value is not None:
                    pred_class[pred_class == mtl_settings.ignore_class_value] = undefined_display_value
                # ground truth
                mask_class = mask[i].clone()
                mask_class[mask_class == 1] = mask_class[mask_class == 1] * (i + 1)
                if mtl_settings.ignore_class_value is not None:
                    mask_class[mask_class == mtl_settings.ignore_class_value] = undefined_display_value

                masks[f"predictions_{class_name}"] = {
                    "mask_data": pred_class.numpy(),
                    "class_labels": {
                        undefined_display_value: "ignored",
                        i + 1: class_name,
                    },
                }
                masks[f"ground_truth_{class_name}"] = {
                    "mask_data": mask_class.numpy(),
                    "class_labels": {
                        undefined_display_value: "ignored",
                        i + 1: class_name,
                    },
                }
        res = wandb.Image(
            img,
            masks=masks,
            caption=f"{self.class_names}\nIOU: {im_iou.numpy()}\n[{torch.min(img):.2f}, {torch.max(img):.2f}]\n{img.shape}\n{metastr}]",
        )
        return res

    def normalize_loss(self, dice_loss: torch.Tensor, focal_loss: torch.Tensor) -> torch.Tensor:
        return dice_loss + (focal_loss * 1.44269)  # log2(exp(1))

    # def log_epoch_metrics(self) -> Tuple[Dict[str, Any], str]:
    #     return {}, f"{self.get_name()} - loss: {self.get_short_status()}"
