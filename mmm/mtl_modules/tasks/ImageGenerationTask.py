import wandb
from typing import Tuple, Literal, Any, Dict
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid

from .MTLTask import MTLTask
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.MTLDecoder import MTLDecoder
from mmm.data_loading.MTLDataset import MTLDataset
from mmm.data_loading.TrainValCohort import TrainValCohort

from mmm.logging.type_ext import StepFeedbackDict, StepMetricDict
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules


def _build_criterion(loss_type: Literal["l1", "l2", "bce"]) -> nn.Module:
    if loss_type == "bce":
        return nn.BCELoss()
    if loss_type == "l1":
        return nn.L1Loss()
    if loss_type == "l2":
        return nn.MSELoss()
    else:
        return nn.CrossEntropyLoss()


class ImageGenerationTask(MTLTask):
    """
    Task to be used in pre-training as auxilliary task for representation learning.
    Aims at reconstructing (parts of) the (augmented) input image. Images will not be clear.
    """

    class Config(MTLTask.Config):
        encoder_key: str = "encoder"
        decoder_key: str = "miniDecoder"
        squeezer_key: str = "squeezer"
        reconstruction_loss: Literal["l1", "l2", "bce"] = "l2"
        predicted_channels: int = 3
        scale_factor: float = 30  # MSE is rather low compared to CE, but works better when generating images. To normalize it to other tasks this factor can be used. tested in 224x224 images

    def __init__(
        self,
        args: Config,
        for_decoder: MTLDecoder,
        cohort: TrainValCohort[MTLDataset],
    ) -> None:
        super().__init__(args, cohort)
        self.args: ImageGenerationTask.Config
        self.task_modules: nn.ModuleDict = nn.ModuleDict(
            {
                "head": nn.Sequential(
                    nn.Conv2d(
                        in_channels=for_decoder.get_output_dim_per_pixel(),
                        out_channels=args.predicted_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
                )
            }
        )

        self.criterion = _build_criterion(self.args.reconstruction_loss)

    def forward(self, x: Any, shared_blocks: Dict[str, SharedBlock]):
        pyr = shared_blocks[self.args.encoder_key](x)

        if hasattr(self.args, "squeezer_key") and self.args.squeezer_key in list(shared_blocks.keys()):
            pyr[-1], _ = shared_blocks[self.args.squeezer_key](pyr)

        decoder_output: torch.Tensor = shared_blocks[self.args.decoder_key](pyr)

        return self.task_modules["head"](decoder_output)

    def prepare_batch(self, batch: Dict[str, Any]) -> Any:
        batch["image"] = batch["image"].to(self.torch_device)
        batch["target"] = batch["target"].to(self.torch_device)

        return batch

    def training_step(
        self, batch: Dict[str, Any], shared_modules: SharedModules
    ) -> Tuple[torch.Tensor, StepFeedbackDict]:
        input_batch, target_batch = batch["image"], batch["target"]
        pred_batch = shared_modules.forward(input_batch, self.forward)

        # scale to prediction output size
        size = pred_batch.shape[-2:]
        target_batch = F.interpolate(target_batch, size=size, mode="bicubic")

        # for vis
        input_batch = F.interpolate(input_batch, size=size, mode="bicubic")

        loss = self.criterion(torch.sigmoid(pred_batch), torch.sigmoid(target_batch)) * self.args.scale_factor
        step_results: StepMetricDict = {}

        self.add_step_result(loss.item(), step_results)
        live_vis = self._visualize_step(
            source_img=input_batch.detach().cpu()[0],
            target_img=torch.sigmoid(target_batch).cpu()[0],
            pred_img=torch.sigmoid(pred_batch.detach().cpu())[0],
        )
        return loss, live_vis

    def _visualize_step(self, source_img, target_img, pred_img):
        vis_n = min(self._takeout_vis_budget(), source_img.size(0))
        if vis_n <= 0:
            return {}
        shape = pred_img.shape[-2:]
        target_img = F.interpolate(target_img.unsqueeze(0), size=shape).squeeze()
        pred_img = F.interpolate(pred_img.unsqueeze(0), size=shape).squeeze()
        diffimg = torch.abs(target_img - pred_img)
        strs = [
            f"Source: {np.min(source_img.numpy()):.3f}, {np.max(source_img.numpy()):.3f}, shape:{source_img.shape}\n"
            f"Target: {np.min(target_img.numpy()):.3f}, {np.max(target_img.numpy()):.3f}, shape:{target_img.shape}\n"
            f"Pred: {np.min(pred_img.numpy()):.3f}, {np.max(pred_img.numpy()):.3f}, shape:{pred_img.shape}\n"
            f"Difference: {torch.sum(diffimg)}"
        ]

        return {
            "vis-channel-split": wandb.Image(
                make_grid(
                    [
                        source_img[0].unsqueeze(0),
                        target_img[0].unsqueeze(0),
                        pred_img[0].unsqueeze(0),
                        source_img[1].unsqueeze(0),
                        target_img[1].unsqueeze(0),
                        pred_img[1].unsqueeze(0),
                        source_img[2].unsqueeze(0),
                        target_img[2].unsqueeze(0),
                        pred_img[2].unsqueeze(0),
                    ],
                    nrow=9,
                ),
                caption="\n".join(strs),
            ),
            "vis-img": wandb.Image(
                make_grid([source_img, target_img, pred_img], nrow=3),
                caption="\n".join(strs),
            ),
        }
