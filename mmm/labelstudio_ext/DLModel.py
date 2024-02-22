from __future__ import annotations
from pathlib import Path
import logging
import itertools
from typing_extensions import Annotated
from pydantic import Field, field_validator, model_validator
from typing import Literal
import numpy as np
from tiffslide import TiffSlide
import cv2
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms._functional_tensor import _get_gaussian_kernel2d

try:
    from fastapi import APIRouter
except ImportError:
    APIRouter = None

from mmm.labelstudio_ext.projects import LSProject
from mmm.labelstudio_ext.LSModel import LSModel, LabelStudioTask, SerializedArray
from mmm.labelstudio_ext.utils import download_image
from mmm.labelstudio_ext.utils import binary_mask_to_result
from mmm.labelstudio_ext.NativeBlocks import NativeBlocks
from mmm.transforms import UnifySizes
from mmm.BaseModel import BaseModel
from mmm.mtl_modules.tasks.SemSegTask import SemSegTask
from mmm.data_loading.geojson.utils import (
    annotations_from_mask,
    create_featurecollection,
)


def compute_uncertainty_score(probas, preds):
    # For multiclass segmentation, probas has shape [C, H, W] between 0 and one
    uncertainy = torch.stack([torch.min(probas[i], 1 - probas[i]) * 2 for i in range(len(probas))])  # each class
    mean_per_class = torch.mean(uncertainy, dim=(1, 2))
    return torch.mean(mean_per_class).item()
    # return torch.mean(uncertainy).item()


def by_number_of_blobs(probas, preds):
    def count_blobs_in_mask(mask):
        # Each mask is (H, W)
        return len(
            cv2.findContours(
                mask.cpu().numpy().astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )[0]
        )

    if len(preds.shape) == 3:
        # multi-label
        num_blobs_by_class = [count_blobs_in_mask(preds[i]) for i in range(len(preds))]
    else:
        # multi-class (H, W)
        num_blobs_by_class = [count_blobs_in_mask(preds)]
    return sum(num_blobs_by_class)


class SingleInvocation(BaseModel):
    invocation_type: Literal["single"] = "single"
    task: LabelStudioTask
    mtl_task_id: str
    return_type: Literal["predictions", "probabilities"] = "predictions"


class WSIWindows(Dataset):
    def __init__(
        self,
        coords: list[tuple[int, int]],
        window_size: int,
        slide: TiffSlide,
        level: int,
    ) -> None:
        self.coords, self.window_size, self.slide, self.level = (
            coords,
            window_size,
            slide,
            level,
        )

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index) -> torch.Tensor:
        row, col = self.coords[index]
        downsample_fac = self.slide.level_downsamples[self.level]
        row_l0, col_l0 = row * downsample_fac, col * downsample_fac
        try:
            img = self.slide.read_region(
                (col_l0, row_l0),
                self.level,
                (self.window_size, self.window_size),
                as_array=True,
            )
        except Exception as e:
            img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
            logging.error(f"Error while reading region {(row, col)}: {e}")
        return {"image": F.to_tensor(img), "coords": (row, col), "level": self.level}


class WSIInvocation(BaseModel):
    """
    GeoJSON inference for WSI.
    """

    invocation_type: Literal["wsi"] = "wsi"
    wsi_path: str
    mtl_task_id: str
    for_levels: list[int] = [0]
    out_downsample_fac: int = Field(
        default=8,
        description="Downsample factor of the output mask, greatly helps with reducing runtime and network load",
    )
    patch_size: int = 1024
    batch_size: int = 4
    filter_tiny_obj: bool = False
    obj_min_area: float = Field(default=0.0, description="Minimum area of an object in pixels in level 0")
    minimum_confidence: float = Field(
        default=0.0,
        description="Minimum pixel confidence for considering a pixel as a class",
    )
    vote_type: Literal["uniform", "gaussian"] = "gaussian"

    @field_validator("for_levels")
    def check_for_levels(cls, v):
        if len(v) == 0:
            raise ValueError("for_levels must not be empty")
        if min(v) < 0:
            raise ValueError("for_levels must be >= 0")
        if len(set(v)) != len(v):
            raise ValueError("for_levels must be unique")
        return v

    def get_patch_coords(self, height: int, width: int) -> list[tuple[int, int]]:
        min_row, min_col = -self.patch_size // 2, -self.patch_size // 2
        max_row, max_col = height - self.patch_size // 2, width - self.patch_size // 2
        return [
            (row, col)
            for (row, col) in itertools.product(
                range(min_row, max_row + 1 + self.patch_size // 2, self.patch_size),
                range(min_col, max_col + 1 + self.patch_size // 2, self.patch_size),
            )
        ]

    async def wsi_inference(self, torch_modules: NativeBlocks):
        mtl_task: SemSegTask = torch_modules[self.mtl_task_id]

        if self.wsi_path.startswith("s3://"):
            from mmm.data_loading.s3 import S3Path

            slide = TiffSlide(S3Path.from_str(self.wsi_path).download())
        else:
            slide = TiffSlide(Path(self.wsi_path))

        assert len(self.for_levels) == 1, "For now, only single-level inference is implemented"

        # The lowest level determines the granularity in the stitched output
        lowest_level: int = min(self.for_levels)

        # Patchify exhaustively at the given levels
        patch_coords = self.get_patch_coords(*slide.level_dimensions[lowest_level][::-1])

        # Stitch together the probabilities of each class
        # by voting with gaussian masks such that the patch-borders have less impact than the center
        out_HW = tuple(
            map(
                lambda x: x // self.out_downsample_fac,
                slide.level_dimensions[lowest_level][::-1],
            )
        )
        stitch_CHW: torch.Tensor = torch.zeros((len(mtl_task.class_names),) + out_HW, dtype=torch.float32).to(
            torch_modules.get_device()
        )
        if self.vote_type == "uniform":
            vote_mask = torch.ones(
                (
                    self.batch_size,
                    len(mtl_task.class_names),
                    self.patch_size // self.out_downsample_fac,
                    self.patch_size // self.out_downsample_fac,
                ),
                dtype=torch.float32,
            ).to(torch_modules.get_device())
        elif self.vote_type == "gaussian":
            sigma = self.patch_size // 16
            single_vote = _get_gaussian_kernel2d(
                (
                    self.patch_size // self.out_downsample_fac,
                    self.patch_size // self.out_downsample_fac,
                ),
                [sigma, sigma],
                torch.float32,
                stitch_CHW.device,
            )
            # Scale such that the average vote is 1
            single_vote /= single_vote.mean()

            # Visualize heatmap and save
            # import matplotlib.pyplot as plt
            # import seaborn as sns
            # sns.heatmap(single_vote.cpu().numpy())
            # plt.savefig("heatmap.png")

            vote_mask = single_vote.repeat(self.batch_size, len(mtl_task.class_names), 1, 1)
        else:
            raise NotImplementedError()

        with torch.no_grad():
            # Organize into batches
            wsi_windows = WSIWindows(patch_coords, self.patch_size, slide, lowest_level)
            logging.info(f"WSI inference: {len(wsi_windows)} patches to build a {stitch_CHW.shape} mask")
            for batch in tqdm(DataLoader(wsi_windows, batch_size=self.batch_size, num_workers=0)):
                torch_input = batch["image"].to(torch_modules.get_device())
                logits = mtl_task.forward(torch_input, torch_modules)
                # probas = mtl_task.logits_to_probas(logits=logits, output_size=torch_input.shape[-2:])
                logits_resized = F.resize(
                    logits,
                    [
                        self.patch_size // self.out_downsample_fac,
                        self.patch_size // self.out_downsample_fac,
                    ],
                    interpolation=InterpolationMode.NEAREST_EXACT,
                )
                # Apply voting mask (H, W) to the last two dimensions of logits (B, C, H, W)
                # The last batch might be smaller than the batch size
                vote = logits_resized * vote_mask[: logits_resized.shape[0]]

                # Stamp the votes onto the stitched image
                for i in range(vote.shape[0]):
                    row, col = (
                        batch["coords"][0][i].item(),
                        batch["coords"][1][i].item(),
                    )
                    row, col = (
                        row // self.out_downsample_fac,
                        col // self.out_downsample_fac,
                    )
                    # Only use the valid region of the vote
                    vote_row_offset = max(0, -row)
                    vote_col_offset = max(0, -col)
                    vote_row_end = min(vote.shape[2], stitch_CHW.shape[1] - row)
                    vote_col_end = min(vote.shape[3], stitch_CHW.shape[2] - col)
                    vote_valid = vote[i][:, vote_row_offset:vote_row_end, vote_col_offset:vote_col_end]
                    stitch_CHW[
                        :,
                        row + vote_row_offset : row + vote_row_end,
                        col + vote_col_offset : col + vote_col_end,
                    ] += vote_valid

            # Normalize the stitched image
            stitchprobas = mtl_task.logits_to_probas(stitch_CHW.unsqueeze(0))
            stitchpreds = (
                mtl_task.probas_to_preds(stitchprobas, pixel_threshold=self.minimum_confidence).squeeze(0).cpu().numpy()
            )

        regions = annotations_from_mask(
            stitchpreds,
            for_values={
                class_index: class_name for class_index, class_name in enumerate(mtl_task.class_names)
            },  # {1: "tumor", 2: "stroma", 3: "necrosis"},
            downsample_fac=slide.level_downsamples[lowest_level] * self.out_downsample_fac,
            coarse=self.filter_tiny_obj,
            min_area=self.obj_min_area,
        )
        outgeojson_dict: dict = create_featurecollection(regions)

        return outgeojson_dict


class ModelInput(BaseModel):
    """
    This model itself is not a union because FastAPI doesn't like it.
    On their main branch it is already fixed by the time of writing.
    """

    invo: Annotated[SingleInvocation | WSIInvocation, Field(discriminator="invocation_type")]


class DLModel(LSModel):
    """
    A deep learning model does not use the train route.
    """

    class Config(LSModel.Config):
        category: str = "DeepLearning"
        pixel_confidence_threshold: float = 0.5
        region_confidence_threshold: float = 0.0

    def __init__(self, cfg: Config, ls_client, prefix: str, blocks: NativeBlocks) -> None:
        self.cfg: DLModel.Config
        self.model_version = "t24"
        self.torch_modules = blocks
        self.shared_blocks: list[str] = self.torch_modules.get_sharedblock_keys()
        self.tasks: list[str] = self.torch_modules.get_task_keys()
        super().__init__(cfg, ls_client, prefix)

    @torch.no_grad()
    async def predict_for_tensor(self, mtl_task: SemSegTask, torch_input: torch.Tensor):
        logits = mtl_task.forward(torch_input, self.torch_modules)
        probas = mtl_task.logits_to_probas(logits=logits, output_size=torch_input.shape[-2:])
        preds = mtl_task.probas_to_preds(probas, pixel_threshold=self.cfg.pixel_confidence_threshold)
        return probas, preds

    @torch.no_grad()
    async def predict_for_task(self, mtl_task, ls_task: dict):
        input_image = download_image(ls_task["data"]["image"], ls_client=self.ls_client)
        torch_input = (
            UnifySizes(divisable_by=32, enforce_order=True)([{"image": F.to_tensor(input_image)}])[0]["image"]
            .unsqueeze(0)
            .to(self.torch_modules.get_device())
        )
        probas, preds = await self.predict_for_tensor(mtl_task, torch_input)
        return probas.squeeze(0), preds.squeeze(0)

    @torch.no_grad()
    async def predict_single_mcseg(self, ls_task: dict, task_id: str, label, cfg):
        mtl_task: SemSegTask = self.torch_modules[task_id]
        probas, preds = await self.predict_for_task(mtl_task, ls_task)
        all_results, all_scores = [], []
        for i in range(len(mtl_task.class_names)):
            score = probas[i][preds == i].mean().item()
            res = binary_mask_to_result(
                (preds == i).cpu().numpy(),
                mtl_task.class_names[i],
                mtl_task.get_name(),
                score=score,
            )
            if score > self.cfg.region_confidence_threshold:
                all_results.append(res)
                all_scores.append(score)

        # result_dict = {"result": , "score": sum(all_scores) / len(all_scores) if len(all_scores) > 0 else 0}
        # result_dict.update(await self.get_version())
        return all_results, all_scores

    @torch.no_grad()
    async def predict_single(self, project: LSProject, ls_task: dict, context: dict | None = None):
        good_results, prediction_scores = [], []
        for label, cfg in project.ls.parsed_label_config.items():
            matching_tasks = [taskid for taskid in self.tasks if label.startswith(taskid)]
            for taskid in matching_tasks:
                results, scores = await self.predict_single_mcseg(ls_task, taskid, label, cfg)
                good_results.extend(results)
                prediction_scores.extend(scores)
        return {
            "result": good_results,
            "model_version": self.model_version,
            "score": np.mean(prediction_scores) if len(prediction_scores) > 0 else 0,
        }

    @torch.no_grad()
    async def predict(self, project: LSProject, tasks: list[dict], context: dict | None = None):
        if len(tasks) >= 1:
            return {"results": [await self.predict_single(project, lstask, context) for lstask in tasks]}
        return {}

    async def labeling_priority(
        self,
        task_id: str,
        ls_task: LabelStudioTask,
        scoring_type: Literal["uncertainty", "by_number_of_blobs"] = "uncertainty",
    ) -> float:
        if task_id not in self.tasks:
            return f"Task {task_id} not found"
        mtl_task = self.torch_modules[task_id]
        probas, preds = await self.predict_for_task(mtl_task, ls_task.model_dump())
        # Render tensor as json
        # return {"probas": probas.cpu().numpy().tolist(), "preds": preds.cpu().numpy().tolist()}
        scorers = {
            "uncertainty": compute_uncertainty_score,
            "by_number_of_blobs": by_number_of_blobs,
        }
        return scorers[scoring_type](probas, preds)

    async def invocation(self, inps: ModelInput) -> SerializedArray | dict:
        """
        Invokes the model on a single input and returns the raw result.
        """
        invo = inps.invo
        if invo.invocation_type == "single":
            if invo.mtl_task_id not in self.tasks:
                return f"Task {invo.mtl_task_id} not found"
            mtl_task = self.torch_modules[invo.mtl_task_id]
            probas, preds = await self.predict_for_task(mtl_task, invo.ls_task.model_dump())
            if invo.return_type == "probabilities":
                return SerializedArray.from_numpy(probas.cpu().numpy())
            elif invo.return_type == "predictions":
                return SerializedArray.from_numpy(preds.cpu().numpy())
            else:
                raise NotImplementedError()
        elif invo.invocation_type == "wsi":
            geojson = await invo.wsi_inference(self.torch_modules)
            return geojson

    def get_task_ids(self) -> list[str]:
        return self.tasks

    def get_shared_block_ids(self) -> list[str]:
        return self.shared_blocks

    def get_task_info(self, task_id: str) -> dict:
        res = self.torch_modules[task_id].args.model_dump()
        for attr in ["class_names", "training_state", "torch_device"]:
            if hasattr(self.torch_modules[task_id], attr):
                res[attr] = getattr(self.torch_modules[task_id], attr)
        return res

    def build_router(self) -> APIRouter:
        res = super().build_router()
        res.get("/task_ids", tags=[self.get_tag()])(self.get_task_ids)
        res.get("/task_info/{task_id}", tags=[self.get_tag()])(self.get_task_info)
        res.get("/shared_block_ids", tags=[self.get_tag()])(self.get_shared_block_ids)
        return res

    def get_versions(self):
        return {"versions": [self.model_version]}
