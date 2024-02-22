from __future__ import annotations
from typing import Literal, Any
import jinja2

import numpy as np
import torch
from torchvision.transforms._functional_tensor import _get_gaussian_kernel2d
import uuid
import cv2
import torch.nn as nn
import torch.nn.functional as nnF
import logging
from pydantic import Field

try:
    import lightgbm as lgb
    from label_studio_sdk import Project
except ImportError:
    logging.warning("api extra missing")
    lgb, Project = None, None

from mmm.labelstudio_ext.projects import LSProject
from mmm.labelstudio_ext.LSModel import LSModel, Project
from mmm.labelstudio_ext.LSDataset import LSDataset
from mmm.labelstudio_ext.utils import (
    download_image,
    convert_task_to_seglabel,
    brush_annotation_to_npy,
    binary_mask_to_result,
)
from mmm.transforms import UnifySizes
from mmm.data_loading.MultiLabelSemSegDataset import masks_from_ls
from mmm.BaseModel import BaseModel
from mmm.labelstudio_ext.NativeBlocks import NativeBlocks
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split


def compute_pixel_features_for_sklearn(feature_maps, keep_batch_dim: bool, masks: torch.Tensor, ignore_index: int = 0):
    """Computation of the dense features

    Take for example these feature maps from the encoder and decoder for one image:

    ```python
    feature_maps = torch.Tensor([
        [[0, 0], [2, 0]],
        [[1, 0], [1, 1]],
        [[2, 4], [2, 2]],
    ])
    feature_maps = torch.stack([feature_maps.clone(), (feature_maps + 3).clone()]) # (B=2, C=3, H=2, W=2)
    # or:
    feature_maps = torch.stack([
        torch.stack([torch.arange(16).view(4, 4) for i in range(3)])
        for _ in range(2)])
    masks = torch.stack([torch.arange(16).view(4, 4) for _ in range(2)])
    ```
    The first feature map is [[0, 0], [0, 0]] and the last is [[2, 2], [2, 2]].
    The first pixel should have the feature vector [0, 1, 2].
    The pixel features should have shape (8, 3) or (B=2, 4, C=3)
    """
    assert feature_maps.shape[0] == masks.shape[0], f"{feature_maps.shape=} {masks.shape=}"

    # Transpose the channel dimension to the end
    transposed = feature_maps.permute(0, 2, 3, 1)
    if keep_batch_dim:
        # Reshape into (B, H*W, C)
        flattened = transposed.contiguous().view(transposed.size(0), -1, transposed.size(3))
        flat_mask = masks.contiguous().view(masks.size(0), -1)
    else:
        # Reshape into (H*W, C)
        flattened = transposed.contiguous().view(-1, transposed.size(3))
        flat_mask = masks.contiguous().view(-1)

    # Return only the pixels that are not ignored
    return flattened[flat_mask != ignore_index], flat_mask[flat_mask != ignore_index]


class LGBBase:
    async def update_model(self, features: tuple, label, cfg, tasks):
        raise NotImplementedError

    async def predict_results(self, features, label, cfg, project: LSProject, task: dict, context: dict | None):
        raise NotImplementedError


class LGBMCSegmentor(LGBBase):
    def __init__(self, model_cfg: LGBModel.Config) -> None:
        self.class_names, self.model = None, None
        self.model_cfg = model_cfg
        self.X, self.y = None, None

    async def update_model(self, features: tuple, label, cfg, tasks):
        dense_features = features["image"][0]
        # Compute label maps from RLE representation in tasks
        self.class_names = cfg["labels"]
        full_masks = [convert_task_to_seglabel(t, ["Unsure"] + self.class_names) for t in tasks]
        # Resize all masks to be the same size as the features
        resized_masks = [
            (
                F.resize(
                    torch.Tensor(mask).unsqueeze(0),
                    feat.shape[-2:],
                    interpolation=F.InterpolationMode.NEAREST,
                )[0]
                if mask is not None
                else None
            )
            for feat, mask in zip(dense_features, full_masks)
        ]

        # Filter all pixels that have no label
        Xy = [
            compute_pixel_features_for_sklearn(
                dense_features[i],
                keep_batch_dim=False,
                masks=resized_masks[i].unsqueeze(0),
            )
            for i in range(len(tasks))
            if resized_masks[i] is not None
        ]
        X, y = torch.cat([x[0] for x in Xy]), torch.cat([x[1] for x in Xy])
        X_train, X_val, y_train, y_val = train_test_split(X.numpy(), y.numpy() - 1, test_size=0.2)

        # If only a single task is given, we add it to the training data
        if self.X is None:
            self.X, self.y = (X_train, X_val), (y_train, y_val)
        else:
            self.X = (
                np.concatenate([self.X[0], X_train]),
                np.concatenate([self.X[1], X_val]),
            )
            self.y = (
                np.concatenate([self.y[0], y_train]),
                np.concatenate([self.y[1], y_val]),
            )
            # self.X, self.y = torch.cat([self.X, X]), torch.cat([self.y, y])
        logging.info(f"Training segmentor for {label} with {self.X[0].shape=}, {self.y[0].shape=}")
        train_ds = lgb.Dataset(self.X[0], label=self.y[0])
        val_ds = lgb.Dataset(self.X[1], label=self.y[1], reference=train_ds)

        if self.model is None:
            self.model = lgb.train(
                {
                    "objective": "multiclass",
                    "num_class": len(self.class_names),
                    "force_col_wise": True,
                    "metric": {"multi_logloss"},
                },  # , "device_type": "cuda"},
                train_ds,
                # keep_training_booster=True,
                num_boost_round=100,
                valid_sets=[val_ds],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=3),
                ],
            )
        else:
            logging.info(f"{self.model=} already exists, only train on the new data")
            self.model = lgb.train(
                {
                    "objective": "multiclass",
                    "num_class": len(self.class_names),
                    "force_col_wise": True,
                    "metric": {"multi_logloss"},
                },  # , "device_type": "cuda"},
                train_ds,
                init_model=self.model,
                num_boost_round=10,
                valid_sets=[val_ds],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=3),
                ],
            )

        # self.model = lgb.train(
        #     {
        #         "objective": "multiclass",
        #         "num_class": len(self.class_names),
        #         "boosting_type": "rf",
        #         "num_trees": 100,
        #         "bagging_fraction": 0.9,  # use fraction of the data for each tree
        #         "feature_fraction": 0.9,  # use fraction of the features for each tree
        #     },
        #     lgb.Dataset(X.numpy(), label=y.numpy() - 1),
        # )
        logging.info(f"Trained segmentor {self} for {label} with {self.X[0].shape=}, {self.y[0].shape=}")
        return self

    async def predict_results(self, features, label, cfg, project: LSProject, task: dict, context: dict | None):
        if self.model is None:
            logging.warning(f"Skipping {label} because the model is not trained yet")
            return [], []

        spatial_features = features[cfg["inputs"][0]["value"]][0][0]
        img_size = features[cfg["inputs"][0]["value"]][-1][0]
        X = compute_pixel_features_for_sklearn(
            spatial_features,
            keep_batch_dim=False,
            masks=torch.ones(spatial_features.shape[0], *spatial_features.shape[-2:]),
        )[0].numpy()
        probas = torch.Tensor(self.model.predict(X))
        probas_spatial = probas.view(*spatial_features.shape[-2:], len(self.class_names))
        probas_spatial = F.resize(
            probas_spatial.permute(2, 0, 1),
            img_size,
            interpolation=F.InterpolationMode.NEAREST_EXACT,
        ).permute(1, 2, 0)
        y_hat = torch.argmax(probas_spatial, axis=-1)
        # Set those pixels to -1 that have a confidence lower than the threshold
        y_hat[probas_spatial.max(axis=-1).values < self.model_cfg.pixel_confidence_threshold] = -1

        # y_hat_original = F.resize(
        #     y_hat.unsqueeze(0),
        #     [x * 2 for x in y_hat.shape],
        #     # download_image(task["data"][cfg["inputs"][0]["value"]]).shape[:2],
        #     interpolation=F.InterpolationMode.NEAREST,
        # )[0]

        results, prediction_scores = [], []
        for i, class_name in enumerate(self.class_names):
            if self.model_cfg.split_connected_comp:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((y_hat == i).numpy().astype(np.uint8))
                areas = [s[4] for s in stats]
            else:
                # The first label is the background
                num_labels = 2 if i in y_hat else 0
                labels = (y_hat == i).numpy().astype(np.uint8)
                # count the ones in the mask
                areas = [None, torch.sum(y_hat == i).item()]

            for label_index in range(1, num_labels):
                region_mask = labels == label_index
                region_score = torch.mean(probas_spatial[region_mask][:, i]).item()
                if (
                    region_score > self.model_cfg.region_confidence_threshold
                    and areas[label_index] > self.model_cfg.region_minimum_area
                ):
                    results.append(binary_mask_to_result(region_mask, class_name, label, score=region_score))
                    prediction_scores.append(region_score)

        # prediction_scores = [torch.mean(probas_spatial[:, :, i]).item() for i in range(len(self.class_names))]
        # Only take the average across the region of that class
        # prediction_scores = [
        #     torch.mean(probas_spatial[y_hat == i][:, i]).item() if i in y_hat else 0
        #     for i in range(len(self.class_names))
        # ]

        # Scale back from inference size to original image size

        # results = [
        #     binary_mask_to_result((y_hat_original == i).numpy(), self.class_names[i], label, score=prediction_scores[i])
        #     for i in range(len(self.class_names))
        #     if i in y_hat_original
        # ]
        return results, prediction_scores


class LGBMLSegmentor(LGBBase):
    def __init__(self, model_cfg: LGBModel.Config) -> None:
        self.class_names, self.models = None, None
        self.model_cfg = model_cfg

    async def update_model(self, features: tuple, label, cfg, tasks):
        dense_features = features[cfg["inputs"][0]["value"]][0]
        # Compute label maps from RLE representation in tasks
        self.class_names = cfg["labels"]

        def get_results(task):
            return [
                result
                for annotation in task["annotations"]
                for result in annotation["result"]
                if result["type"] == "brushlabels"
            ]

        full_masks = [torch.Tensor(masks_from_ls(get_results(t), None, self.class_names)) for t in tasks]
        # Resize all masks to be the same size as the features
        resized_masks = [
            F.resize(
                mask,
                dense_features.shape[-2:],
                interpolation=F.InterpolationMode.NEAREST,
            )
            for mask in full_masks
        ]

        # For each class, train a binary classifier
        self.models = {}
        for i, class_name in enumerate(self.class_names):
            # Filter all pixels that have no label
            X, y = compute_pixel_features_for_sklearn(
                dense_features,
                keep_batch_dim=False,
                masks=torch.stack([m[i] for m in resized_masks]),
                ignore_index=-1,
            )
            logging.info(f"Training segmentor for {label} with {X.shape=}, {y.shape=}")
            self.models[class_name] = lgb.train(
                {"objective": "binary"},
                lgb.Dataset(X.numpy(), label=y.numpy()),
            )
            logging.info(f"Trained segmentor {self} for {label} with {X.shape=}, {y.shape=}")
        return self

    async def predict_results(self, features, label, cfg, project: LSProject, task: dict, context: dict | None):
        if self.models is None:
            logging.warning(f"Skipping {label} because the model is not trained yet")
            return [], []

        spatial_features = features[cfg["inputs"][0]["value"]][0]
        X = compute_pixel_features_for_sklearn(
            spatial_features,
            keep_batch_dim=False,
            masks=torch.ones(spatial_features.shape[0], *spatial_features.shape[-2:]),
        )[0].numpy()

        results, prediction_scores = [], []
        for i, class_name in enumerate(self.class_names):
            probas = torch.Tensor(self.models[class_name].predict(X))
            # H, W
            probas_spatial = probas.view(*spatial_features.shape[-2:])
            probas_spatial = F.resize(
                probas_spatial.unsqueeze(0),
                [x * 2 for x in probas_spatial.shape],
                # download_image(task["data"][cfg["inputs"][0]["value"]]).shape[:2],
                interpolation=F.InterpolationMode.NEAREST_EXACT,
            )[0]
            y_hat = probas_spatial > self.model_cfg.pixel_confidence_threshold

            # Process each connected component separately
            if self.model_cfg.split_connected_comp:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(y_hat.numpy().astype(np.uint8))
                areas = [s[4] for s in stats]
            else:
                num_labels = 2
                labels = y_hat.numpy().astype(np.uint8)
                # count the ones in the mask
                areas = [None, torch.sum(y_hat).item()]

            # num_labels, labels = cv2.connectedComponents(y_hat.numpy().astype(np.uint8))
            for label_index in range(1, num_labels):
                # left, top, width, height, area =
                # area =
                component_mask = labels == label_index
                region_score = torch.mean(probas_spatial[component_mask]).item()

                # Filter regions
                if (
                    region_score > self.model_cfg.region_confidence_threshold
                    and areas[label_index] > self.model_cfg.region_minimum_area
                ):
                    results.append(binary_mask_to_result(component_mask, class_name, label, score=region_score))
                    prediction_scores.append(region_score)
        return results, prediction_scores


class LGBModel(LSModel):
    """
    A model based on lightgbm.

    Initializes from an LSDataset for loading the annotations, and respects the train route for updating the model.
    """

    class Config(LSModel.Config):
        category: str = "LightGBM"
        # trainer_checkpoint: str = Field(description="The mtl trainer checkpoint is used for extracting features.")

        split_connected_comp: bool = Field(
            default=False,
            description="All class names that match will be processed using connected components.",
        )
        pixel_confidence_threshold: float = Field(
            default=0.5,
            description="Only pixels with a confidence greater than this threshold will be considered.",
        )
        region_confidence_threshold: float = Field(
            default=0.8,
            description="Only regions with a confidence greater than this threshold will be considered.",
        )
        region_minimum_area: int = Field(
            default=100,
            description="Only regions with an area greater than this threshold will be considered.",
        )

        neighbourhood_features: Literal["none", "avgpool", "blur"] = "avgpool"

        segmentation_mode: Literal["multiclass", "multilabel"] = "multiclass"

    def __init__(self, cfg: Config, ls_client, prefix: str, torch_modules: NativeBlocks) -> None:
        self.cfg: LGBModel.Config
        self.torch_modules = torch_modules
        self.pooling: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.version = 0
        self.env = jinja2.Environment(loader=jinja2.PackageLoader("mmm", "resources"))

        # Feature buffer for each (project, tag). Usually just one called "image".
        self._feature_buffer: dict[str, tuple] = {}

        # For each Choices label there should be a trained classifier
        self.classifiers: dict[tuple[int, str], Any] = {}
        self.segmentors: dict[tuple[int, str], Any] = {}

        super().__init__(cfg, ls_client, prefix)

    async def get_user_html(self) -> str:
        return self.env.get_template("lgbmodel_status.jinja2").render(
            {
                "prefix": self.prefix,
                "config": self.cfg,
                "model": self,
                "classifiers": self.classifiers,
                "segmentors": self.segmentors,
            }
        )

    @torch.no_grad()
    async def predict_single_clf(self, features, label, cfg, project: LSProject, task: dict, context: dict | None):
        res, scores = [], []

        model, class_names = self.classifiers[label]
        X = features[cfg["inputs"][0]["value"]][1].numpy()
        logging.info(f"Predicting {label} for {X.shape=}")
        y_hat = model.predict(X)[0]
        best_index = np.argmax(y_hat)
        res.append(
            {
                "id": str(uuid.uuid4())[0:8],
                "type": "choices",
                "value": {"choices": [class_names[best_index]]},
                "origin": "manual",
                "to_name": cfg["inputs"][0]["value"],
                "from_name": label,
                "image_rotation": 0,
                "score": y_hat[best_index],
            }
        )
        scores.append(y_hat[best_index])
        logging.info([f"{name}: {x:.3f}" for name, x in zip(class_names, y_hat)])

        return res, scores

    @torch.no_grad()
    async def predict_single(self, project: LSProject, task: dict, context: dict | None):
        goodresults = []
        prediction_scores = []
        features = {}
        for label, cfg in project.ls.parsed_label_config.items():
            for k in [inp["value"] for inp in cfg["inputs"] if inp["type"] == "Image"]:
                if k not in features:
                    features[k] = self.get_features([task], k)
            if (model_key := (project.ls.id, label)) in self.classifiers:
                results, scores = await self.predict_single_clf(features, label, cfg, project, task, context)
                goodresults.extend(results)
                prediction_scores.extend(scores)
            elif (model_key := (project.ls.id, label)) in self.segmentors:
                results, scores = await self.segmentors[model_key].predict_results(
                    features, label, cfg, project, task, context
                )
                goodresults.extend(results)
                prediction_scores.extend(scores)
            else:
                logging.warning(f"Skipping {label} of {project} because it has no classifier or segmentor")
        return {
            "result": goodresults,
            "model_version": self.version,
            "score": np.mean(prediction_scores) if len(prediction_scores) > 0 else 0,
        }

    @torch.no_grad()
    async def predict(self, project: LSProject, tasks: list[dict], context: dict | None = None):
        if len(tasks) >= 1:
            return {"results": [await self.predict_single(project, lstask, context) for lstask in tasks]}
        return {}

    async def train_classifier(self, label, cfg, tasks):
        assert len(cfg["inputs"]) == 1, "Choices with multiple inputs not supported yet"
        assert cfg["inputs"][0]["type"] == "Image", "Only image inputs supported for now"

        class_names = cfg["labels"]
        class_labels = self.get_classification_labels(tasks, label)
        X = self._clf_features["image"]
        y = np.array([class_names.index(class_labels[task_id.item()]) for task_id in self.task_ids["image"]])
        self.classifiers[label] = (
            lgb.train(
                {
                    "objective": "multiclass",
                    "min_data_in_leaf": min(y.shape[0] // len(class_names), 20),
                    "num_class": len(class_names),
                    "verbosity": -1,
                    # "num_boosting_round": 10,
                },
                lgb.Dataset(X.numpy(), label=y),
            ),
            class_names,
        )

    async def train(
        self,
        project: LSProject,
        tasks: list[int],
        mode: Literal["retrain", "continue"] = "retrain",
    ):
        logging.info(f"Training for {len(tasks)=}")

        # Add the new features to the datasets for each label
        for label, cfg in project.ls.parsed_label_config.items():
            for k in [inp["value"] for inp in cfg["inputs"] if inp["type"] == "Image"]:
                if k not in self._feature_buffer:
                    # self._dense_features[k], self._clf_features[k], self.task_ids[k] = self.get_features(tasks, k)
                    self._feature_buffer[k] = self.get_features(tasks, k)
            if cfg["type"] == "Choices":
                assert cfg["inputs"][0]["value"] == "image" and len(cfg["inputs"]) == 1, "Only image inputs supported"
                await self.train_classifier(label, cfg, tasks)
            elif cfg["type"] in ["BrushLabels"]:
                logging.info(f"Assuming region labels for {label}, training segmentor with {len(tasks)=}")
                if self.cfg.segmentation_mode == "multiclass":
                    if (project.ls.id, label) not in self.segmentors:
                        self.segmentors[(project.ls.id, label)] = await LGBMCSegmentor(self.cfg).update_model(
                            self._feature_buffer, label, cfg, tasks
                        )
                    else:
                        if mode == "retrain":
                            m = self.segmentors[(project.ls.id, label)]
                            m.model, m.X, m.y = None, None, None
                        await self.segmentors[(project.ls.id, label)].update_model(
                            self._feature_buffer, label, cfg, tasks
                        )
                elif self.cfg.segmentation_mode == "multilabel":
                    self.segmentors[(project.ls.id, label)] = await LGBMLSegmentor(self.cfg).update_model(
                        self._feature_buffer, label, cfg, tasks
                    )
                else:
                    raise ValueError(f"{self.cfg.segmentation_mode=} not supported")
            elif cfg["type"] in ["Brush"]:
                logging.debug(f"Skipping {cfg['type']} during training because its a tool, not a label")
            else:
                logging.error(f"{cfg['type']} not supported")
        self.version += 1

        # Release the features cache
        self._feature_buffer = {}

        # Delete all old predictions using the API if the model was retrained
        if mode == "retrain":
            pred_ids = await project.delele_all_predictions()
            logging.info(f"Deleting all predictions for {project}: {pred_ids=}")

        return {}

    def get_classification_labels(self, tasks: list[dict], label: str):
        tasks_with_label: dict[int, str] = {}
        for task in tasks:
            for result in task["annotations"][0]["result"]:
                if result["from_name"] == label:
                    tasks_with_label[task["id"]] = result["value"]["choices"][0]
        return tasks_with_label

    async def get_version(self):
        return {"model_version": f"{self.version}"}

    async def get_versions(self):
        return {"versions": [f"{self.version}"]}

    def _build_blurrer(self, like_tensor: torch.Tensor, kernel_size: int = 15, sigma: float = 2.5) -> nn.Module:
        kernel = _get_gaussian_kernel2d(
            (kernel_size, kernel_size),
            [sigma, sigma],
            like_tensor.dtype,
            like_tensor.device,
        ).repeat(like_tensor.shape[1], 1, 1, 1)
        blurrer = nn.Conv2d(
            like_tensor.shape[1],
            like_tensor.shape[1],
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            padding_mode="replicate",
            groups=like_tensor.shape[1],
            bias=False,
        )
        blurrer.weight.data = kernel
        return blurrer

    @torch.inference_mode()
    def get_features_batch(self, torch_inputs: list[torch.Tensor]):
        """
        All torch_inputs should have the same shape like.
        """
        input_batch = torch.stack(torch_inputs, dim=0).to(self.torch_modules.get_device())

        feature_pyramid = self.torch_modules["encoder"].forward(input_batch)
        squeezed_map = self.torch_modules["squeezer"].forward(feature_pyramid)
        squeezed = self.flatten(self.pooling(squeezed_map))

        # Interpolate all feature maps from the feature_pyramid to the size of the pixel features
        pixel_features = [self.torch_modules["decoder"].forward(feature_pyramid)]
        # Pixel features shape: (batch_size, channels, height, width)

        interpolated = [
            F.resize(
                feature_map,
                size=pixel_features[0].shape[-2:],
                interpolation=F.InterpolationMode.NEAREST,
            )
            for feature_map in feature_pyramid[1:2]
        ]
        pixel_features.extend(interpolated)

        # Neighbourhood features enable the decision tree to also use the neighbourhood of a pixel
        if self.cfg.neighbourhood_features == "avgpool":
            pixel_features.append(
                nnF.avg_pool2d(
                    nnF.pad(pixel_features[0], (3, 3, 3, 3), mode="reflect"),
                    kernel_size=7,
                    stride=1,
                )
            )
        elif self.cfg.neighbourhood_features == "blur":
            blurrer = self._build_blurrer(pixel_features[0])
            pixel_features.append(blurrer(pixel_features[0]))
        elif self.cfg.neighbourhood_features == "none":
            logging.debug(f"Skipping neighbourhood features, because {self.cfg.neighbourhood_features=}")
        else:
            raise ValueError(f"{self.cfg.neighbourhood_features=} not supported")
        return torch.cat(pixel_features, dim=1).cpu(), squeezed.cpu()

    @torch.inference_mode()
    def get_features(self, tasks: dict, for_image: str):
        """
        For a batch of tasks, returns a tuple with

        - Dense features (num_pixels, #Channels): from the output of the decoder concatenated with the feature maps
        - Image features: (num_images, #Features): avg pooled from the output of the squeezer
        """
        input_images = [download_image(task["data"][for_image], ls_client=self.ls_client) for task in tasks]
        dimensions = [image.shape[:2] for image in input_images]
        torch_inputs = [F.to_tensor(input_image) for input_image in input_images]
        torch_inputs = [
            x["image"] for x in UnifySizes(divisable_by=32, enforce_order=True)([{"image": i} for i in torch_inputs])
        ]

        # Build batches, for now just process one image at a time
        batch_features = [self.get_features_batch([x]) for x in torch_inputs]

        return (
            [x[0] for x in batch_features],
            [x[1] for x in batch_features],
            torch.LongTensor([task["id"] for task in tasks]),
            dimensions,
        )

    async def get_instruction(self, project: Project, previous_instruction: str) -> tuple:
        def f_key(instruction):
            return "iframe" in instruction

        def f_instr(project, instruction):
            return self.env.get_template("lgbmodel_expertinstruction.jinja2").render({"modelurl": self.prefix})

        return f_key, f_instr
