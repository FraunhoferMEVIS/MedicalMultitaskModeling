"""
- Extends the DLModel to provide an API compatible with Labelstudio's ML backend.
- Possible routes for Labelstudio: https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/api.py
- The setup route provides the credentials to work with the Labelstudio API
- The webhook only receives the project-id but no info which labelstudio instance called it.
  - Therefore, the Labelstudio that latest called the /setup route is used to connect to find the project.
"""

from __future__ import annotations

import asyncio
import json
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Literal

import logfire
from m3_sdk.types import CompressType

from mmm.api.api_worker import cache_instances
from mmm.api.DLModel import DLModel, _get_dataset_key
from mmm.api.functions.compression import CacheInstances
from mmm.api.utils import download_image
from mmm.settings import mtl_settings

from .models import MSubject
from .mtl_adapter import LabelingConfig

try:
    # API extras
    from celery.result import AsyncResult
    from fastapi import APIRouter, Request
    from label_studio_sdk import Client, Project
    from label_studio_sdk.client import LabelStudio

    from .celery import app
except ImportError:
    if not TYPE_CHECKING:  # makes sure that IDE is not confused
        Client, Project, APIRouter, Request, AsyncResult, app, LabelStudio = (None,) * 7
    else:
        raise  # Avoids errors in type checking tools

TRAIN_EVENTS = [
    "ANNOTATION_CREATED",
    "ANNOTATION_UPDATED",
    "ANNOTATION_DELETED",
    "START_TRAINING",
    "ANNOTATIONS_DELETED",
]
LABELSTUDIO_CREDENTIALS_KV_KEY = "m3api:labelstudio_credentials"


def delete_all_predictions(client: Client, project_id: int) -> list[int]:
    prediction_ids = [p["id"] for p in client.make_request("GET", f"/api/predictions?project={project_id}").json()]
    for pred_id in prediction_ids:
        client.make_request("DELETE", f"/api/predictions/{pred_id}")
    return prediction_ids


def _get_finetuning_id(prefix: str, project_id: int):
    """
    May not contain :, _ or other special characters
    """
    return f"{prefix}{project_id}"


class LSModel(DLModel):
    """
    Usage is documented in `model_api.ipynb` example notebook.
    """

    def __init__(self, prefix: str) -> None:
        super().__init__(prefix)
        self.router = APIRouter()  # Make sure the original routes are not duplicated
        labelstudio_tag = f"Labelstudio {self.cfg.category}"
        self.router.post("/predict", tags=[labelstudio_tag])(self._predict_handler)
        self.router.get("/health", tags=[labelstudio_tag])(self.health_handler)
        self.router.post("/webhook", tags=[labelstudio_tag])(self.webhook)
        self.router.post("/setup", tags=[labelstudio_tag])(self._setup_handler)

    def get_openapi_tags(self) -> list[dict]:
        return [
            {
                "name": f"Labelstudio {self.cfg.category}",
                "description": "Only used for Labelstudio integration.",
            },
        ]

    async def _predict_handler(self, request: Request):
        req = await request.json()
        project_id = int(req["project"].split(".")[0])
        logfire.info("Labelstudio called predict for tasks: {ids}", ids=[t["id"] for t in req["tasks"]])
        return await self.predict(req["tasks"], project_id, LabelingConfig(xml=req["label_config"]))

    async def predict(self, tasks: list[dict], project_id: int, labeling_config: LabelingConfig):
        finetuning_id = _get_finetuning_id(self.cfg.category, project_id)
        if len(tasks) >= 1 and bool(
            mtl_settings.kv.exists(f"{mtl_settings.adapter_prefix}:{finetuning_id}:latest:model")
        ):
            return await self.inference(
                [self.process_subject_for_m3(MSubject(**t), project_id) for t in tasks],
                labeling_config,
                finetuning_id,
            )

        logfire.info(
            "Model {finetuning_id} has not been trained yet, skipping prediction.",
            finetuning_id=finetuning_id,
        )
        return {}

    async def train_labelstudio(
        self,
        labeling_config: LabelingConfig,
        project: Project,
        mode: Literal["retrain", "continue"] = "retrain",
    ):
        """
        Trains the model with the dataset currently in the feature buffer under that name.
        """
        res = await self.train(self.get_dataset_key(project.id), labeling_config, mode)
        await self.post_training_labelstudio(project.id)
        return res

    def get_compress_type(self) -> CompressType:
        return CompressType.rgbimage

    def download_with_lsclient(self, url: str, project_id: int):
        try:
            return download_image(url)
        except ValueError:
            ls_client = self.get_labelstudio_client(project_id)
            r = ls_client.make_request("GET", url)
            img_bytes = BytesIO(r.content)
            return download_image(img_bytes)

    def process_subject_for_m3(self, subj: MSubject, project_id: int) -> MSubject:
        make_static = ""
        for input_key in subj.data.keys():
            if isinstance(subj.data[input_key], str) and subj.data[input_key].startswith("/data/upload/"):
                make_static = "Labelstudio upload detected"
                break
        if make_static:
            with logfire.span(
                "Making subject {subj_id} static for project {project_id} because {reason}",
                subj_id=subj.id,
                project_id=project_id,
                reason=make_static,
            ):
                subj = subj.make_static(downloader=partial(self.download_with_lsclient, project_id=project_id))
        return subj

    async def add_training_case(self, project: Project, subj: MSubject):
        """
        Datasets are represented as sets of subjects within the Redis database.
        """
        dataset_name = _get_finetuning_id(self.cfg.category, project.id)
        subj = self.process_subject_for_m3(subj, project.id)
        subject_keys = await self.add_subjects([subj], add_to_dataset=_get_dataset_key(dataset_name))

        add_instances_task: AsyncResult = cache_instances.delay(
            CacheInstances.Args(
                subject_keys=subject_keys,
                for_type=self.get_compress_type(),
                with_labels=project.parsed_label_config.keys(),
            )
        )

        logfire.info("Created task for adding {subj_id} to {dataset_name}", subj_id=subj.id, dataset_name=dataset_name)
        return add_instances_task

    async def delete_training_case(self, dataset_name: str, case_id: str):
        # Remove from the dataset
        mtl_settings.kv.srem(_get_dataset_key(dataset_name), case_id)

        # Remove the features
        all_repr_keys = mtl_settings.kv.keys(f"repr:{case_id}:*")
        mtl_settings.kv.srem(f"compressed:{self.get_compress_type().value}", case_id)

        # Remove the labels
        all_label_keys = mtl_settings.kv.keys(f"gt:{case_id}:*:{self.get_compress_type().value}")

        delete_keys = [f"{mtl_settings.subj_prefix}:{case_id}"] + all_repr_keys + all_label_keys
        logfire.info(
            "Deleting training case {case_id} with keys {delete_keys}, {hits}",
            case_id=case_id,
            delete_keys=delete_keys,
            hits=mtl_settings.kv.delete(*delete_keys),
        )

    async def get_version(self):
        return {"model_version": "defaultmodel"}

    async def _setup_handler(self, request: Request):
        req_body = await request.json()

        ls_creds = {"base_url": req_body["hostname"], "api_key": req_body["access_token"]}
        ls_client: Client = Client(url=ls_creds["base_url"], api_key=ls_creds["api_key"])
        assert ls_client.check_connection(), "Could not connect to Labelstudio with the provided credentials"

        project_id: int = int(req_body["project"].split(".")[0])

        mtl_settings.kv.hset(
            name=LABELSTUDIO_CREDENTIALS_KV_KEY,
            key=project_id,
            value=json.dumps(ls_creds),
        )

        return await self.get_version()

    def get_labelstudio_client(self, project_id: int) -> Client:
        creds = json.loads(mtl_settings.kv.hget(LABELSTUDIO_CREDENTIALS_KV_KEY, project_id))
        return Client(url=creds["base_url"], api_key=creds["api_key"])

    def get_labelstudio(self, project_id: int) -> LabelStudio:
        creds = json.loads(mtl_settings.kv.hget(LABELSTUDIO_CREDENTIALS_KV_KEY, project_id))
        return LabelStudio(base_url=creds["base_url"], api_key=creds["api_key"])

    async def post_training_labelstudio(self, project_id: int):
        pred_ids = delete_all_predictions(self.get_labelstudio_client(project_id), project_id)
        logfire.info("Deleting outdated predictions for {project}: {pred_ids}", project=project_id, pred_ids=pred_ids)

    def get_dataset_key(self, project_id: int) -> str:
        assert isinstance(project_id, int), "Labelstudio project_id must be an integer"
        return _get_dataset_key(_get_finetuning_id(self.cfg.category, project_id))

    async def webhook(self, request: Request):
        body = await request.json()
        logfire.info("Received Labelstudio webhook: {action}", action=body["action"], data=body)

        if body["action"] in TRAIN_EVENTS:
            logfire.info(
                "Train event {action} for: {project_id}",
                action=body["action"],
                project_id=(project_id := body["project"]["id"]),
            )
            proj = self.get_labelstudio_client(project_id).get_project(project_id)
            missing_task_ids = []

            # If this event is about a specific task, force the cache to update by deleting the task
            if body["action"] == "ANNOTATIONS_DELETED" and "annotations" in body:
                logfire.info("No info about updated tasks available from Labelstudio, rebuilding cache.")
                for task_id in proj.get_tasks_ids():
                    await self.delete_training_case(
                        _get_finetuning_id(self.cfg.category, project_id),
                        task_id,
                    )

            if "task" in body:
                missing_task_ids.append(body["task"]["id"])
            # This should iterate through all tasks and find out if any is missing from the feature buffer
            already_in_dataset = set(
                map(
                    lambda s: s.decode(),
                    mtl_settings.kv.smembers(self.get_dataset_key(project_id)),
                )
            )
            missing_task_ids.extend(set(map(str, proj.get_labeled_tasks_ids())) - already_in_dataset)

            with logfire.span(
                "Updating dataset {dataset_name} for {missing_task_ids}",
                dataset_name=self.get_dataset_key(project_id),
                missing_task_ids=missing_task_ids,
            ):
                if missing_task_ids:
                    add_tasks = await asyncio.gather(
                        *[
                            self.add_training_case(
                                project=proj,
                                subj=MSubject(**_strip_group_id_for_labelstudio(task)),
                            )
                            for task in proj.get_tasks(selected_ids=list(set(missing_task_ids)))
                        ]
                    )

                    for _ in range(100):
                        if not all([t.ready() for t in add_tasks]):
                            await asyncio.sleep(0.1)

            await self.train_labelstudio(
                LabelingConfig(xml=proj.label_config),
                proj,
                "retrain" if body["action"] == "START_TRAINING" else "continue",
            )
        return {}

    async def health_handler(self):
        """
        Returns status->UP if there is at least one worker available.
        """
        return {"status": "UP", "model_class": self.cfg.category}


def _strip_group_id_for_labelstudio(task: dict) -> dict:
    """
    For using the API with labelstudio we need to use the ID given by Labelstudio.
    But if a group_id is set, it becomes the ID of the task.
    """
    if "group_id" in task.get("meta", {}):
        task["meta"].pop("group_id")
    return task
