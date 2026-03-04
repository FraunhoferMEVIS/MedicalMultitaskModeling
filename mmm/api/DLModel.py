from __future__ import annotations

import asyncio
import json
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import aiohttp

from mmm.api.functions.compression import CacheInstances

try:
    from celery.result import AsyncResult
    from fastapi import APIRouter, BackgroundTasks, HTTPException

    from .celery import app
except ImportError:
    if not TYPE_CHECKING:  # makes sure that IDE is not confused
        APIRouter, AsyncResult, app, BackgroundTasks, HTTPException = None, None, None, None, None
    else:
        raise  # Avoids errors in type checking tools


import logfire
from m3_sdk.models import Webhook, WebhookResponse
from m3_sdk.types import CompressType

from mmm.api.api_worker import (
    CacheSubjects,
    cache_instances,
    cache_subjects,
    finetune,
    get_model_info,
    get_task_ids,
    predict,
)
from mmm.api.data import KVReprCohort, ReprDataset
from mmm.api.evaluation import TaskDefinition
from mmm.api.Finetuner import FineTuner
from mmm.api.functions.deeplearning import Finetune, Predict
from mmm.api.models import MSubject
from mmm.api.mtl_adapter import LabelingConfig
from mmm.BaseModel import BaseModel
from mmm.event_selectors import FixedEventSelector
from mmm.settings import mtl_settings


def _get_dataset_key(dataset_name: str):
    return f"datasets:{dataset_name}"


class PostDatasetBody(BaseModel):
    subject_ids: list[str]


class DLModel:
    class Config(BaseModel):
        category: str = "M3_PEFT"
        pixel_confidence_threshold: float = 0.5

        finetune_config: FineTuner.Config = FineTuner.Config(
            mtl_validation_selector=FixedEventSelector(at_iterations=[])  # never validate
        )

    def __init__(self, prefix: str, settings_kv_key="m3api:settings") -> None:
        """
        The prefix indicates the relative path of the model in the API such as /deeplearning_model.
        """
        self.prefix, self.settings_kv_key = prefix, settings_kv_key
        self.tag = "PEFT"

        self.router = APIRouter()

        self.dataset_category = f"Dataset Management"
        self.router.get("/dataset", tags=[self.dataset_category])(self.get_dataset)
        self.router.post("/dataset", tags=[self.dataset_category])(self.post_dataset)
        self.router.delete("/dataset", tags=[self.dataset_category])(self.delete_dataset)
        self.router.get("/datasets", tags=[self.dataset_category])(self.list_datasets)

        self.router.get("/job_state", tags=[self.cfg.category])(self.get_job_state)
        self.router.get("/info", tags=[self.cfg.category])(self.info)
        self.router.get("/compress", tags=[self.cfg.category])(self.get_representations)

        # Deprecated
        self.router.post("/add_subjects", tags=[f"{self.cfg.category} Deprecated"])(self.add_subjects)
        self.router.post("/train", tags=[f"{self.cfg.category} Deprecated"])(self.train)
        self.router.post("/cache_instances", tags=[f"{self.cfg.category} Functionals"])(self.api_cache_instances)
        self.router.post("/cache_subjects", tags=[f"{self.cfg.category} Functionals"])(self.cache_subjects)

        # Functional
        self.router.post("/subjects", tags=[f"{self.cfg.category} Functionals"])(self.cache_subjects)
        self.router.post("/finetune", tags=[f"{self.cfg.category} Functionals"])(self.finetune)
        self.router.post("/predict", tags=[f"{self.cfg.category} Functionals"])(self.api_predict)
        self.router.post("/preprocess", tags=[f"{self.cfg.category} Functionals"])(self.api_cache_instances)

        # Settings
        settings_tag = f"Settings {self.cfg.category}"
        self.router.get("/settingsschema", tags=[settings_tag])(self.get_settings_schema)
        self.router.get("/settings", tags=[settings_tag])(self.get_current_settings)
        self.router.post("/settings", tags=[settings_tag])(self.set_settings)

    def get_openapi_tags(self) -> list[dict]:
        return [
            {"name": self.cfg.category, "description": "Endpoints for PEFT model management and inference."},
            {
                "name": f"{self.cfg.category} Functionals",
                "description": "Recommended, programmatically created endpoints.",
            },
            {
                "name": self.dataset_category,
                "description": "Endpoints for managing groups of subjects to use for training or inference.",
            },
        ]

    @property
    def cfg(self) -> Config:
        if not mtl_settings.kv.exists(self.settings_kv_key):
            logfire.info(
                "No settings found at {settings_key}. Creating default settings.", settings_key=self.settings_kv_key
            )
            mtl_settings.kv.set(self.settings_kv_key, DLModel.Config().model_dump_json())
        return self.Config.model_validate_json(mtl_settings.kv.get(self.settings_kv_key).decode())

    async def get_settings_schema(self) -> dict:
        return self.cfg.model_json_schema()

    async def get_current_settings(self) -> DLModel.Config:
        return self.cfg

    async def set_settings(self, body: dict) -> DLModel.Config:
        mtl_settings.kv.set(self.settings_kv_key, self.Config.model_validate(body).model_dump_json())
        return await self.get_current_settings()

    async def get_job_state(self, task_id: str) -> Literal["PENDING", "SUCCESS", "FAILURE"]:
        """
        Returns the state of a Celery task. The API user needs to keep track of the task_id returned by
        other endpoints such as /finetune.
        """
        return AsyncResult(task_id, app=app).state

    def get_task_definition(self, labeling_config: LabelingConfig, dataset_key: str):
        task_definition = TaskDefinition(labeling=labeling_config, splits=[])

        num_subjects_in_dataset = mtl_settings.kv.scard(dataset_key)
        if num_subjects_in_dataset < (min_num := task_definition.get_minimum_cohort_size()):
            logfire.info(
                "{dataset_key} has {num_subjects_in_dataset} subjects, but needs {min_num} to train.",
                dataset_key=dataset_key,
                num_subjects_in_dataset=num_subjects_in_dataset,
                min_num=min_num,
            )
            return None
        return task_definition

    async def add_subjects(self, subjects: list[MSubject], add_to_dataset: None | str = None):
        r: CacheSubjects.Results = cache_subjects.delay(CacheSubjects.Args(subjects=subjects)).get()
        if add_to_dataset is not None:
            # Labelstudio overwrites the id, so use the original id which is written by Pydantic into the meta field
            subject_ids = [
                subj.id if subj.meta is None else subj.meta.get("original_subject_id", subj.id) for subj in subjects
            ]
            mtl_settings.kv.sadd(add_to_dataset, *subject_ids)
        return r.subject_keys

    async def compress_subjects(
        self,
        subject_keys: list[str],
        compress_type: CompressType = CompressType.token,
        with_labels: None | list[str] = None,
        subject_groups: int = 1,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> list[str]:
        subject_batches = [
            subject_keys[subject_batch : subject_batch + subject_groups]
            for subject_batch in range(0, len(subject_keys), subject_groups)
        ]
        add_instances_tasks: AsyncResult = [
            cache_instances.delay(
                CacheInstances.Args(
                    for_type=compress_type,
                    subject_keys=ss,
                    with_labels=with_labels,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            )
            for ss in subject_batches
        ]
        return [t.id for t in add_instances_tasks]

    async def cache_subjects(self, args: CacheSubjects.Args, return_result: bool = True) -> CacheSubjects.Results | str:
        task: AsyncResult = cache_subjects.delay(args)  # pyright: ignore[reportFunctionMemberAccess]
        return task.get() if return_result else task.id

    async def api_cache_instances(
        self, args: CacheInstances.Args, return_result: bool = True
    ) -> CacheInstances.Results | str:
        task: AsyncResult = cache_instances.delay(args)  # pyright: ignore[reportFunctionMemberAccess]
        return task.get() if return_result else task.id

    async def finetune(
        self,
        args: Finetune.Args,
        background_tasks: BackgroundTasks,
        return_result: bool = True,
        # webhook: Webhook | None = None,
    ) -> Finetune.Results | str:
        task: AsyncResult = finetune.delay(args)  # pyright: ignore[reportFunctionMemberAccess]
        # if webhook is not None:
        #     background_tasks.add_task(
        #         self.forward_events_to_webhook,
        #         task.id,
        #         webhook,
        #         eventqueue=args.eventqueue,
        #     )
        return task.get() if return_result else task.id

    async def api_predict(self, args: Predict.Args, return_result: bool = True) -> Predict.Results | str:
        task: AsyncResult = predict.delay(args)  # pyright: ignore[reportFunctionMemberAccess]
        return task.get() if return_result else task.id

    async def forward_events_to_webhook(
        self,
        terminate_with_celery_task_id: str,
        webhook: Webhook,
        eventqueue: str,
    ):
        p = mtl_settings.kv.pubsub()
        p.subscribe(eventqueue)
        try:
            async with aiohttp.ClientSession() as session:
                while True:
                    await asyncio.sleep(0.1)
                    # Drain all available messages per iteration
                    while (msg := p.get_message(timeout=0.1)) is not None:
                        if msg["type"] == "message":
                            await self._send_to_webhook(session, webhook, msg)
                    # Check task completion after draining
                    if AsyncResult(terminate_with_celery_task_id, app=app).ready():
                        # Final drain with longer patience for late messages
                        await asyncio.sleep(1)
                        while (msg := p.get_message(timeout=1.0)) is not None:
                            if msg["type"] == "message":
                                await self._send_to_webhook(session, webhook, msg)
                        break
        finally:
            p.unsubscribe(eventqueue)
            p.close()

    async def _send_to_webhook(self, session: aiohttp.ClientSession, webhook: Webhook, message) -> None:
        data = WebhookResponse(
            return_id=webhook.return_id,
            event=json.loads(message["data"].decode()),
        )
        if data.event.event_type not in webhook.events:
            return
        try:
            await session.post(
                webhook.url,
                json=data.model_dump(exclude_none=True),
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            logfire.error("Error sending webhook: {error}", error=str(e))

    async def train(
        self,
        dataset_name: str,
        labeling_config: LabelingConfig,
        background_tasks: BackgroundTasks,
        mode: Literal["retrain"] | Literal["continue"] = "continue",
        loops: int = 1,
        from_compress_type: CompressType = CompressType.rgbimage,
        finetune_config: FineTuner.Config | None = None,
        wandb_args: dict | None = None,
        webhook: Webhook | None = None,
        finetuning_id: str | None = None,
        batch_size: int = 8,
        num_workers: int = 1,
    ):
        if (task_definition := self.get_task_definition(labeling_config, dataset_name)) is None:
            return {}

        cohorts = []
        for label, cfg in labeling_config.get_parsed().items():
            assert len(cfg["to_name"]) == 1
            label_cohort = KVReprCohort.Config(
                batch_size=(batch_size, batch_size),
                num_workers=num_workers,
                labeling_config=labeling_config,
                train_dataset=dataset_name,
                validation_dataset=dataset_name,
                for_data=ReprDataset.Config(for_input=cfg["to_name"][0], for_label=label),
                length=1000,
                compress_type=from_compress_type,
            )
            cohorts.append(label_cohort)

        finetune_config = deepcopy(self.cfg.finetune_config) if finetune_config is None else finetune_config
        if mode == "retrain":
            finetune_config.load_prefix = ""

        task: AsyncResult = finetune.delay(
            Finetune.Args(
                cohorts=cohorts,
                finetuning_id=dataset_name.split(":")[-1] if finetuning_id is None else finetuning_id,
                cfg=finetune_config,
                num_loops=loops,
                wandb_args=wandb_args,
                eventqueue=(training_event_queue := f"training_{uuid.uuid4()}"),
            )
        )
        if webhook is not None:
            background_tasks.add_task(
                self.forward_events_to_webhook,
                task.id,
                webhook,
                eventqueue=training_event_queue,
            )
        return task.id

    async def info(self):
        return {"task_ids": get_task_ids.delay().get(), "model": get_model_info.delay().get()}

    async def get_representations(
        self,
        pattern: str = f"repr:*:{CompressType.token.value}",
        dataset: str | None = None,
        return_values: Literal["only_number", "all_contexts"] = "all_contexts",
    ) -> dict[str, list[str] | int]:
        """
        - First, finds representations using the pattern.
        - Then, if a dataset is given, filters only subjects within that dataset which consists.
          - The dataset should contain subject ids like "Tom", not keys like "subject:Tom".
        - Depending on return_values, returns either the number of contexts or all contexts.

        For example, if a subject a has a representation for the context "(0,)" and the context "(1,)", the return value
        would be {"a": ["(0,)", "(1,)"]}. All returned values are valid Python programs and can be parsed using eval.
        """
        repr_keys = mtl_settings.kv.keys(pattern)
        if dataset is not None:
            repr_keys = [
                key
                for key in repr_keys
                if key.decode().split(":")[1]
                in list(map(lambda x: x.decode(), mtl_settings.kv.smembers(_get_dataset_key(dataset))))
            ]
        with mtl_settings.kv.pipeline() as pipe:
            for subj_key in repr_keys:
                pipe.hkeys(subj_key)
            all_keys = pipe.execute()
        res = {
            repr_key.decode().split(":")[1]: [key.decode() for key in keys]
            for repr_key, keys in zip(repr_keys, all_keys)
        }
        if return_values == "only_number":
            res = {repr_key: len(keys) for repr_key, keys in res.items()}
            return res
        elif return_values == "all_contexts":
            return res
        else:
            raise ValueError(f"Unknown return value {return_values}")

    async def get_dataset(self, dataset_key: str):
        subject_ids = mtl_settings.kv.smembers(dataset_key)
        return {
            "subject_ids": subject_ids,
            "subject_prefix": mtl_settings.subj_prefix,
        }

    async def post_dataset(self, body: PostDatasetBody, dataset_key: str):
        with mtl_settings.kv.pipeline() as pipe:
            # Check if all subject ids exist
            for subject_id in body.subject_ids:
                pipe.exists(f"{mtl_settings.subj_prefix}:{subject_id}")
            exists = pipe.execute()
            if not all(exists):
                missing_subjects = [subject_id for subject_id, exist in zip(body.subject_ids, exists) if not exist]
                msg = f"Subjects with ids {missing_subjects} do not exist and cannot be added to dataset."
                raise HTTPException(status_code=400, detail=msg)
        mtl_settings.kv.sadd(dataset_key, *body.subject_ids)
        return {"dataset_key": dataset_key, "num_subjects": mtl_settings.kv.scard(dataset_key)}

    async def delete_dataset(self, dataset_key: str):
        mtl_settings.kv.delete(dataset_key)
        return {"dataset_key": dataset_key}

    async def list_datasets(self):
        return mtl_settings.kv.keys(_get_dataset_key("*"))


DLModel.cache_subjects.__doc__ = CacheSubjects.__doc__
DLModel.api_cache_instances.__doc__ = CacheInstances.__doc__
DLModel.finetune.__doc__ = Finetune.__doc__
DLModel.api_predict.__doc__ = Predict.__doc__
