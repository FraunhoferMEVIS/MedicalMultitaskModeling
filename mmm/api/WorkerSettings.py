import os

from m3_sdk.DistributedPath import DistributedPath
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mmm.api.M3Model import DEFAULT_MODEL, M3_MODELS
from mmm.settings import mtl_settings


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MMMWORKER_")

    modules_path: DistributedPath | str = M3_MODELS[DEFAULT_MODEL]
    redis_url: str = Field(
        default_factory=lambda: os.getenv("CELERY_BROKER_URL", default=mtl_settings.redis_url),
    )
    redis_backend_prefix: str = "celery_api_worker:"
