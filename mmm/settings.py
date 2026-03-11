"""
Enables other modules to import a settings object like this: `from mmm.settings import mtl_settings`
"""

import os

from m3_sdk.DistributedPath import DistributedPath
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    A nested model can be defined as an environment variable like `export MMM_settingname='{"key": "value"}'`.
    See https://docs.pydantic.dev/latest/concepts/pydantic_settings/#parsing-environment-variable-values
    """

    model_config = SettingsConfigDict(env_prefix="MMM_")

    # Training
    ignore_class_value: int = -1

    # Logging
    efficiency_hints: bool = False
    default_log_folder: DistributedPath | None = Field(
        default=None,
        description="Used for storing files such as predictions.",
    )
    max_classes_detailed_logging: int = 25
    st_app_base: str = "http://localhost:8501"

    # API
    redis_url: str = "redis://localhost:6379/0"
    multinode_handshake_key: str = "ddp:multinodehandshake"
    adapter_prefix: str = "task"
    ongoing_training_prefix: str = "ongoing_training"
    subj_prefix: str = "subject"

    # S3 settings
    s3_url: str = Field(default_factory=lambda: os.getenv(key="S3URL", default="http://localhost:9000"))
    s3_access_key: str = Field(default_factory=lambda: os.getenv(key="AWS_ACCESS_KEY_ID", default="minioadmin"))
    s3_secret_key: str = Field(default_factory=lambda: os.getenv(key="AWS_SECRET_ACCESS_KEY", default="minioadmin"))

    # Integrations
    labelstudio: dict = Field(
        default={"base_url": "http://localhost:8080", "api_key": "apitoken"},
        description="Used to authenticate a Labelstudio client as label_studio_sdk.client.Labelstudio(**labelstudio)",
    )

    @property
    def kv(self):
        try:
            return self._rclient
        except AttributeError:
            import redis
            from redis.backoff import ExponentialBackoff
            from redis.retry import Retry

            self._rclient = redis.Redis.from_url(
                self.redis_url, retry=Retry(backoff=ExponentialBackoff(cap=25.0, base=1.0), retries=10)
            )
            return self._rclient

    @property
    def ls(self):
        try:
            return self._ls
        except AttributeError:
            from label_studio_sdk import Client

            self._ls = Client(url=self.labelstudio["base_url"], api_key=self.labelstudio["api_key"])
            return self._ls

    @property
    def s3(self):
        try:
            return self._s3
        except AttributeError:
            from minio import Minio

            self._s3 = Minio(
                self.s3_url.replace("http://", "").replace("https://", ""),
                self.s3_access_key,
                self.s3_secret_key,
                secure=self.s3_url.startswith("https://"),
            )
            return self._s3


mtl_settings = Settings()
