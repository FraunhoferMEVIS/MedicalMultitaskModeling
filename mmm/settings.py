from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    max_classes_detailed_logging: int = 25
    ignore_class_value: int = -1

    class Config:
        env_prefix = "mtl_"


mtl_settings = Settings()
