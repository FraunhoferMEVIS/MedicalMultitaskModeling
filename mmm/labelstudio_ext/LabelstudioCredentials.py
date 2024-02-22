from typing import Any
from pydantic_settings import BaseSettings

try:
    from label_studio_sdk import Client, Project
    from label_studio_converter.brush import (
        decode_rle,
        encode_rle,
        decode_from_annotation,
    )
except ImportError:
    Client, Project = Any, Any


class LabelstudioCredentials(BaseSettings):
    url: str
    token: str

    class Config:
        env_prefix = "mtltorchlabelstudio_"

    def build_client(self) -> Client:
        return Client(self.url, self.token)
