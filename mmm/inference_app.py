from __future__ import annotations
from typing import Any
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from urllib.parse import urljoin

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from label_studio_sdk import Client
except ImportError:
    FastAPI, CORSMiddleware, Client = Any, Any, Any

from mmm.data_loading.DistributedPath import DistributedPath
from mmm.labelstudio_ext.DLModel import DLModel
from mmm.labelstudio_ext.LGBModel import LGBModel
from mmm.labelstudio_ext.NativeBlocks import NativeBlocks, MMM_MODELS, DEFAULT_MODEL
from mmm.labelstudio_ext.LabelstudioCredentials import LabelstudioCredentials


class APISettings(BaseSettings):
    class Config:
        env_prefix = "MTLAPI_"

    modules_path: DistributedPath | str = MMM_MODELS[DEFAULT_MODEL]
    labelstudio_base: str = "http://datanodefec:9505"
    labelstudio_token: str = "1234567890"
    app_base: str = Field(
        "http://localhost:8000",
        description="The base URL of this service",
    )
    device_identifier: str = "cuda:0"
    dlconfig: DLModel.Config = DLModel.Config()
    lgbconfig: LGBModel.Config = LGBModel.Config()

    allow_cors: bool = True


def build_app(settings: APISettings) -> FastAPI:
    app = FastAPI()

    if settings.allow_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.settings = settings

    creds = LabelstudioCredentials(url=settings.labelstudio_base, token=settings.labelstudio_token)
    ls_client: Client = creds.build_client()

    nativeblocks = NativeBlocks(settings.modules_path, settings.device_identifier)

    dlmodel = DLModel(
        settings.dlconfig,
        ls_client,
        urljoin(settings.app_base, "/dl"),
        nativeblocks,
    )
    app.include_router(dlmodel.build_router(), prefix="/dl")

    lgbmmodel = LGBModel(
        settings.lgbconfig,
        ls_client,
        urljoin(settings.app_base, "/lgbm"),
        nativeblocks,
    )
    app.include_router(lgbmmodel.build_router(), prefix="/lgbm")

    @app.get("/status")
    async def status():
        return {
            "client_status": ls_client.check_connection(),
            "modules_path": settings.modules_path,
        }

    return app


if __name__ == "__main__":
    import uvicorn

    app = build_app(APISettings())

    uvicorn.run("mmm.inference_api:app", host="0.0.0.0", port=8000, reload=True)
