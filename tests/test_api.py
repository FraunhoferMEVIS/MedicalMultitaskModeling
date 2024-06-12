from __future__ import annotations
from typing import Any
import os
from pathlib import Path
import pytest
import requests
from pydantic import ValidationError


from mmm.data_loading.medical.wsi_service import WSIService
from mmm.data_loading.DistributedPath import DistributedPath

try:
    from fastapi.testclient import TestClient
    from mmm.labelstudio_ext.DLModel import (
        DLModel,
        ModelInput,
        WSIInvocation,
        SingleInvocation,
    )
    from mmm.labelstudio_ext.LGBModel import LGBModel
except ImportError:
    pass


@pytest.fixture
def wsi_service() -> WSIService:
    wsi_service = WSIService(os.getenv("WSISERVICEURL", default="http://localhost:9506"))
    alive_url = wsi_service.get_is_alive_route()
    try:
        assert requests.get(alive_url).json()["status"] == "ok"
        return wsi_service
    except Exception as e:
        pytest.skip(f"WSI Service not available due to {e}")


@pytest.fixture
def api_client() -> TestClient:
    if TestClient is None:
        pytest.skip("Extra 'api' not available.")
    try:
        from mmm.inference_app import build_app, APISettings

        TESTING_MODULES_PATH = DistributedPath(uri="/jfs/output/mum_modules/mum_t46_48_2_e101.pt")
        app = build_app(APISettings(modules_path=TESTING_MODULES_PATH))
        return TestClient(app)
    except requests.exceptions.ConnectionError as e:
        pytest.skip(f"API Service not available due to {e}")


@pytest.fixture(
    params=[
        "s3://dataroot/histoartefact_preannotation/SynD_0_001.svs",
        "s3://dataroot/semicol/DATASET_VAL/02_BX/2pT4GFzn.ome.tiff",
        "/jfs/panda/train_images/4b77f015acc40c7e39470f3ebb658818.tiff",
    ]
)
def example_wsi(request) -> str:
    return request.param


def test_tiffslide_available(wsi_service: WSIService):
    status = requests.get(wsi_service.get_is_alive_route()).json()
    plugin_names = [d["name"] for d in status["plugins"]]
    assert "tiffslide" in plugin_names, "MMM is only tested with wsi service with tiffslide plugin"

    assert status["version"].startswith("0.12."), "MMM is only tested with wsi service 0.12.x"


def test_api_initialization(api_client: TestClient):
    status = api_client.get("/status").json()
    modules_path: DistributedPath = DistributedPath(**status["modules_path"])
    assert modules_path.exists(), f"Testing the api requires trained MTL modules"


def test_dl_initialization(api_client: TestClient):
    # The API should make the deep learning tasks available
    dl_settings = DLModel.Config(**api_client.get("/dl/settings").json())
    assert dl_settings.category == "DeepLearning"


def test_lgb_initialization(api_client: TestClient):
    # The API should make the deep learning tasks available
    lgb_settings = LGBModel.Config(**api_client.get("/lgbm/settings").json())
    assert lgb_settings.category == "LightGBM"


def test_invocation_validation_duplicate(api_client: TestClient, example_wsi: str):
    with pytest.raises(ValidationError):
        ModelInput(
            invo=WSIInvocation(
                invocation_type="wsi",
                wsi_path=example_wsi,
                mtl_task_id="whatever",
                for_levels=[0, 0],
            )
        )


def test_invocation_validation_empty(api_client: TestClient, example_wsi: str):
    with pytest.raises(ValidationError):
        ModelInput(
            invo=WSIInvocation(
                invocation_type="wsi",
                wsi_path=example_wsi,
                mtl_task_id="whatever",
                for_levels=[],
            )
        )


def test_invocation_validation_invalid(api_client: TestClient, example_wsi: str):
    with pytest.raises(ValidationError):
        ModelInput(
            invo=WSIInvocation(
                invocation_type="wsi",
                wsi_path=example_wsi,
                mtl_task_id="whatever",
                for_levels=[-1],
            )
        )


def test_patchification(example_wsi: str):
    H, W, P = 14, 24, 7
    wsi_invocation = WSIInvocation(
        invocation_type="wsi",
        wsi_path=example_wsi,
        mtl_task_id="whatever",
        for_levels=[0],
        patch_size=P,
    )
    patch_coords = wsi_invocation.get_patch_coords(H, W)
    rows, cols = zip(*patch_coords)
    assert min(rows) < 0 and max(rows) + P > H, "Patches should exceed the image size"
    assert min(cols) < 0 and max(cols) + P > W, "Patches should exceed the image size"
    assert max(rows) <= H, "Patches should be within the image size"
    assert max(cols) <= W, "Patches should be within the image size"
    should_more_than = (H // P) * (W // P)
    assert len(patch_coords) >= should_more_than, "There should be at least one patch per patch size"


def test_wsi_inference(api_client: TestClient, example_wsi: str):
    invo = ModelInput(
        invo=WSIInvocation(
            invocation_type="wsi",
            wsi_path=example_wsi,
            mtl_task_id="histoarteseg",
            for_levels=[2],
        )
    )
    r = api_client.post("/dl/invocation", json=invo.model_dump()).json()
    assert r["type"] == "FeatureCollection"
    assert "features" in r
