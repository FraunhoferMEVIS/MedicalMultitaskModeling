from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest
import redis

from mmm.api.utils import download_image
from mmm.settings import mtl_settings

try:
    from fastapi.testclient import TestClient
except ImportError:
    if not TYPE_CHECKING:
        TestClient = None

from ..test_data import IMAGE_IDS, IMAGE_PROPERTIES, IMAGES


@pytest.mark.parametrize(
    "params,expected",
    [({"url": i}, p) for i, p in zip(IMAGES, IMAGE_PROPERTIES)],
    ids=IMAGE_IDS,
)
def test_image_download(params: dict, expected: dict):
    img = download_image(**params)

    assert np.array(img).shape == (*expected["HW"], 3)
    assert np.array(img).dtype == np.uint8, f"Image {params} has dtype {img.dtype} instead of uint8"


@pytest.fixture(scope="session")
def kv_initializer(request) -> Callable[[str], redis.Redis]:
    """
    Should use first logical database for communication such that tests see their own logical database.

    However, due to current limitations of the Redis service within the CI, only one logical database is used.
    """
    try:
        mtl_settings.kv.ping()
        mtl_settings.kv.publish("pytest", f"Initializing key-value store for tests")
    except Exception as e:
        pytest.skip(f"Could not connect to Redis at {mtl_settings.redis_url} due to {e}")

    client = redis.Redis.from_url(mtl_settings.redis_url)
    client.select(1)

    def builder(db_category: str = "pytest") -> redis.Redis:
        return client

    def finalizer():
        mtl_settings.kv.publish("pytest", f"Cleaning up kv_initializer")
        client.flushdb()
        client.close()

    request.addfinalizer(finalizer)
    return builder


@pytest.fixture
def api_client() -> TestClient:
    if TestClient is None:
        pytest.skip("Extra 'api' not available.")

    from mmm.inference_app import build_app

    return TestClient(build_app())


def test_api_initialization(api_client: TestClient):
    status = api_client.get("/status").json()
    assert "kv_status" in status
