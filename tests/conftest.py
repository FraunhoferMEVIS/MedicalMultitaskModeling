"""
For this large test suite, this file should be kept as minimal as possible to enable small tests to be fast.
"""

from pathlib import Path
import os
import uuid
import tempfile
import wandb
import pytest
import torch


def pytest_generate_tests(metafunc):
    # turn_off_wandb_sync
    os.environ["WANDB_MODE"] = "disabled"
    # Set ML_DATA_OUTPUT to a temporary directory, if it does not exist
    # Used e.g. in node-shared caching
    if "ML_DATA_OUTPUT" not in os.environ:
        os.environ["ML_DATA_OUTPUT"] = str(Path(tempfile.gettempdir()) / "ml_data_output")

    from mmm.settings import mtl_settings, DistributedPath

    if mtl_settings.default_log_folder is None:
        mtl_settings.default_log_folder = DistributedPath.from_string(
            str(Path(tempfile.gettempdir()).joinpath(f"m3_pytest_{uuid.uuid4()}"))
        )
        mtl_settings.default_log_folder.upath().mkdir(parents=True, exist_ok=True)


@pytest.fixture
def wandb_run(tmp_path: Path):
    # Make sure nothing from the tests is synced to any server
    os.environ["WANDB_MODE"] = "disabled"
    return wandb.init(dir=str(tmp_path))


@pytest.fixture
def torch_device() -> str:
    """
    If possible, try to use a specified GPU for testing. Uses environment variable LOCAL_RANK if set.
    """
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", default=0)))
        return "cuda"
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)
        return "cuda"
    else:
        return "cpu"
