from pathlib import Path
import typing as tau
from typing import Callable
import os
import pytest
import wandb
import torch
import torchvision.transforms as transforms
import tempfile

from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.mtl_modules.tasks.SemSegTask import SemSegTask
from mmm.optimization.MTLOptimizer import MTLOptimizer, SchedulerType

from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.data_loading.SemSegDataset import SemSegDataset
from mmm.data_loading.synthetic.mockup import ClassificationMockupDataset
from mmm.data_loading.synthetic.shape_dataset import ShapeDataset, CanvasConfig
from mmm.transforms import KeepOnlyKeysInDict

from mmm.mtl_modules.shared_blocks.PyramidDecoder import PyramidDecoder


# Extern fixtures
from tests.test_mtloptimizer import default_optim_config
from tests.test_shared_blocks import default_encoder_factory, default_decoder_factory
from tests.test_mtl_training import mtl_semseg_trainer_factory


def pytest_generate_tests(metafunc):
    # turn_off_wandb_sync
    os.environ["WANDB_MODE"] = "offline"
    # Set ML_DATA_OUTPUT to a temporary directory, if it does not exist
    # Used e.g. in node-shared caching
    if "ML_DATA_OUTPUT" not in os.environ:
        os.environ["ML_DATA_OUTPUT"] = str(Path(tempfile.gettempdir()) / "ml_data_output")


@pytest.fixture(params=[True, False])
def BOOLEAN(request) -> bool:
    """If a test can be executed in exactly two ways this fixture can help"""
    return request.param


@pytest.fixture
def wandb_run(tmp_path: Path):
    # Make sure nothing from the tests is synced to any server
    os.environ["WANDB_MODE"] = "offline"
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


@pytest.fixture
def task_factory() -> Callable[[str], MTLTask]:
    def create_task(task_name: str) -> MTLTask:
        t = ClassificationMockupDataset.build_classification_task(task_name, 8, 4)
        t.cohort.prepare_epoch(0)
        return t

    return create_task


@pytest.fixture
def list_of_tasks():
    ts = []
    for name, train_n, val_n in [("t1", 4, 2), ("t2", 6, 2), ("t3", 6, 2)]:
        t = ClassificationMockupDataset.build_classification_task(name, train_n, val_n)
        t.cohort.prepare_epoch(0)
        ts.append(t)
    yield ts


@pytest.fixture(params=[schedulerconfig for schedulerconfig in tau.get_args(SchedulerType)])
def default_scheduler_config(request):
    return request.param


@pytest.fixture
def shape_semseg_cohort() -> TrainValCohort[SemSegDataset]:
    shapes_train_ds = ShapeDataset(ShapeDataset.Config(N=400, canvas=CanvasConfig(canvas_size=(96, 96))))
    shapes_val_ds = ShapeDataset(ShapeDataset.Config(N=100, canvas=CanvasConfig(canvas_size=(96, 96))))
    base_transform = transforms.Compose([ShapeDataset.cohort_transform(), KeepOnlyKeysInDict(keys={"image", "label"})])
    return TrainValCohort(
        TrainValCohort.Config(batch_size=(8, 8), num_workers=0),
        SemSegDataset(
            shapes_train_ds,
            class_names=["background"] + shapes_train_ds.get_class_names(),
            src_transform=base_transform,
        ),
        SemSegDataset(
            shapes_val_ds,
            class_names=["background"] + shapes_val_ds.get_class_names(),
            src_transform=base_transform,
        ),
    )


@pytest.fixture
def shape_segtask_factory(shape_semseg_cohort: TrainValCohort):
    def shape_segtask(for_decoder: PyramidDecoder, task_name="shapeseg") -> SemSegTask:
        # shapes_train_ds = TransformedSubset(shapes_ds, transform=ShapeSegmentationDataset.semseg_transform())
        mock_segtask = SemSegTask(
            shape_semseg_cohort.datasets[0].class_names,
            for_decoder,
            SemSegTask.Config(module_name=task_name, encoder_key="encoder", decoder_key="decoder"),
            shape_semseg_cohort,
        )
        return mock_segtask

    return shape_segtask
