import pytest
from typing import Callable, get_args
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mmm.data_loading.synthetic.mockup import ClassificationMockupDataset
from mmm.optimization.MTLOptimizer import (
    MTLOptimizer,
    OptimizerType,
    OptimizerConfig,
    SchedulerConfig,
    ReduceLROnPlateauConfig,
)


@pytest.fixture(params=[optimconfig for optimconfig in get_args(OptimizerType)])
def default_optim_config(request):
    return request.param


@pytest.fixture
def minimal_default_optim(default_optim_config):
    """
    Builds all optimizers of the default configs for a single small parameter
    """
    optim_config = default_optim_config()
    o = optim_config.get_type()([torch.rand(5)], **optim_config.get_params())
    return o, optim_config


def test_optim_from_config(minimal_default_optim):
    o, optim_config = minimal_default_optim
    assert o.param_groups[0]["lr"] == optim_config.lr


def test_scheduler_from_config(default_scheduler_config: Callable[[], SchedulerConfig], minimal_default_optim):
    # optimizer = default_optim_config().build_instance([torch.rand(5)])
    optim, _ = minimal_default_optim
    s = default_scheduler_config().build_instance(optim, 100)

    if isinstance(s, ReduceLROnPlateau):
        return

    # Learning rate should always be greater than 0.
    lrs = []
    for _ in range(100):
        s.step()
        lrs.append(optim.param_groups[0]["lr"])
    assert all([l > 0.0 for l in lrs])


def test_plateau_lr_optim(default_optim_config: Callable[[], OptimizerType]):
    o = MTLOptimizer(
        {"somelayer": nn.Linear(5, 5)},
        MTLOptimizer.Config(
            optim_config=default_optim_config(),
            lr_scheduler_configs=[ReduceLROnPlateauConfig()],
        ),
        max_epochs=100,
    )
    o.add_task(ClassificationMockupDataset.build_classification_task("t1", 4, 2))

    # Same training losses for all steps, this is a plateau and needs to be solved!
    lrs = []
    for i in range(50):
        o.after_loop_step(i, [[0.3], [0.3]], [[0.3], [0.3]])
        lrs.append(o.optimizer.param_groups[0]["lr"])
    assert lrs[-1] < lrs[0]

    # losses reduce, this should be ok!
    lrs = []
    for i in range(50):
        o.after_loop_step(i, [[0.3], [0.3]], [[0.3], [0.3 * (0.97**i)]])
        lrs.append(o.optimizer.param_groups[0]["lr"])
    assert lrs[0] == lrs[-1]
