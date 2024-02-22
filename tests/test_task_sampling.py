import pytest
from typing import List, get_args, Type
import torch
from torch.utils.data import Dataset

from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder
from mmm.mtl_modules.shared_blocks.Squeezer import Squeezer
from mmm.data_loading.ClassificationDataset import ClassificationDataset
from mmm.mtl_modules.tasks.ClassificationTask import ClassificationTask
from mmm.task_sampling import (
    BaseSampler,
    ConcatTaskSampler,
    BalancedTaskSampler,
    CyclicTaskSampler,
)
from mmm.task_sampling import TaskSamplerConfig, TaskSamplerTypes
from mmm.data_loading.synthetic.mockup import ClassificationMockupDataset
from mmm.DataSplit import DataSplit
from mmm.trainer.Loop import Loop, LoopConfig, TrainLoopConfig
from mmm.optimization.MTLOptimizer import MTLOptimizer
from mmm.neural.modules.simple_cnn import MiniConvNet
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules


@pytest.fixture(params=[samplerconfig for samplerconfig in get_args(TaskSamplerConfig)])
def default_tasksampler_config(request):
    return request.param


def test_concat_task_sampler():
    ts: List[MTLTask] = [
        ClassificationMockupDataset.build_classification_task(name, train_n, val_n)
        for name, train_n, val_n in [("t1", 4, 2), ("t2", 5, 2)]
    ]  # type: ignore

    for t in ts:
        t.cohort.prepare_epoch(0)

    task_sampler = ConcatTaskSampler(ConcatTaskSampler.Config(), ts, DataSplit.train)
    loop_result = [x[1].get_name() for x in task_sampler]
    assert loop_result == ["t1", "t1", "t2", "t2", "t2"]


def test_cyclic_task_sampler_finite(list_of_tasks):
    sampler = CyclicTaskSampler(
        CyclicTaskSampler.Config(mode="break_with_shortest_loader"),
        list_of_tasks,
        DataSplit.train,
    )
    loop_result = [x[1].get_name() for x in sampler]
    assert loop_result == ["t1", "t2", "t3"] * 2


def test_cyclic_task_sampler_infinite(wandb_run):
    # Define the dataset which tracks the __getitem__ accesses
    class DS(Dataset):
        def __init__(self, N) -> None:
            self.accesses, self.N = [], N

        def __len__(self) -> int:
            return self.N

        def __getitem__(self, index: int):
            self.accesses.append(index)
            return {"image": torch.rand(3, 28, 28), "class": index}

    # g = iter(sampler)

    enc = PyramidEncoder(PyramidEncoder.Config(model=MiniConvNet.Config()))
    squeezer = Squeezer(
        Squeezer.Config(),
        enc_out_channels=enc.get_feature_pyramid_channels(),
        enc_strides=enc.get_strides(),
    )
    sharedmodules = SharedModules({"encoder": enc, "squeezer": squeezer})

    # Define the classification tasks using the dataset
    dss = [DS(N=10) for _ in range(4)]
    random_classnames = [f"c{i}" for i in range(10)]
    tasks: List[MTLTask] = [
        ClassificationTask(
            squeezer.get_hidden_dim(),
            # class_names=random_classnames,
            args=ClassificationTask.Config(module_name=f"ds{i}"),
            cohort=TrainValCohort(
                TrainValCohort.Config(batch_size=(2, 2), num_workers=1),
                train_ds=ClassificationDataset(ds, class_names=random_classnames),
                val_ds=ClassificationDataset(ds, class_names=random_classnames),
            ),
        )
        for i, ds in enumerate(dss)
    ]

    for task in tasks:
        task.cohort.prepare_epoch(0)

    sampler = CyclicTaskSampler(CyclicTaskSampler.Config(mode="infinite"), tasks, DataSplit.train)

    def create_and_drain_loop():
        optim = MTLOptimizer({"encoder": enc, "squeezer": squeezer}, MTLOptimizer.Config(), max_epochs=50)
        for task in tasks:
            optim.add_task(task)
        loop = Loop(
            TrainLoopConfig(max_steps=4),
            sampler,
            accumulate_losses_for_steps=1,
            training_mode=True,
            epoch=0,
            task_step=lambda batch, task: task.training_step(task.prepare_batch(batch), sharedmodules),
            shared_blocks=sharedmodules,
            prefix="testloop",
            step_counters={},
            optim=optim,
        )
        return loop.drain_to_dict()

    create_and_drain_loop()
    create_and_drain_loop()
    assert False not in [len(set(samples)) == len(samples) for samples in [ds.accesses for ds in dss]]
    for task in tasks:
        task.cohort.terminate_workers()
    create_and_drain_loop()
    assert False not in [len(set(samples)) == len(samples) for samples in [ds.accesses for ds in dss]]


def test_balanced_task_sampler(list_of_tasks):
    loop_result = [x for x in BalancedTaskSampler(BalancedTaskSampler.Config(), list_of_tasks, DataSplit.train)]
    task_names = [x[1].get_name() for x in loop_result]
    batches = [x[0] for x in loop_result]
    assert len(loop_result) == 8 and batches[0]["image"].size() == (2, 1, 4, 4)
    assert task_names.count("t1") == 2 and task_names.count("t2") == 3 and task_names.count("t3") == 3


def test_loading_tasksampler_from_config(list_of_tasks, default_tasksampler_config: Type[TaskSamplerConfig]):
    config = default_tasksampler_config()
    r = TaskSamplerTypes[config.sampler_type](config, list_of_tasks, DataSplit.train)
    # Hard to test specifics, so just test that the result is a child of the BaseSampler
    assert isinstance(r, BaseSampler) and type(r) != BaseSampler

    # And test some examples
    if isinstance(default_tasksampler_config(), CyclicTaskSampler.Config):
        assert isinstance(r, CyclicTaskSampler)
    elif isinstance(default_tasksampler_config(), BalancedTaskSampler.Config):
        assert isinstance(r, BalancedTaskSampler)
