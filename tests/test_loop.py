from typing import Any, Dict, List, Mapping, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmm.data_loading.MTLDataset import MTLDataset
from mmm.data_loading.synthetic.mockup import ClassificationMockupDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.logging.type_ext import StepFeedbackDict
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.trainer.Loop import Loop, LoopConfig, TrainLoopConfig, ValLoopConfig


class CounterDS(Dataset[Dict[str, Any]]):
    def __init__(self, n) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index) -> Dict[str, Any]:
        return {"i": index}


class TrackerClfTask(MTLTask):
    num = 0

    def __init__(self, train_n, val_n) -> None:
        config = MTLTask.Config(module_name=f"tracker{TrackerClfTask.num}")
        TrackerClfTask.num += 1

        class D(MTLDataset):
            @staticmethod
            def get_mandatory_keys():
                return ["i"]

        self.clf_cohort = TrainValCohort(
            TrainValCohort.Config(batch_size=(1, 1), shuffle_loaders=(False, False), num_workers=0),
            D(CounterDS(train_n)),
            D(CounterDS(val_n)),
            for_task_name=config.module_name,
        )
        self.clf_cohort.prepare_epoch(0)
        self.batches = []
        super().__init__(config, self.clf_cohort)
        self.task_modules = nn.ModuleDict({})

    def training_step(self, batch: Dict[str, Any], shared_blocks: Mapping[str, SharedBlock]):
        self.batches.append(batch)
        return None


def test_full_valloop(wandb_run):
    tasks = [TrackerClfTask(10, 3), TrackerClfTask(10, 7)]
    stepcounter = {}
    loop = ValLoopConfig().build_instance(
        tasks,  # type: ignore
        0,
        lambda b, t: t.training_step(b, {}),
        SharedModules({}),
        "testfullvalloop",
        stepcounter,
        None,
    )
    losses = loop.drain()
    assert not losses
    assert (
        len(set([b["i"] for b in tasks[0].batches])) == 3 and len(set([b["i"] for b in tasks[1].batches])) == 7
    ), "The default validation loop should exhaust the validation data"
