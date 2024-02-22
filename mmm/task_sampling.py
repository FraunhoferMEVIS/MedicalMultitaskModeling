"""
Task samplers are used to build multi-task optimization steps.

Task samplers mainly require a cohort. However, they require a list of tasks because tasks have an id.
Further, a single task might reference multiple cohorts in the future.
"""

from __future__ import annotations

import itertools
from typing import Optional, List, Any, Tuple, Generator, Sized, cast, Dict
import random
import logging
from abc import ABC, abstractmethod
from typing import Literal, Union

from torch.utils.data import DataLoader

from mmm.BaseModel import BaseModel
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.data_loading.MTLDataset import MTLDataset, DatasetStyle
from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.DataSplit import DataSplit


class BaseSampler(ABC):
    class Config(BaseModel):
        sampler_type: str

    def __init__(self, args: Config, tasks: List[MTLTask], loader_index: DataSplit) -> None:
        """
        loader_index DataSplit.train for the training data loader, DataSplit.val for validation
        """
        self.args = args
        self.loader_index: DataSplit = loader_index
        self.tasks: List[MTLTask] = tasks
        task_ids = [t.get_name() for t in self.tasks]
        assert len(task_ids) == len(set(task_ids))

        self._iterator = None

    def _get_dataset_styles(self) -> List[DatasetStyle]:
        dss: List[MTLDataset] = [t.cohort.get_dataset(self.loader_index) for t in self.tasks]
        return [ds.get_dataset_style() for ds in dss]

    @abstractmethod
    def is_finite(self) -> bool:
        """
        A finite task sampler has a __len__ property
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """
        The total number of batches of all tasks combined. Only finite task samplers have a length.
        """
        raise NotImplementedError(f"Task sampler {self} does not have a length")

    @abstractmethod
    def _get_generator(self) -> Generator:
        raise NotImplementedError(f"Task sampler {self} needs to implement __iter__ protocol.")

    def continue_iter(self):
        if self._iterator is None:
            self._iterator = self._get_generator()
        return self._iterator

    def __iter__(self):
        return self._get_generator()


class ConcatTaskSampler(BaseSampler):
    """
    Iterates over all tasks and yields each batch one after another.
    All batches of task n will be returned before the first batch of task n+1.

    All cohort's dataloaders need to have a finite length.
    """

    class Config(BaseSampler.Config):
        sampler_type: Literal["concat"] = "concat"

    def __init__(self, args: Config, tasks: List[MTLTask], loader_index: DataSplit):
        super().__init__(args, tasks, loader_index)
        self.loader_lengths: List[int] = [len(t.cohort.get_dataloader(self.loader_index)) for t in self.tasks]

    def is_finite(self) -> bool:
        return True

    def __len__(self) -> int:
        return sum(self.loader_lengths)

    def _get_generator(self):
        for task in self.tasks:
            task_loader: DataLoader = task.cohort.get_dataloader(self.loader_index)
            for x in task_loader:
                yield x, task
            logging.debug(f"Done iterating through {task.get_name()}")


class BalancedTaskSampler(BaseSampler):
    """
    Randomly yields tasks, such that all tasks are drained exactly once.

    Weights are only computed in the beginning,
    as a result extra care might have to be taken with map-style datasets that change their length over time.
    """

    class Config(BaseSampler.Config):
        sampler_type: Literal["balanced"] = "balanced"

    def __init__(self, args: Config, tasks: List[MTLTask], loader_index: DataSplit):
        super().__init__(args, tasks, loader_index)

        def get_weight(cohort: TrainValCohort) -> float:
            if False not in [s is DatasetStyle.MapStyle for s in self._get_dataset_styles()]:
                return len(cohort.get_dataloader(self.loader_index)) / self.__len__()
            else:  # if the dataloader doesn't have a fixed len, approximate!
                return 1.0 / len(self.tasks)

        self.task_weights = [(t, get_weight(t.cohort)) for t in self.tasks]
        logging.debug(f"{self.task_weights=}")
        self.iters = {t.get_name(): t.cohort.build_iterator(self.loader_index) for t in self.tasks}

    def __len__(self) -> int:
        return sum([len(t.cohort.get_dataloader(self.loader_index)) for t in self.tasks])

    def is_finite(self) -> bool:
        return False not in [s is DatasetStyle.MapStyle for s in self._get_dataset_styles()]

    def _get_generator(self):
        while len(self.task_weights) > 0:
            t, task_weight = random.choices(self.task_weights, weights=[t[1] for t in self.task_weights], k=1)[0]
            try:
                yield next(self.iters[t.get_name()]), t
            except StopIteration:
                self.task_weights.remove((t, task_weight))
                logging.debug(f"Removing {t.get_name()} from task sampler")


class CyclicTaskSampler(BaseSampler):
    """
    Alternates deterministically through the tasks.

    If `break_with_shortest_loader` is True, `len(shortest_dataloader) * len(tasks)` is the length.
    If not, the task sampler is infinite.
    """

    class Config(BaseSampler.Config):
        sampler_type: Literal["cyclic"] = "cyclic"
        mode: Literal["break_with_shortest_loader", "break_with_longest_loader", "infinite"] = (
            "break_with_longest_loader"
        )

    def __init__(self, args: Config, tasks: List[MTLTask], loader_index: DataSplit):
        self.args: CyclicTaskSampler.Config
        super().__init__(args, tasks, loader_index)

    def __len__(self) -> int:
        assert self.is_finite()

        if self.args.mode == "break_with_shortest_loader":
            relevant_length = min(
                [
                    len(t.cohort.get_dataloader(self.loader_index))  # type: ignore (len of "possibly" None)
                    for t in self.tasks
                ]
            )
        else:
            relevant_length = max(
                [
                    len(t.cohort.get_dataloader(self.loader_index))  # type: ignore (len of "possibly" None)
                    for t in self.tasks
                ]
            )
        return relevant_length * len(self.tasks)

    def is_finite(self) -> bool:
        styles: List[DatasetStyle] = self._get_dataset_styles()
        if DatasetStyle.IterStyle in styles:
            return False
        else:
            if False in [s is DatasetStyle.MapStyle for s in styles]:
                # Once a weird thing happenened and if does happen again, this warning exposes it
                logging.warning([s is DatasetStyle.MapStyle for s in styles])
            assert DatasetStyle.IterStyle not in styles
            return self.args.mode in [
                "break_with_shortest_loader",
                "break_with_longest_loader",
            ]

    def _get_generator(self) -> Generator[Tuple[Any, MTLTask], None, None]:
        # iter_task_tuples = [(iter(t.cohort.get_dataloader(self.loader_index)), t)
        #                     for t in self.tasks]
        iter_task_tuples = [(t.cohort.build_iterator(self.loader_index), t) for t in self.tasks]
        if self.is_finite():
            counting_iter = range(self.__len__() // len(iter_task_tuples))
        else:
            counting_iter = itertools.count()

        tasks_that_stopped, renew_and_continue = [], True

        for _ in counting_iter:
            for task_index, (batch_iterator, task) in enumerate(iter_task_tuples):
                task: MTLTask
                try:
                    yield next(batch_iterator), task
                except StopIteration:
                    if task.get_name() not in tasks_that_stopped:
                        tasks_that_stopped.append(task.get_name())

                    if self.args.mode == "break_with_shortest_loader":
                        assert len(tasks_that_stopped) == 1
                        logging.debug(f"Breaking loop with task {task.get_name()} because it was shortest")
                        renew_and_continue = False
                    elif self.args.mode == "break_with_longest_loader":
                        renew_and_continue = len(tasks_that_stopped) < len(self.tasks)
                        logging.debug(f"{tasks_that_stopped=}, {renew_and_continue=}, {len(self.tasks)=}")
                    elif self.args.mode == "infinite":
                        renew_and_continue = True
                    else:
                        raise Exception(f"Unknown mode {self.args.mode}")
                    if renew_and_continue:
                        logging.debug(f"RENEWING ITERATOR FOR TASK {task.get_name()}")
                        # ite = iter(task.cohort.get_dataloader(self.loader_index))
                        ite = task.cohort.build_iterator(self.loader_index)
                        yield next(ite), task
                        iter_task_tuples[task_index] = (ite, task)
            if not renew_and_continue:
                break


TaskSamplerTypes: Dict = {
    ConcatTaskSampler.Config().sampler_type: ConcatTaskSampler,
    BalancedTaskSampler.Config().sampler_type: BalancedTaskSampler,
    CyclicTaskSampler.Config().sampler_type: CyclicTaskSampler,
}
TaskSamplerConfig = Union[
    ConcatTaskSampler.Config,
    BalancedTaskSampler.Config,
    CyclicTaskSampler.Config,
]
