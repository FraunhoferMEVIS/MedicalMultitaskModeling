"""
Task samplers are used to build multi-task optimization steps.

Task samplers mainly require a cohort. However, they require a list of tasks because tasks have an id.
Further, a single task might reference multiple cohorts in the future.
"""

from __future__ import annotations

import itertools
import logging
import random
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Literal, Tuple, Union

import logfire
from torch.utils.data import DataLoader

from mmm.BaseModel import BaseModel
from mmm.data_loading.MTLDataset import DatasetStyle, MTLDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.DataSplit import DataSplit
from mmm.mtl_modules.tasks.MTLTask import MTLTask


def get_main_and_subprocess_memory_report():
    import psutil

    mainproc = psutil.Process()
    res = f"Main process {mainproc.pid} memory: {mainproc.memory_info().rss / 1024 ** 2:.2f} MB\n"
    for child in mainproc.children(recursive=True):
        res += f"Subprocess {child.pid} memory: {child.memory_info().rss / 1024 ** 2:.2f} MB\n"
    return res


class BaseSampler(ABC):
    class Config(BaseModel):
        ...

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
        mode: Literal[
            "break_with_shortest_loader", "break_with_longest_loader", "infinite"
        ] = "break_with_longest_loader"

    def __init__(self, args: Config, tasks: List[MTLTask], loader_index: DataSplit):
        self.args: CyclicTaskSampler.Config
        super().__init__(args, tasks, loader_index)

    def __len__(self) -> int:
        assert self.is_finite()

        if self.args.mode == "break_with_shortest_loader":
            relevant_length = min([len(t.cohort.get_dataloader(self.loader_index)) for t in self.tasks])  # type: ignore (len of "possibly" None)
        else:
            relevant_length = max([len(t.cohort.get_dataloader(self.loader_index)) for t in self.tasks])  # type: ignore (len of "possibly" None)
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

    def return_item(self, iterator, task: MTLTask):
        return next(iterator), task

    def _get_generator(self) -> Generator[Tuple[Any, MTLTask], None, None]:
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
                    yield self.return_item(batch_iterator, task)
                except StopIteration:
                    if task.get_name() not in tasks_that_stopped:
                        tasks_that_stopped.append(task.get_name())

                    if self.args.mode == "break_with_shortest_loader":
                        assert len(tasks_that_stopped) == 1
                        logfire.debug(
                            "Breaking loop with task {task_name} because it was shortest", task_name=task.get_name()
                        )
                        renew_and_continue = False
                    elif self.args.mode == "break_with_longest_loader":
                        renew_and_continue = len(tasks_that_stopped) < len(self.tasks)
                        logfire.debug(
                            "Tasks that stopped so far: {tasks_that_stopped}. Continuing: {renew_and_continue}",
                            tasks_that_stopped=tasks_that_stopped,
                            renew_and_continue=renew_and_continue,
                        )
                    elif self.args.mode == "infinite":
                        renew_and_continue = True
                    else:
                        raise Exception(f"Unknown mode {self.args.mode}")
                    if renew_and_continue:
                        logfire.debug("RENEWING ITERATOR FOR TASK {task_name}", task_name=task.get_name())
                        # ite = iter(task.cohort.get_dataloader(self.loader_index))
                        ite = task.cohort.build_iterator(self.loader_index)
                        yield self.return_item(ite, task)
                        iter_task_tuples[task_index] = (ite, task)
                except Exception as e:
                    with logfire.span(
                        "Error in task {task_name}: {error}, killing workers and trying to continue!",
                        task_name=task.get_name(),
                        error=e,
                        traceback="".join(traceback.format_exception(e)),
                        mem_before_killing_workers=get_main_and_subprocess_memory_report(),
                    ) as span:
                        task.cohort.terminate_workers()
                        span.set_attribute("mem_after_killing_workers", get_main_and_subprocess_memory_report())
                        ite = task.cohort.build_iterator(self.loader_index)
                        try:
                            yield self.return_item(ite, task)
                        except Exception as e:
                            logfire.error(
                                "Error in task {task_name} after restarting workers: {error}",
                                task_name=task.get_name(),
                                error=e,
                            )
                            raise e
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
