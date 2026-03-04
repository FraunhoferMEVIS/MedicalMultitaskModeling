from __future__ import annotations

import json
import logging
import time
import traceback
from abc import abstractmethod
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union

import logfire
import torch
import torch.distributed as dist
import wandb
from m3_sdk.models import TrainingStepEvent
from pydantic import Field
from tqdm.auto import tqdm
from typing_extensions import Annotated

from mmm.BaseModel import BaseModel
from mmm.DataSplit import DataSplit
from mmm.event_selectors import EventSelector, FixedEventSelector, RecurringEventSelector
from mmm.logging.type_ext import StepFeedbackDict
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.optimization.MTLOptimizer import MTLOptimizer
from mmm.settings import mtl_settings
from mmm.task_sampling import BalancedTaskSampler, BaseSampler, CyclicTaskSampler, TaskSamplerConfig, TaskSamplerTypes


class FixedMultistep(BaseModel):
    """
    Does an optimization step after a fixed number of forward passes `step_num`.
    """

    multistep_type: Literal["fixed"] = "fixed"
    step_num: int = 1


class LinearMultistep(BaseModel):
    """
    Scales the number of accumulated steps with the number of tasks.

    For example, if you use three tasks and factor one there will be three forward passes per optimization step.
    """

    multistep_type: Literal["linear"] = "linear"
    factor: int = 1


MultistepMode = Union[FixedMultistep, LinearMultistep]


class LoopLogConfig(BaseModel):
    progress_bar: bool = True
    print_dateformat: str = "%H:%M.%S"
    log_result: bool = Field(
        default=True,
        description="If true, logs more detailed statistics after the loop",
    )
    include_mem: Annotated[EventSelector, Field(discriminator="selector_type")] = FixedEventSelector(
        at_iterations=[0, 1, 2]
    )


class LoopConfig(BaseModel):
    max_steps: int = Field(default=-1, description="Maximum steps per loop, -1 for unlimited")
    log_args: LoopLogConfig = LoopLogConfig()
    multistep_mode: Annotated[MultistepMode, Field(discriminator="multistep_type")] = LinearMultistep()

    task_sampler: Annotated[TaskSamplerConfig, Field(discriminator="sampler_type")]

    def _compute_multistep_num(self, num_tasks: int) -> int:
        if self.multistep_mode.multistep_type == FixedMultistep().multistep_type:
            return self.multistep_mode.step_num
        elif self.multistep_mode.multistep_type == LinearMultistep().multistep_type:
            return self.multistep_mode.factor * num_tasks
        else:
            raise Exception(f"Unknown multistep mode {self.multistep_mode}")

    @abstractmethod
    def build_instance(
        self,
        tasks: List[MTLTask],
        epoch: int,
        task_step: Callable[[Any, MTLTask], Tuple[torch.FloatTensor, StepFeedbackDict]],
        shared_blocks: List[SharedBlock],
        prefix: str,
        step_counters: Dict[str, int],
        optim: MTLOptimizer,
    ) -> Loop:
        raise NotImplementedError


class TrainLoopConfig(LoopConfig):
    task_sampler: Annotated[TaskSamplerConfig, Field(discriminator="sampler_type")] = CyclicTaskSampler.Config(
        mode="break_with_longest_loader"
    )

    def build_instance(self, tasks: List[MTLTask], *args, **kwargs) -> Loop:
        task_sampler = TaskSamplerTypes[self.task_sampler.sampler_type](self.task_sampler, tasks, DataSplit.train)
        multistep_num = self._compute_multistep_num(len(tasks))

        return Loop(self, task_sampler, multistep_num, True, *args, **kwargs)


class ValLoopConfig(LoopConfig):
    task_sampler: Annotated[TaskSamplerConfig, Field(discriminator="sampler_type")] = BalancedTaskSampler.Config()
    multistep_mode: Annotated[MultistepMode, Field(discriminator="multistep_type")] = FixedMultistep(step_num=1)

    def build_instance(self, tasks: List[MTLTask], *args, **kwargs) -> Loop:
        task_sampler = TaskSamplerTypes[self.task_sampler.sampler_type](self.task_sampler, tasks, DataSplit.val)
        # multistep_num = self._compute_multistep_num(len(tasks))

        return Loop(self, task_sampler, -1, False, *args, **kwargs)


class Loop:
    """
    Loops through a tasksampler.

    - prefix is the name of the loop. Controls the step which is logged.
    """

    def __init__(
        self,
        args: LoopConfig,
        task_sampler: BaseSampler,
        accumulate_losses_for_steps: int,
        training_mode: bool,
        epoch: int,
        task_step: Callable,
        shared_blocks: SharedModules,
        prefix: str,
        step_counters: Dict[str, int],
        optim: Optional[MTLOptimizer] = None,
        rank: int = 0,
        syncon_syncoff: Optional[Tuple[Callable, Callable]] = None,
        eventqueue: str = "",
    ) -> None:
        self.args: LoopConfig = args
        self.epoch: int = epoch
        self.task_step = task_step
        self.training_mode: bool = training_mode
        self.task_sampler: BaseSampler = task_sampler
        self.shared_blocks = shared_blocks
        self.prefix: str = prefix
        self.step_counters: Dict[str, int] = step_counters
        self.optim = optim
        self.accumulate_losses_for_steps = accumulate_losses_for_steps
        self.rank = rank
        self.one_update_per_task = self.accumulate_losses_for_steps == len(self.task_sampler.tasks)
        self.eventqueue: str = eventqueue
        if syncon_syncoff is None:
            self.sync_on, self.sync_off = None, None
        else:
            self.sync_on, self.sync_off = syncon_syncoff

        if self.args.log_args.log_result:
            printout = f"Loop {self.prefix}, rank {self.rank}: {[t.get_name() for t in self.task_sampler.tasks]} "
            if training_mode:
                printout += f"Accu.-steps: {self.accumulate_losses_for_steps}. "
                printout += f"Tasks reduced each step: {self.one_update_per_task=}"
            logging.info(printout)

    def drain(self) -> List[float]:
        return [x for _, x in self]

    def drain_to_dict(self) -> dict[MTLTask, list[float]]:
        res: dict[MTLTask, list[float]] = {t: [] for t in self.task_sampler.tasks}
        for task, task_loss in self:
            res[task].append(task_loss)
        return res

    def has_len(self) -> bool:
        if self.args.max_steps > 0:
            return True
        else:
            return self.task_sampler.is_finite()

    def __len__(self) -> int:
        if self.task_sampler.is_finite():
            return (
                min(self.args.max_steps, len(self.task_sampler)) if self.args.max_steps > 0 else len(self.task_sampler)
            )
        else:
            assert self.args.max_steps > 0, "Using infinite task samplers require a max number of steps per loop."
            return self.args.max_steps

    def _prepare_loop(self):
        # Make sure the user did not forget to make their model MTL compatible
        for shared_block in self.shared_blocks.module.shared_modules.values():  # type: ignore
            shared_block: SharedBlock
            shared_block.prepare_epoch(self.epoch, self.prefix, training_mode=self.training_mode)
            assert shared_block._made_mtl_compatible, f"{shared_block} not yet MTL compatible"

        frozen_blocks = [f"{m.get_name()}: {m.training=}" for m in self.shared_blocks.module.shared_modules.values() if not m.training == self.training_mode]  # type: ignore
        if frozen_blocks:
            logging.info(f"Assuming frozen blocks, because loop is in {self.training_mode=}: {frozen_blocks}")

        for task in self.task_sampler.tasks:
            assert task is not None
            task.prepare_epoch(self.epoch, self.prefix, training_mode=self.training_mode)
            assert (
                not task.is_currently_being_trained()
            ), f"Task {task.get_name()} thinks it is already being trained before the loop"

        if self.prefix not in self.step_counters:
            self.step_counters[self.prefix] = 0

        if self.training_mode:
            # Stuff like optimizer.zero_grad
            self.optim.reset_shared_grads()

        if dist.is_initialized() and self.sync_off is not None:
            self.sync_off()

    def _finish_up(self):
        logging.info(
            f"{datetime.now():{self.args.log_args.print_dateformat}}:"
            + f" {self.prefix} done with: {self._get_progbar_desc()}"
        )
        log_dict = {f"counter/{self.prefix}epoch": self.epoch}
        for task in self.task_sampler.tasks:
            task_res_dict, task_print_str = task.log_epoch_metrics()
            logging.info(task_print_str)
            for metric_name, metric_val in task_res_dict.items():
                log_dict[f"{task.get_name()}/{self.prefix}epoch.{metric_name}"] = metric_val

        if self.training_mode:
            optim_state: Dict[str, Any] = self.optim.get_current_logstate()
            for k, v in optim_state.items():
                log_dict[f"optimizer/{self.prefix}_{k}"] = v

        if self.args.log_args.log_result:
            wandb.log(log_dict)

    def _get_progbar_desc(self) -> str:
        task_statuses = [t.get_short_status() for t in self.task_sampler.tasks]
        return f"p{self.rank}_{self.prefix}_e{self.epoch}: {', '.join(task_statuses)}"

    def __iter__(self):
        try:
            import redis

            redis.Redis.from_url(mtl_settings.redis_url).ping()
            db_available = True
        except Exception as e:
            db_available = False
            logfire.info("Not publishing loop stats to DB", exception=e)
        self._prepare_loop()

        iteration_start_time, is_update_step = time.perf_counter(), False

        # assert self.task_sampler.is_finite() or self.args.max_steps > 0, \
        #     f"Either use a finite task sampler or specify the maximum number of steps in {self}"
        task_iterator: Iterator[Tuple[Any, MTLTask]] = self.task_sampler.continue_iter()

        def do_update_step() -> None:
            self.optim.after_iteration_step()
            self.optim.reset_shared_grads()

        with torch.set_grad_enabled(self.training_mode):
            with tqdm(
                total=self.__len__() if self.has_len() else None,
                leave=self.args.log_args.log_result,
                position=self.rank,
                disable=not self.args.log_args.progress_bar,
            ) as pbar:
                for i, (batch, task) in enumerate(task_iterator):
                    if (batch_extraction_s := time.perf_counter() - iteration_start_time) > 1.0:
                        logfire.warning(
                            "Task {task_name} on rank {rank} blocked for {batch_extraction_s:.2f} s in iteration {i}",
                            task_name=task.get_name(),
                            rank=self.rank,
                            batch_extraction_s=batch_extraction_s,
                            i=i,
                        )
                    mtl_dataset = task.cohort.datasets[self.task_sampler.loader_index.value]
                    if db_available and i != 0:  # the first iteration often hangs
                        with mtl_settings.kv.pipeline() as pipe:
                            k = mtl_dataset.loader_info.get_cache_key("batch_extraction_s")
                            pipe.lpush(k, batch_extraction_s)
                            pipe.ltrim(k, 0, 100)

                            pipe.execute()

                    is_update_step = (self.has_len() and (i >= self.__len__() - 1)) or (
                        i % self.accumulate_losses_for_steps == (self.accumulate_losses_for_steps - 1)
                    )  # either the last step in the loop or an update step
                    is_update_step = self.training_mode and is_update_step  # Do update steps only during training

                    assert task.training == self.training_mode
                    # Set active task for all shared blocks
                    for block in self.shared_blocks.module.shared_modules.values():  # type: ignore
                        block: SharedBlock
                        block.set_active_task(task.get_name())

                    if self.training_mode and is_update_step and (self.sync_on is not None):
                        # the forward step needs to know if a sync will happen, so turn on sync here already
                        self.sync_on()

                    with logfire.span(
                        "Running step for {task_name} on rank {rank}", task_name=task.get_name(), rank=self.rank
                    ):
                        try:
                            with self.optim.forward_context if self.training_mode else nullcontext():
                                task_step_result = self.task_step(batch, task)
                        except Exception as e:
                            if "out of memory" in str(e):
                                logfire.error(
                                    "Task {task_name} on rank {self.rank} encountered {e}, reduce size of {batch}!",
                                    task_name=task.get_name(),
                                )
                                raise e
                            else:
                                logfire.error(
                                    f"Encountered exception {e} with task {task.get_name()} on rank {self.rank}!"
                                )
                                raise e

                    if task_step_result is not None:
                        task_step_loss, task_log_dict = task_step_result

                        if torch.isnan(task_step_loss).item():
                            raise Exception(f"Got a nan loss for \n{batch}\n in task: \n {task}")

                        if self.training_mode:
                            # Updates the task head if there is exactly one update per task in each iteration
                            self.optim.run_backward(task_step_loss)
                            self.optim.after_backward(of_task=task, reduce_task_immediately=self.one_update_per_task)
                            if self.sync_off is not None:
                                self.sync_off()

                        if is_update_step:
                            do_update_step()

                        network_passes_s = time.perf_counter() - iteration_start_time - batch_extraction_s

                        step_prefix = f"{self.prefix}step"
                        task_uniqueness = f"_{task.get_name()}"
                        step_log_dict = {
                            f"counter/{self.prefix}epoch": self.epoch,
                            f"counter/{step_prefix}": self.step_counters[self.prefix],
                            f"{step_prefix}loss/batch_loss{task_uniqueness}": task_step_loss.item(),
                        }

                        if is_update_step and self.optim.scaler.get_scale() != 1:
                            step_log_dict["general/grad_scale"] = self.optim.scaler.get_scale()

                        if self.args.log_args.log_result:
                            task_log_dict = {
                                f"{task.get_name()}/{step_prefix}.{k}": v for k, v in task_log_dict.items()
                            }
                            step_log_dict.update(task_log_dict)

                        step_log_dict.update(
                            {
                                f"{step_prefix}time/batchextraction_time{task_uniqueness}": batch_extraction_s,
                                f"{step_prefix}time/networkpasses_time{task_uniqueness}": network_passes_s,
                                f"{step_prefix}time/fulliteration_time{task_uniqueness}": network_passes_s
                                + batch_extraction_s,
                            }
                        )

                        if self.args.log_args.include_mem.is_event(self.epoch):
                            mem_pref = f"{step_prefix}memory/"
                            step_log_dict[
                                f"{mem_pref}allocated{task_uniqueness}"
                            ] = torch.cuda.max_memory_allocated() / (1024**2)
                            torch.cuda.reset_peak_memory_stats()

                        wandb.log(step_log_dict)
                        if self.eventqueue:
                            mtl_settings.kv.publish(
                                channel=self.eventqueue,
                                message=TrainingStepEvent(
                                    loss=task_step_loss.item(),
                                ).model_dump_json(),
                            )

                        yield task, task_step_loss.item()
                    else:
                        logging.warning(f"Task {task.get_name()} wants to skip step {i}")

                    self.step_counters[self.prefix] += 1
                    iteration_start_time = time.perf_counter()
                    pbar.update(1)
                    pbar.set_description(self._get_progbar_desc())

                    if self.has_len() and (i >= self.__len__() - 1):
                        break

            if (not is_update_step) and self.training_mode:
                assert self.sync_off is None and self.sync_on is None, "Multi GPU not compatible with sudden loop end"
                # If the last step in the loop was not an update step there is an unused gradient
                do_update_step()

        self._finish_up()

    def __repr__(self) -> str:
        return f"""Loop with train: {self.training_mode}\n
        With Args:\n{json.dumps(self.args.dict(), indent=2)}\n
        For tasks: {[t.get_name() for t in self.task_sampler.tasks]}"""
