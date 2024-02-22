from __future__ import annotations
import os
from copy import deepcopy, copy
import traceback
import wandb
import json
import logging
from pathlib import Path
from pydantic import Field

from typing_extensions import Annotated
from typing import Any, Callable, Dict, List, Optional, Iterable, Tuple, Literal

import itertools

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from mmm.mtl_modules.MTLModule import MTLModule
from mmm.optimization.MTLOptimizer import MTLOptimizer
from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.mtl_modules.tasks.TaskModule import TaskModule
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.trainer.Loop import Loop
from mmm.utils import remove_folder_blocking_if_exists, recursive_equality
from mmm.trainer.TaskPurpose import TaskPurpose
from mmm.DataSplit import DataSplit
from .CallbackType import CallbackType
from mmm.event_selectors import (
    EventSelector,
    FixedEventSelector,
    RecurringEventSelector,
)
from mmm.BaseModel import BaseModel
from mmm.optimization.MTLOptimizer import MTLOptimizer
from mmm.trainer.Loop import TrainLoopConfig, ValLoopConfig

import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel


class EarlyStoppingConfig(BaseModel):
    """
    Groups setting which concern early stopping.
    """

    early_stopping_patience: int = Field(
        default=10,
        description="Number of downstream training loops with no improvement after which training will be stopped. "
        + "Criterion checking for early stopping starts after 2*patience loops.",
    )
    improvement_delta: float = Field(
        default=0.03,
        description="""
Minimum loss improvement to count as an improvement for the patience attribute of early stopping.
If early stopping is used, the trainer saves a checkpoint "earlystop" after each best loop.
""",
    )
    min_train_loops: int = 5
    criterion: Literal["trainloss", "valloss"] = "trainloss"


class MTLTrainer:
    class Config(BaseModel):
        train_device: Literal["cpu", "cuda"] = "cuda"
        max_epochs: int = Field(
            default=150,
            description="Maximum number of epochs. Can be overriden by fit().",
        )

        checkpoint_cache_folder: Path = Field(
            default_factory=lambda: Path(os.getenv("ML_DATA_OUTPUT", default="./")) / "trainer_checkpoints",
            description="""
The parent folder where the trainer will store checkpoints.
            """,
        )

        optim: MTLOptimizer.Config = Field(
            default=MTLOptimizer.Config(),
            description="If set, takes precedence over all tasks and shared block's optimizers.",
        )

        mtl_train_selector: Annotated[EventSelector, Field(discriminator="selector_type")] = RecurringEventSelector(
            every_n=1, starting_at=1
        )
        mtl_train_loop: TrainLoopConfig = TrainLoopConfig()

        mtl_validation_selector: Annotated[EventSelector, Field(discriminator="selector_type")] = (
            RecurringEventSelector(every_n=1)
        )
        mtl_val_loop: ValLoopConfig = ValLoopConfig()

        early_stopping: EarlyStoppingConfig | None = None

        keep_checkpoints_of_epochs: Annotated[EventSelector, Field(discriminator="selector_type")] = FixedEventSelector(
            at_iterations=[]
        )

        rebuild_dataloaders: Annotated[EventSelector, Field(discriminator="selector_type")] = FixedEventSelector(
            at_iterations=[0]
        )

    class State(BaseModel):
        epoch: int
        step_counters: dict[str, int]
        best_val_losses: list[float] | None = None
        avg_losses_per_epoch: list[tuple[float | None, float | None]] = []
        stage_index: int = 0

    def __init__(
        self,
        args: Config,
        experiment_name: str,
        global_rank: int = int(os.getenv("RANK", default=0)),
        local_rank: int = int(os.getenv("LOCAL_RANK", default=0)),
        local_world_size: int = int(os.getenv("LOCAL_WORLD_SIZE", default=1)),
        world_size: int = int(os.getenv("WORLD_SIZE", default=1)),
        clear_checkpoints: bool = False,
    ):
        self.experiment_name = experiment_name
        self.rank, self.world_size = global_rank, world_size
        self.local_rank, self.local_world_size = local_rank, local_world_size
        self.mtl_tasks: List[MTLTask] = []
        # self.eval_tasks: List[MTLTask] = []
        self.args: MTLTrainer.Config = args
        self.stages = [""]

        # Not a nn.ModuleDict, because the trainer is not an nn.Module
        self.shared_blocks: Dict[str, SharedBlock] = {}
        self.ddp_model = None

        self.state = self.State(epoch=0, step_counters={}, best_val_losses=None, avg_losses_per_epoch=[])

        self.experiment_ckpt_folder = self.args.checkpoint_cache_folder / self.experiment_name
        if self.rank == 0:
            if clear_checkpoints:
                remove_folder_blocking_if_exists(self.experiment_ckpt_folder)
            if not self.experiment_ckpt_folder.exists():
                self.experiment_ckpt_folder.mkdir(parents=True)
                logging.info(f"Created trainer's cache folder: {self.experiment_ckpt_folder}")
        if dist.is_initialized():
            dist.barrier()

        self.mtl_optimizer = None
        self.callbacks: Dict[CallbackType, Dict[str, Callable]] = {cb_type: {} for cb_type in CallbackType}

    def __repr__(self) -> str:
        res = f"Trainer config: {self.args.model_dump()}\n"
        res += f"Trainer state: {self.state}\n"
        res += f"{self.rank=}, {self.local_rank=}, {self.world_size=}, {self.local_world_size=}\n"
        res += f"Tasks used for pretraining:{[t.get_name() for t in self.mtl_tasks]}\n"
        return res

    def get_shared_block(self, n: str) -> SharedBlock:
        return self.shared_blocks[n]

    def add_shared_block(self, v: SharedBlock) -> None:
        if torch.cuda.is_available():
            self.shared_blocks[v.args.module_name] = v.set_device(self.args.train_device)
        else:
            logging.warning("CUDA is not available, using CPU instead.")
            self.shared_blocks[v.args.module_name] = v

    def add_shared_blocks(self, vs: Iterable[SharedBlock]) -> MTLTrainer:
        for shared_block in vs:
            self.add_shared_block(shared_block)
        return self

    def add_mtl_task(self, t: MTLTask):
        for task in self.mtl_tasks:
            if task.get_name() == t.get_name():
                raise Exception(f"{t.get_name()} is already known to the trainer.")

        if torch.cuda.is_available():
            self.mtl_tasks.append(t.set_device(self.args.train_device))
        else:
            logging.warning("CUDA is not available, using CPU instead.")
            self.mtl_tasks.append(t)

        for block in self.shared_blocks.values():
            block.prepare_for_task(t.get_name())

        return t

    def get_task_by_name(self, task_name: str) -> MTLTask | None:
        for t in self.mtl_tasks:
            if t.get_name() == task_name:
                return t
        return None

    def callback_after_each_epoch(self, shared_model_train_mode=False, use_inference_mode=True):
        """
        The trainer will call the decorated function each epoch.
        """

        # Python calls this with the function `f` of the user when defining `f`
        def setup_callback(cb: Callable[[MTLTrainer], None]) -> Callable[[MTLTrainer], None]:
            logging.info(f"Trainer will call {cb.__name__} after each epoch.")
            logging.info(f"It will apply {shared_model_train_mode=} to all MTLModules")
            logging.info(f"It will set the default task for all task-specific parts of shared blocks")
            logging.info(f"If you want to use the task-specific parts of a task, use `set_active_task(...)`")
            logging.info("It will wrap your callback in torch's inference mode:")

            def wrapper(trainer: MTLTrainer) -> None:
                for m in self.mtl_tasks:  # + self.eval_tasks:  # type: ignore
                    m.prepare_epoch(
                        self.state.epoch,
                        f"cb_{cb.__name__}",
                        training_mode=shared_model_train_mode,
                    )
                if self.ddp_model is not None:
                    self.ddp_model.train(shared_model_train_mode)

                for shared_block in self.shared_blocks.values():
                    shared_block.set_active_task("original")

                with torch.inference_mode(mode=use_inference_mode):
                    return cb(trainer)

            self.callbacks[CallbackType.each_epoch][cb.__name__] = wrapper
            return wrapper

        return setup_callback

    def get_task_module(self, task_name: str) -> TaskModule:
        """
        Builds a task module for the given task name, wrapping its dependencies from the shared blocks.

        If the shared blocks have task-specific parts, they will be set to the exported task.
        """
        task = self.get_task_by_name(task_name)
        if task is None:
            raise Exception(f"Task {task_name} is not known to the trainer.")
        export_module = TaskModule(task, self.shared_blocks)
        for shared_module in export_module.shared_modules.values():  # type: ignore
            shared_module: SharedBlock
            shared_module.set_active_task(task.get_name())
        return export_module

    @torch.no_grad()
    def save_blocks_native(self, filepath: Path, only_inference: bool = True):
        """
        Wraps all shared blocks and tasks into a dictionary.

        If only_inference is True, the model is set to evaluation mode and the cohort objects are removed.

        This export is suitable for machine to machine exports that happen quickly and on the same codebase.
        If the mmm code is slightly changed, the exported file will break.
        In that case, use the ONNX export.
        """
        exportdict = nn.ModuleDict(self.shared_blocks)

        for exporttask in self.mtl_tasks:
            if only_inference:
                # A shallow copy is enough, because we do not modify the deeper structures
                exporttask = copy(exporttask)
                exporttask.cohort = None
            exportdict[exporttask.get_name()] = exporttask
        if only_inference:
            exportdict = exportdict.eval().cpu()
        torch.save(exportdict, filepath)
        return exportdict

    @torch.no_grad()
    def save_task_native(
        self,
        task_name: str,
        file_path: Path | None = None,
        only_inference: bool = False,
    ):
        """
        If file_path is none, will save to ./task_name.pt

        The model can be used like this:

        ```
        # Load the model using PyTorch native saving
        export_task_id = "taskname"
        model = torch.load(f"{export_task_id}.pt")
        model = model.eval()

        with torch.inference_mode():
            model(inputbatch: torch.Tensor)
        ```

        Disadvantage of this (from PyTorch docs):

        The disadvantage of this approach is that the serialized data is bound to the specific classes
        and the exact directory structure used when the model is saved.
        The reason for this is because pickle does not save the model class itself.
        Rather, it saves a path to the file containing the class, which is used during load time.
        Because of this, your code can break in various ways when used in other projects or after refactors.
        """
        if file_path is None:
            file_path = Path(f"./{task_name}.pt")

        export_module = self.get_task_module(task_name)

        try:
            # Workers cannot be pickled, kill them!
            export_module.task.cohort.terminate_workers()
        except Exception as e:
            logging.warning(f"Could not terminate workers: {e}")

        if only_inference:
            export_module.task = deepcopy(export_module.task)
            del export_module.task.cohort

        export_module.eval()

        torch.save(export_module, file_path)

    @torch.no_grad()
    def save_task_as_onnx(self, task_name: str, file_path: Path | None = None):
        """
        Given a task name (either a pretraining task or a downstream task), creates an onnx network.
        """
        if file_path is None:
            file_path = Path("./somenetwork.onnx")

        # ONNX import only relevant for this function, do not put at top to reduce startup time
        import torch.onnx

        task = self.get_task_by_name(task_name)
        assert task is not None
        onnx_net = TaskModule(task, self.shared_blocks)
        for shared_module in onnx_net.shared_modules.values():  # type: ignore
            shared_module: SharedBlock
            shared_module.set_active_task(task.get_name())
        example_x = onnx_net.task.cohort.get_onnx_input(f"cuda:{self.rank}")
        onnx_net.eval()

        torch.onnx.export(
            onnx_net,  # model being run
            example_x,  # model input (or a tuple for multiple inputs)
            str(file_path),  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=13,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            training=torch.onnx.TrainingMode.EVAL,
            dynamic_axes={
                "input": {0: "batch_size"},  # variable length axes
                "output": {0: "batch_size"},
            },
        )

    def create_checkpoint(self, prefix: str) -> Path:
        folder_path = self.experiment_ckpt_folder / f"{prefix}-{self.state.epoch}"
        if self.rank == 0:
            if folder_path.exists():
                logging.warning(f"Checkpoint {folder_path} already exists. Probably the writing was interrupted.")
            folder_path.mkdir(parents=True, exist_ok=True)

        # Make all processes wait until rank 0 has created the folder
        if dist.is_initialized():
            dist.barrier()

        logging.info(f"Saving checkpoint to {folder_path}")
        for k, v in self.shared_blocks.items():
            if self.rank == 0:
                v.save_checkpoint(folder_path / k)
            else:
                v.save_checkpoint(folder_path / f"{k}{self.rank}")

        # All checkpoints are created, now sanity check that all shared weights are identical
        if dist.is_initialized():
            dist.barrier()
            if self.rank == 0 and self.world_size > 1:
                for k, v in self.shared_blocks.items():
                    shouldbestate = str(
                        torch.load(
                            folder_path / k / "module.ckpt",
                            map_location=torch.device("cpu"),
                        )["model_state"]
                    )
                    for other_rank in range(0, dist.get_world_size()):
                        # Load the other state dict to cpu
                        pk = f"{k}{other_rank}" if other_rank != 0 else k
                        other_state = torch.load(
                            folder_path / pk / "module.ckpt",
                            map_location=torch.device("cpu"),
                        )
                        # if not recursive_equality(other_state, v.state_dict(), approx=True):
                        otherstate_str = str(other_state["model_state"])
                        if not otherstate_str == shouldbestate:
                            # Save both string to text files for debugging
                            with open(f"state_dict_{k}_rank0.txt", "w") as f:
                                f.write(shouldbestate)
                            with open(f"state_dict_{k}_rank{other_rank}.txt", "w") as f:
                                f.write(otherstate_str)
                            logging.error(f"Shared block {k} is not identical on rank 0 and rank {other_rank}")
                        else:
                            logging.debug(f"Shared block {k} is identical on rank 0 and rank {other_rank}")

        # Save the optimizer of rank 0
        if self.mtl_optimizer is None:
            logging.warning(f"Optimizer not initialized, initializing optimizer for: {folder_path}")
            self.init_optimizer()

        mtl_optim_folder = folder_path / "trainer_optim"
        self.mtl_optimizer.create_checkpoint(mtl_optim_folder, self.rank)

        # Always save the checkpoints of tasks assuming that different ranks each have unique tasks
        for task in self.mtl_tasks:
            task.save_checkpoint(folder_path / task.get_name())

        # Save the state last, this is the criterion to determine if a checkpoint is fully written
        if self.rank == 0:
            (folder_path / "meta.json").write_text(self.state.model_dump_json(indent=4))
        if dist.is_initialized():
            dist.barrier()
        return folder_path

    @staticmethod
    def verify_checkpoint(folder_path: Path) -> bool:
        try:
            with open(folder_path / "meta.json", "r+") as f:
                meta_dict = json.load(f)
                teststate = MTLTrainer.State(**meta_dict)
            return True
        except Exception as e:
            logging.info(f"Checkpoint {folder_path} is invalid due to {e}")
            return False

    def load_checkpoint(
        self,
        folder_path: Path,
        load_optim_state=True,
        load_meta=True,
        load_tasks=True,
    ) -> bool:
        """
        Uses an absolute folder path to enable loading checkpoints from other experiments.
        """
        if folder_path.exists():
            # First verify that the checkpoint is valid, we assume it is valid when the last thing was written:
            assert MTLTrainer.verify_checkpoint(folder_path), f"Checkpoint {folder_path} is invalid."

            shared_only_success = True
            for shared_module_name, shared_module in self.shared_blocks.items():
                try:
                    shared_module.load_checkpoint(folder_path / shared_module_name)
                    logging.info(f"Successfully loaded checkpoint of {shared_module_name}")
                except Exception as e:
                    logging.warning(f"Couldn't load checkpoint for shared block {shared_module_name} due to {e}")
                    shared_only_success = False

            if load_tasks:
                for task in self.mtl_tasks:
                    try:
                        if (folder_path / task.get_name()).exists():
                            task.load_checkpoint(folder_path / task.get_name())
                            logging.info(f"Successfully loaded checkpoint of {task.get_name()}")
                        else:
                            logging.warn(f"Couldn't find checkpoint of task {task.get_name()}")
                    except RuntimeError as e:
                        logging.warning(f"Couldn't load checkpoint for task {task.get_name()} due to {e}")
                    except ValueError as e:
                        logging.warning(f"Couldn't load checkpoint for task {task.get_name()} due to {e}")
                    except FileNotFoundError as e:
                        logging.warning(f"Couldn't find checkpoint for task {task.get_name()} due to {e}")

            if load_optim_state:
                try:
                    assert shared_only_success, "Cannot load optimizer state if there was a change to a shared block."
                    if self.mtl_optimizer is None:
                        self.init_optimizer()
                    self.mtl_optimizer.load_checkpoint(folder_path / "trainer_optim")
                except Exception as e:
                    logging.warn(f"Couldn't load optimizer checkpoint {folder_path / 'trainer_optim'} due to {e}")

            if load_meta:
                self.state = self.State(**json.loads((folder_path / "meta.json").read_text()))

            logging.info(f"Loaded checkpoint from {folder_path} with epoch {self.state.epoch}")
            return True
        else:
            logging.info(f"Checkpoint does not exist at {folder_path}")
            return False

    def _task_step(self, batch: Any, task: MTLTask):
        batch = task.prepare_batch(batch)
        return task.training_step(batch, self.ddp_model)

    def _controlsync(self, on: bool):
        self.ddp_model.require_backward_grad_sync = on  # type: ignore
        # for ddpmodel in self.ddp_model.values():
        #     ddpmodel.require_backward_grad_sync = on  # type: ignore

    def run_mtl_train_epoch(self) -> Dict[MTLTask, List[float]]:
        """
        A pretrain epoch's purpose is training the shared blocks.

        Multi-task pre-training is a typical example.
        """
        allocated_train_workers = sum([t.cohort.get_active_workers(DataSplit.train) for t in self.mtl_tasks])
        logging.info(f"{allocated_train_workers} are already allocated for training")
        if len(mp.active_children()) - allocated_train_workers > len(os.sched_getaffinity(0)) // 2:
            logging.info(f"Killing validation workers because too many workers would be allocated")
            for t in self.mtl_tasks:
                t.cohort.terminate_datasplit_workers(DataSplit.val)

        # With distributed training this property of shared blocks is not autoforwarded to DDP wrapper
        self.ddp_model.train(True)

        loop = self.args.mtl_train_loop.build_instance(
            self.mtl_tasks,
            self.state.epoch,
            self._task_step,
            self.ddp_model,
            f"mtltrain{self.stages[self.state.stage_index]}",
            self.state.step_counters,
            self.mtl_optimizer,
            rank=self.rank,
            syncon_syncoff=(
                (
                    lambda: self._controlsync(True),
                    lambda: self._controlsync(False),
                )
                if dist.is_initialized()
                else None
            ),
        )
        return loop.drain_to_dict()

    def run_mtl_val_epoch(self) -> Dict[MTLTask, List[float]]:
        """Validates the pre-training tasks"""
        allocated_val_workers = sum([t.cohort.get_active_workers(DataSplit.val) for t in self.mtl_tasks])
        logging.info(f"{allocated_val_workers} are already allocated for validation")
        if len(mp.active_children()) - allocated_val_workers > len(os.sched_getaffinity(0)) // 2:
            logging.info(f"Killing pretraining workers because too many workers would be allocated")
            for t in self.mtl_tasks:
                t.cohort.terminate_datasplit_workers(DataSplit.train)

        self.ddp_model.train(False)
        loop: Loop = self.args.mtl_val_loop.build_instance(
            self.mtl_tasks,
            self.state.epoch,
            self._task_step,
            self.ddp_model,
            f"mtlval{self.stages[self.state.stage_index]}",
            self.state.step_counters,
            None,  # No optimizer validation,
            rank=self.rank,
        )
        return loop.drain_to_dict()

    def check_workernum(self, auto_adjust=True):
        import math
        from numpy.random import choice
        import os

        def adjust(ls: List[int], max_num: int):
            ls = [math.ceil((x / sum(ls)) * max_num) for x in ls]

            # Now we should be really close but some tasks need to lose some workers. Take from the rich with a higher probab:
            while sum(ls) > max_num:
                i = choice(list(range(len(ls))), 1, p=[x / sum(ls) for x in ls])[0]
                ls[i] = max(0, ls[i] - 1)
            return ls

        workernum_available = len(os.sched_getaffinity(0)) // self.local_world_size

        # For training a safe number seems to be to use half the workers for pretraining and half for validation:
        workers = {task.args.module_name: task.cohort.args.num_workers for task in self.mtl_tasks}
        logging.info(f"{workernum_available} workers available for pretraining tasks. You requested {workers}")

        if auto_adjust and sum(workers.values()) > workernum_available:
            new_workers = adjust(list(workers.values()), workernum_available)
            for new_worker_num, task in zip(new_workers, self.mtl_tasks):
                task.cohort.args.num_workers = max(new_worker_num, 1)
            after_adjustment_workers = {task.args.module_name: task.cohort.args.num_workers for task in self.mtl_tasks}
            logging.info(f"Workers changed to {after_adjustment_workers}")
        else:
            assert sum(workers.values()) <= workernum_available, (
                f"You requested {workers} workers, "
                + f"but your environment should only use {workernum_available} workers in sum."
            )

    def prepare_dataloading(self) -> None:
        # In some environments, requesting more workers than there are CPU cores results in an error:
        self.check_workernum()

        for task in self.mtl_tasks:
            if self.args.rebuild_dataloaders.is_event(self.state.epoch):
                task.cohort.terminate_workers()
                task.cohort.data_loaders = (None, None)
            if None in task.cohort.data_loaders:
                task.cohort.prepare_epoch(epoch=self.state.epoch)

    def run_mtl_epoch(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Returns None if no validation is performed. Otherwise, returns the average validation loss for each task.
        """
        self.prepare_dataloading()

        # Train the shared blocks and task heads:
        if self.args.mtl_train_selector.is_event(self.state.epoch):
            # Always unfreeze shared blocks for mtl epoch
            train_losses_by_task: Dict[MTLTask, List[float]] = self.run_mtl_train_epoch()
            train_result = [task_losses for _, task_losses in train_losses_by_task.items()]
        else:
            train_result = None

        if self.args.mtl_validation_selector.is_event(self.state.epoch):
            # Validate pre-training tasks
            val_losses_by_task: Dict[MTLTask, List[float]] = self.run_mtl_val_epoch()
            val_result: Optional[List[List[float]]] = [task_losses for _, task_losses in val_losses_by_task.items()]
        else:
            val_result = None

        if train_result is not None:
            self.mtl_optimizer.after_loop_step(self.state.epoch, train_result, val_result)
        else:
            logging.warn(f"Without training, no after_loop_step is performed ({self.state.epoch=})")

        train_avg = (
            [sum(task_losses) / len(task_losses) for task_losses in train_result] if train_result is not None else None
        )
        val_avg = (
            [sum(task_losses) / len(task_losses) for task_losses in val_result] if val_result is not None else None
        )
        if train_avg is not None:
            avg_train_loss = sum(train_avg) / len(train_avg)
        else:
            avg_train_loss = None
        if val_avg is not None:
            avg_val_loss = sum(val_avg) / len(val_avg)
        else:
            avg_val_loss = None
        self.state.avg_losses_per_epoch.append((avg_train_loss, avg_val_loss))
        return train_avg, val_avg

    def init_optimizer(self) -> None:
        self.ddp_model: SharedModules = SharedModules(self.shared_blocks)

        if dist.is_initialized():
            # find_unused_parameters is required whenever we have a shared block that is not used by all tasks
            self.ddp_model = DistributedDataParallel(
                self.ddp_model,
                device_ids=[int(self.local_rank)],
                find_unused_parameters=True,
            )  # type: ignore

        self.mtl_optimizer = MTLOptimizer(
            self.shared_blocks,
            self.args.optim,
            self.args.max_epochs,
            # Scale the gradients to account for DDP averaging gradients
            # While the docs say this is only relevant for the number of nodes:
            # "When a model is trained on M nodes with batch=N,
            # the gradient will be M times smaller
            # when compared to the same model trained on a single node with batch=M*N"
            # Other sources say that the gradient is averaged over all processes.
            gradient_scale_factor=float(self.world_size),
        )
        for task in self.mtl_tasks:
            self.mtl_optimizer.add_task(task)

    def get_summary_stats(self) -> Dict[str, Any]:
        module_summary = {}
        blocksummary: str = ""
        for blockname, block in self.shared_blocks.items():
            blocksummary = f"{blocksummary}<h2>{blockname}</h2>{block.__repr_html__()}"
        module_summary["sharedblocks"] = wandb.Html(blocksummary)
        for pretrain_module in self.mtl_tasks:
            module_summary[pretrain_module.get_name()] = wandb.Html(pretrain_module.__repr_html__())

        return module_summary

    def _fit(self, max_epochs: int):
        assert self.mtl_tasks, "No tasks added to trainer. Use `add_mtl_task` to add tasks."
        if self.state.epoch == 0:
            logging.info("Starting a new training. Logging module's stats")
            wandb.log({f"report/{k}": v for k, v in self.get_summary_stats().items()})
        elif self.state.epoch > max_epochs:
            logging.info(f"{max_epochs=} reached because {self.state.epoch=}")
            return
        else:
            logging.info(f"Resuming training at epoch {self.state.epoch=} until {max_epochs=}")

        for _ in itertools.count():
            self._run_callbacks(CallbackType.each_epoch, self)
            if self.rank == 0:
                self.cleanup_checkpoints()
            if dist.is_initialized():
                dist.barrier()
            with torch.no_grad():
                _, val_result = self.run_mtl_epoch()
                self.state.epoch += 1
                # This needs to be synchronized across processes to enable multi-GPU smart checkpointing

                if val_result is not None and (
                    self.state.best_val_losses is None or sum(val_result) < sum(self.state.best_val_losses)
                ):
                    self.state.best_val_losses = val_result

                    if not dist.is_initialized():
                        self.create_checkpoint("bestbyvalidation")

            # Increment global epoch counter which is used for logging and is also checkpointed

            self.create_checkpoint("latest")

            if self.state.epoch > max_epochs:
                logging.info(f"Reached max_epochs ({max_epochs}). Stopping trainer.fit().")
                break

            if self.args.early_stopping is not None:
                recently_reported: List[Optional[float]] = [
                    t[0] if self.args.early_stopping.criterion == "trainloss" else t[1]
                    for t in self.state.avg_losses_per_epoch
                ]
                if recently_reported[-1] is None:
                    logging.warning(f"Epoch {self.state.epoch=} did not log a loss and is ignored in early stopping.")
                recent_losses: List[float] = list(filter(lambda x: x is not None, recently_reported))  # type: ignore
                if recent_losses:
                    if recently_reported[-1] is not None and min(recent_losses) == recent_losses[-1]:
                        self.create_checkpoint("earlystop")

                    if len(recent_losses) >= self.args.early_stopping.min_train_loops:
                        # Check if the last significant improvement was more than early_stopping_patience epochs ago
                        first_epoch_with_current_loss = min(
                            [
                                i
                                for i, l in enumerate(recent_losses)
                                if (l - self.args.early_stopping.improvement_delta) < recent_losses[-1]
                            ]
                        )
                        num_loops_without_improvement = len(recent_losses) - first_epoch_with_current_loss
                        if num_loops_without_improvement > self.args.early_stopping.early_stopping_patience:
                            logging.info(
                                f"Epoch {first_epoch_with_current_loss} already had a similar loss as {self.state.epoch=}. "
                                "Breaking training now. "
                                f"Relevant losses: {[f'{l:.2f}' for l in recent_losses]}"
                            )
                            break
                        else:
                            logging.info(
                                f"Relevant losses for early stopping: {[f'{l:.2f}' for l in recent_losses]}. "
                                f"Continuing because {num_loops_without_improvement=} "
                                f"is smaller than the patience of {self.args.early_stopping.early_stopping_patience}"
                            )

    def _run_callbacks(self, callback_type: CallbackType, *args, **kwargs):
        for cb_name, cb in self.callbacks[callback_type].items():
            logging.info(f"Running hook {cb_name} of type {callback_type}")
            cb(*args, **kwargs)

    def cleanup_checkpoints(self):
        """
        For every prefix, it only keeps checkpoints that are either the most recent of their prefix,
        or are marked to be saved in self.args.
        """
        checkpoints_and_states = [
            (p, self.State(**json.loads((p / "meta.json").read_text())))
            for p in self.experiment_ckpt_folder.iterdir()
            if MTLTrainer.verify_checkpoint(p)
        ]
        marked_for_removal = []
        for checkpoint_path, checkpointstate in checkpoints_and_states:
            if self.args.keep_checkpoints_of_epochs.is_event(checkpointstate.epoch):
                continue

            other_checkpoints_with_prefix = [
                (p, state)
                for p, state in checkpoints_and_states
                if p.name.startswith(checkpoint_path.name.split("-")[0])
            ]
            their_epochs = [state.epoch for _, state in other_checkpoints_with_prefix]
            if checkpointstate.epoch >= max(their_epochs):
                continue

            marked_for_removal.append(checkpoint_path)

        for remove_folder in marked_for_removal:
            logging.info(f"Removing checkpoint {remove_folder}")
            remove_folder_blocking_if_exists(remove_folder)
        return marked_for_removal

    @staticmethod
    def get_recent_checkpoint_paths(parent_folder: Path, prefix: str) -> List[tuple[Path, dict]]:
        # Determine the newest checkpoint and load from that
        checkpoints_and_states = [
            (p, MTLTrainer.State(**json.loads((p / "meta.json").read_text())))
            for p in parent_folder.iterdir()
            if MTLTrainer.verify_checkpoint(p) and p.name.startswith(prefix)
        ]
        return checkpoints_and_states

    def get_recent_checkpoint_path(self, prefix: str) -> Path | None:
        """
        Crawls through all checkpoints with that prefix and returns the Path of the checkpoint with the highest epoch.
        """
        checkpoints_and_states = self.get_recent_checkpoint_paths(self.experiment_ckpt_folder, prefix)
        if not checkpoints_and_states:
            return None
        checkpoint_epochs = [state.epoch for _, state in checkpoints_and_states]
        most_recent_index = checkpoint_epochs.index(max(checkpoint_epochs))
        return checkpoints_and_states[most_recent_index][0]

    def fit(self, until_epoch=None):
        """
        Runs a training. If not interactive it will create emergency checkpoints when an error occurs.

        If fit_for_epochs is set, it will finish if either args.max_epochs is reached or fit_for_epochs.
        """
        if self.mtl_optimizer is None:
            self.init_optimizer()
        if (latest_checkpoint_path := self.get_recent_checkpoint_path("latest")) is not None:
            self.load_checkpoint(
                latest_checkpoint_path,
                load_optim_state=True,
                load_meta=True,
                load_tasks=True,
            )
        return self._fit(until_epoch if until_epoch is not None else self.args.max_epochs)
