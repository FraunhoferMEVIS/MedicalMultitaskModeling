from __future__ import annotations
from abc import abstractmethod
import os
import logging

from pathlib import Path
from typing import Dict, List, Optional, Any, Literal, Union
from typing_extensions import Annotated
from mmm.BaseModel import BaseModel
from pydantic import Field

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist

from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.mtl_modules.tasks.MTLTask import MTLTask


TorchLRScheduler = Union[optim.lr_scheduler._LRScheduler, ReduceLROnPlateau]


class SchedulerConfig(BaseModel):
    @abstractmethod
    def build_instance(self, optim: optim.Optimizer, max_epochs: int) -> TorchLRScheduler:
        raise NotImplementedError


class PolySchedulerConfig(SchedulerConfig):
    scheduler_type: Literal["poly"] = "poly"
    exponent: float = 0.9
    last_epoch: int = -1

    def build_instance(self, optim: optim.Optimizer, max_epochs: int) -> optim.lr_scheduler._LRScheduler:
        initial_lrs = [p["lr"] for p in optim.param_groups]

        def poly_lr(epoch, initial_lr):
            return initial_lr * (1 - epoch / max_epochs + 1) ** self.exponent

        return LambdaLR(
            optim,
            lr_lambda=[lambda e: poly_lr(e, lr) for lr in initial_lrs],
            last_epoch=self.last_epoch,
        )


class ExponentialLRConfig(SchedulerConfig):
    scheduler_type: Literal["exp"] = "exp"
    gamma: float = 0.9

    def build_instance(self, optim: optim.Optimizer, max_epochs: int) -> optim.lr_scheduler._LRScheduler:
        return ExponentialLR(optim, gamma=self.gamma, last_epoch=-1)


class ReduceLROnPlateauConfig(SchedulerConfig):
    scheduler_type: Literal["reduceonplateau"] = "reduceonplateau"
    factor: float = 0.1
    patience: int = 4
    threshold: float = 1e-4
    cooldown: int = 0
    min_lr: float = 1e-7
    eps: float = 1e-8

    def build_instance(self, optim: optim.Optimizer, max_epochs: int) -> ReduceLROnPlateau:
        return ReduceLROnPlateau(
            optim,
            mode="min",
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
            cooldown=self.cooldown,
            min_lr=self.min_lr,
            eps=self.eps,
        )


class DecaySchedulerConfig(SchedulerConfig):
    """
    Config of:
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
    """

    scheduler_type: Literal["step"] = "step"
    factor: float = 0.98
    min_lr: float = 1e-6
    last_epoch: int = -1

    def build_instance(self, optim: optim.Optimizer, max_epochs: int) -> optim.lr_scheduler._LRScheduler:
        def f(epoch):
            r = self.factor**epoch
            return r if r > self.min_lr else self.min_lr

        return LambdaLR(optim, lr_lambda=f, last_epoch=self.last_epoch)


class CosineAnnealingLRSchedulerConfig(SchedulerConfig):
    """
    Config of
    OptimizerType = Union[OptimizerAdamConfig, OptimizerAdamWConfig, OptimizerSGDConfig]
    """

    scheduler_type: Literal["cosine_annealing"] = "cosine_annealing"
    eta_min: float = 1e-6
    last_epoch: int = -1
    cycle_size: int = 5

    def build_instance(self, optim: optim.Optimizer, max_epochs: int) -> optim.lr_scheduler._LRScheduler:
        return CosineAnnealingLR(
            optim,
            T_max=self.cycle_size,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch,
        )


SchedulerType = Union[
    PolySchedulerConfig,
    ExponentialLRConfig,
    ReduceLROnPlateauConfig,
    DecaySchedulerConfig,
    CosineAnnealingLRSchedulerConfig,
]


class OptimizerConfig(BaseModel):
    lr: float = 1e-4

    def get_type(self):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError


class OptimizerAdamWConfig(OptimizerConfig):
    optim_type: Literal["adamw"] = "adamw"
    weight_decay: float = 1e-2

    def get_type(self):
        return optim.AdamW

    def get_params(self):
        return dict(lr=self.lr, weight_decay=self.weight_decay)


class OptimizerSGDConfig(OptimizerConfig):
    optim_type: Literal["sgd"] = "sgd"
    momentum: float = 0.8
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = True
    lr: float = 2.5e-3

    def get_type(self):
        return optim.SGD

    def get_params(self):
        return dict(
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            dampening=self.dampening,
            nesterov=self.nesterov,
        )

    def build_instance(self, params) -> optim.Optimizer:
        return optim.SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            dampening=self.dampening,
            nesterov=self.nesterov,
        )


OptimizerType = Union[OptimizerAdamWConfig, OptimizerSGDConfig]


class MTLOptimizer:
    class Config(BaseModel):
        optim_config: Annotated[OptimizerType, Field(discriminator="optim_type")] = OptimizerAdamWConfig()
        lr_scheduler_configs: List[Annotated[SchedulerType, Field(discriminator="scheduler_type")]] = []
        lr_factor_tasks: float = Field(
            default=1.0,
            description="Scales the learning rate of the task optimizers relative to the global learning rate.",
        )

    def __init__(
        self,
        shared_blocks: SharedModules,
        args: MTLOptimizer.Config,
        max_epochs: int,
        gradient_scale_factor: float = 1.0,
    ) -> None:
        self.args = args
        self.gradient_scale_factor = gradient_scale_factor
        msg = "Setting up Optimizer for: "

        # Parameter are a list of dictionaries, where each dictionary correspond to an MTLModule
        # Each dictionary can contain extra info such as a custom learning rate for that MTLModule
        ps: List[Dict[str, Any]] = []
        for k, shared_block in shared_blocks.items():
            ps.append(
                {
                    "initial_lr": self.args.optim_config.lr,
                    "params": list(shared_block.parameters()),
                }
            )
            msg = f"{msg} block:{k}"

        if dist.is_initialized():
            self.optimizer: optim.Optimizer = ZeroRedundancyOptimizer(
                ps,
                optimizer_class=self.args.optim_config.get_type(),
                **self.args.optim_config.get_params(),
            )
        else:
            self.optimizer: optim.Optimizer = self.args.optim_config.get_type()(
                ps, **self.args.optim_config.get_params()
            )

        self.schedulers = [c.build_instance(self.optimizer, max_epochs) for c in self.args.lr_scheduler_configs]
        self.task_optims: dict[str, optim.Optimizer] = {}
        # A list which contains the names of the tasks which still have UNUSED gradients in the current step
        self.still_has_gradient: list[str] = []
        logging.info(msg)

    def add_task(self, task: MTLTask) -> None:
        """
        Adds a new task's optimizable parameters. Cannot check if the task already exists.
        """
        assert task.get_name() not in list(self.task_optims.keys()), f"{self} already knows {task.get_name()}"
        task_optim_config = self.args.optim_config.model_copy()
        task_optim_config.lr = task_optim_config.lr * self.args.lr_factor_tasks
        task_optim = self.args.optim_config.get_type()(task.parameters(), **task_optim_config.get_params())
        self.task_optims[task.get_name()] = task_optim

    def reset_shared_grads(self):
        # We do not want the memory to accumulate with the number of tasks in a step -> set_to_none=True
        self.optimizer.zero_grad(set_to_none=True)

    def after_backward(self, of_task: MTLTask, reduce_task_immediately: bool):
        if reduce_task_immediately:
            self.task_optims[of_task.get_name()].step()
            # The docs mention that in that case steps might be skipped if gradients are set to none, beware!
            # In a multi-task setting, not all shared blocks might be used in every step resulting in skipped steps.
            self.task_optims[of_task.get_name()].zero_grad(set_to_none=True)
        else:
            if of_task.get_name() not in self.still_has_gradient:
                self.still_has_gradient.append(of_task.get_name())

    def after_iteration_step(self):
        # Gradients are averaged across all processes or at least nodes by DDP
        # We want to compensate for that by multiplying all shared parameter's gradients with the global_world_size
        # This only needs to be applied to the shared parameters, as the parameters of the tasks are optimized locally.
        if self.gradient_scale_factor != 1.0:
            for param_group in self.optimizer.param_groups:
                for param_index, p in enumerate(param_group["params"]):
                    if p.grad is not None:
                        # logging.debug(f"Gradient norm of {param_index}: {torch.norm(p.grad)}")
                        p.grad *= self.gradient_scale_factor
                        # logging.debug(f"Gradient norm of {param_index}: {torch.norm(p.grad)}")
        else:
            assert (
                not dist.is_initialized()
            ), "Scale factor is 1.0 while using DDP. Gradient scaling should be done with DDP."

        self.optimizer.step()

        for task_with_unused_grad in self.still_has_gradient:
            # logging.info(f"Task {task_with_unused_grad} still has unused gradients. Performing step.")
            self.task_optims[task_with_unused_grad].step()
            self.task_optims[task_with_unused_grad].zero_grad(set_to_none=True)
        self.still_has_gradient = []

    def after_loop_step(
        self,
        epoch: int,
        train_losses: List[List[float]],
        val_losses: Optional[List[List[float]]],
        verbose=False,
    ):
        """
        If a scheduler depends on the loss, then a list with the list of losses for all tasks needs to be provided.
        """
        for scheduler in self.schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                losses = val_losses if val_losses else train_losses
                task_averages = [np.average(ll) for ll in losses]
                # Each task contributes equally to the avg_loss
                avg_loss = np.average(task_averages)
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        if verbose:
            lrs = [param_group["lr"] for param_group in self.optimizer.param_groups]
            logging.debug(f"Learning rates for param_groups: {lrs}")

    def get_current_logstate(self) -> Dict[str, Any]:
        res = {}
        # if self.schedulers:
        #     scheduler_info = {f"schedulerlastlr{i}": v
        #                       for i, v in enumerate(self.schedulers[-1].get_last_lr())}
        #     res.update(scheduler_info)
        for i, param_group in enumerate(self.optimizer.param_groups):
            res[f"lr{i}"] = param_group["lr"]
        return res

    def create_checkpoint(self, folder_path: Path, rank: int) -> None:
        if dist.is_initialized():
            self.optimizer.consolidate_state_dict()
        if rank == 0:
            if not folder_path.exists():
                os.mkdir(folder_path)
            torch.save(self.optimizer.state_dict(), folder_path / "optim.ckpt")
            if self.schedulers:
                assert self.schedulers is not None
                torch.save(
                    [s.state_dict() for s in self.schedulers],
                    folder_path / "schedulers.ckpt",
                )

        if dist.is_initialized():
            dist.barrier()
        # Save checkpoints of the tasks
        for task_name, task_optim in self.task_optims.items():
            logging.debug(f"Saving optimizer checkpoint for task {task_name}")
            torch.save(task_optim.state_dict(), folder_path / f"optim_{task_name}.ckpt")

    def load_checkpoint(self, folder_path: Path) -> None:
        for task_name, task_optim in self.task_optims.items():
            logging.debug(f"Loading optimizer checkpoint for task {task_name}")
            task_optim.load_state_dict(torch.load(folder_path / f"optim_{task_name}.ckpt"))
        self.optimizer.load_state_dict(torch.load(folder_path / "optim.ckpt"))
        if self.schedulers:
            schedulers_path = folder_path / "schedulers.ckpt"
            if schedulers_path.exists():
                try:
                    states = torch.load(schedulers_path)
                    for scheduler, state in zip(self.schedulers, states):
                        scheduler.load_state_dict(state)
                except:
                    logging.warn(f"Checkpoint {schedulers_path} not valid for LR schedulers: {self.schedulers}")
            else:
                logging.warn(f"No state found for LR schedulers: {self.schedulers}")
