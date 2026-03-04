"""
Trainer that takes a foundation model and finetunes it. Developed for containerized environments.
"""

from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import logfire
import torch
import torch.nn as nn
from pydantic import Field
from typing_extensions import Annotated

from mmm.api.FoundationModel import FoundationModel
from mmm.event_selectors import EventSelector, RecurringEventSelector
from mmm.settings import mtl_settings as mtl_settings
from mmm.task_sampling import CyclicTaskSampler
from mmm.trainer.Loop import LinearMultistep, LoopLogConfig, TrainLoopConfig, ValLoopConfig
from mmm.trainer.MTLTrainer import EarlyStoppingConfig, MTLTrainer


class FineTuner(MTLTrainer):
    """
    Assumes some shared blocks are frozen and some are not. Finetunes the non-frozen blocks.

    It uses a key-value store for managing checkpoints.
    """

    class Config(MTLTrainer.Config):
        finetune_blocks: list[str] = Field(
            default=["decoder", "grouper", "mixer"],
            description="Only blocks whose name exactly matches one of these patterns will be finetuned.",
        )
        checkpoint_cache_folder: Path | None = Field(default=None, description="Finetuning does not use checkpoints.")

        mtl_train_loop: TrainLoopConfig = TrainLoopConfig(
            max_steps=25,
            task_sampler=CyclicTaskSampler.Config(mode="infinite"),
            multistep_mode=LinearMultistep(factor=1),
            # log_args=LoopLogConfig(progress_bar=False),
        )
        mtl_train_selector: Annotated[EventSelector, Field(discriminator="selector_type")] = RecurringEventSelector(
            starting_at=0
        )

        mtl_val_loop: ValLoopConfig = ValLoopConfig(
            max_steps=25,
            task_sampler=CyclicTaskSampler.Config(mode="infinite"),
            # log_args=LoopLogConfig(progress_bar=False),
        )
        mtl_validation_selector: Annotated[
            EventSelector, Field(discriminator="selector_type")
        ] = RecurringEventSelector(starting_at=0, every_n=1)

        early_stopping: EarlyStoppingConfig | None = EarlyStoppingConfig(
            criterion="trainloss", early_stopping_patience=3, min_train_loops=3
        )
        load_prefix: str = Field(default="latest", description="Prefix for loading checkpoints.")

    def __init__(
        self,
        args: Config,
        foundation_model: FoundationModel,
        adapter_name: str,
        lock_name: str,
        global_rank: int = int(os.getenv("RANK", default=0)),
        local_rank: int = int(os.getenv("LOCAL_RANK", default=0)),
        local_world_size: int = int(os.getenv("LOCAL_WORLD_SIZE", default=1)),
        world_size: int = int(os.getenv("WORLD_SIZE", default=1)),
        kv=None,
    ):
        self.kv = kv if kv is not None else self.kv
        self.lock_name, self.adapter_name = lock_name, adapter_name
        self.lock(self.lock_name)
        self.fm = foundation_model
        super().__init__(args, adapter_name, global_rank, local_rank, local_world_size, world_size)

        all_blocks = self.fm.get_sharedblock_keys()
        self.add_shared_blocks([self.fm[k] for k in all_blocks])

    def lock(self, lock_name: str):
        if lock_name:
            if self.kv.exists(lock_name):
                raise Exception(f"Lock {lock_name} already exists")
            self.kv.set(lock_name, "1", ex=60)  # expires after 1 minute

    def create_checkpoint(self, prefix):  # type: ignore[override]
        self.kv.set(f"{self._get_model_key(prefix)}:state", self.state.model_dump_json(indent=2))

        with logfire.span(
            "Creating checkpoint {prefix} for {adapter_name} in epoch {epoch}",
            prefix=prefix,
            adapter_name=self.adapter_name,
            epoch=self.state.epoch,
        ) as span:
            exportdict: nn.ModuleDict = self.save_blocks_native(
                None,
                only_for_blocks=[
                    k for k, sharedblock in self.shared_blocks.items() if sharedblock.training_state == "trainable"
                ],
            )
            span.set_attribute("modules", list(exportdict.keys()))
            torch.save(exportdict, byt := BytesIO())
            self.kv.set(model_key := f"{self._get_model_key(prefix)}:model", byt.getvalue())
            span.set_attribute("model_key", model_key)

            exportdict.cuda()  # Saving blocks converts them to cpu

            if self.mtl_optimizer is not None:
                with self.kv.pipeline() as pipe:
                    torch.save(self.mtl_optimizer.shared_state_to_dict(), byt := BytesIO())
                    pipe.set(
                        name=f"{self._get_model_key(prefix)}:optim",
                        value=byt.getvalue(),
                    )
                    for task_name, task_optim in self.mtl_optimizer.task_optims.items():
                        torch.save(task_optim.state_dict(), byt := BytesIO())
                        pipe.hset(
                            name=f"{self._get_model_key(prefix)}:task_optim",
                            key=task_name,
                            value=byt.getvalue(),
                        )
                    pipe.execute()
                span.set_attribute("optimizer_key", f"{self._get_model_key(prefix)}:optim")

        return self._get_model_key(prefix)

    def load_checkpoint(self, prefix, load_optim_state=True, load_meta=True, load_tasks=True):
        if self.kv.exists(f"{self._get_model_key(prefix)}:model"):
            with logfire.span(
                "Loading model {model} from DB",
                model=self._get_model_key(prefix),
            ) as span:
                exportdict = torch.load(
                    BytesIO(self.kv.get(f"{self._get_model_key(prefix)}:model")), weights_only=False
                )
                span.set_attribute("modules", list(exportdict.keys()))
                for k, v in self.shared_blocks.items():
                    if k in exportdict:
                        if self.shared_blocks[k].training_state != "trainable":
                            logfire.warning("Shared block {block} is not trainable, skipping loading", block=k)
                            continue
                        try:
                            v.load_state_dict(exportdict[k].state_dict())
                            logfire.info("Loaded shared block {block}", block=k)
                        except Exception as e:
                            logfire.error("Could not load shared block {block} due to {error}", block=k, error=e)

                if load_tasks:
                    for task in self.mtl_tasks:
                        if task.get_name() in exportdict:
                            try:
                                task.load_state_dict(exportdict[task.get_name()].state_dict())
                                logfire.info("Loaded task {task}", task=task.get_name())
                            except Exception as e:
                                logfire.error(
                                    "Could not load task {task} due to {error}", task=task.get_name(), error=e
                                )

            if load_optim_state:
                with self.kv.pipeline() as pipe:
                    pipe.get(optim_key := f"{self._get_model_key(prefix)}:optim")
                    pipe.hgetall(f"{self._get_model_key(prefix)}:task_optim")
                    results = pipe.execute()
                if results[0] is not None:
                    try:
                        self.mtl_optimizer.load_shared_state_from_dict(torch.load(BytesIO(results[0])))
                        logfire.info("Loaded optimizer state", optimizer_key=optim_key)
                    except Exception as e:
                        logfire.error("Could not load optimizer state", optimizer_key=optim_key, error=e)
                if results[1]:
                    task_optim_keys = [s.decode() for s in results[1].keys()]
                    if set(self.mtl_optimizer.task_optims.keys()) != set(task_optim_keys):
                        logfire.warning(
                            "Task optimizers in checkpoint do not match the current task optimizers. "
                            "Checkpoint: {checkpoint}, Current: {current}",
                            checkpoint=set(task_optim_keys),
                            current=set(self.mtl_optimizer.task_optims.keys()),
                        )
                    for task_name, task_optim in self.mtl_optimizer.task_optims.items():
                        if task_name in task_optim_keys:
                            try:
                                task_optim.load_state_dict(torch.load(BytesIO(results[1][task_name.encode()])))
                            except Exception as e:
                                logfire.warning(
                                    "Could not load task optimizer {task} due to {error}", task=task_name, error=e
                                )
                        else:
                            logfire.warning("Task optimizer {task} not found in checkpoint", task=task_name)

            if load_meta:
                self.state = self.State(**json.loads(self.kv.get(f"{self._get_model_key(prefix)}:state")))

    def _get_model_key(self, prefix: str):
        key = f"{mtl_settings.adapter_prefix}:{self.adapter_name}"
        if prefix:
            key += f":{prefix}"
        return key

    def cleanup_checkpoints(self):
        ...

    @staticmethod
    def is_done(finetuning_id: str, cfg: FineTuner.Config, kv=None):
        if kv is None:
            kv = mtl_settings.kv
        if kv.exists(latest_state_key := f"{mtl_settings.adapter_prefix}:{finetuning_id}:latest:state"):
            return (latest_state := FineTuner.State(**json.loads(kv.get(latest_state_key)))).epoch >= cfg.max_epochs
        else:
            return False

    @logfire.instrument("Finetuning for {num_loops} loops")
    def fit(self, num_loops=None):
        required_blocks = set([block_name for task in self.mtl_tasks for block_name in task.needs_shared_blocks()])
        logfire.info("Blocks required by tasks: {blocks}", blocks=required_blocks)

        # Will also decide which weights will be saved and loaded every time from DB
        finetune_blocks = [
            (
                block_key,
                any(re.match(pattern, block_key) for pattern in self.args.finetune_blocks)
                and block_key in required_blocks,
            )
            for block_key in self.shared_blocks.keys()
        ]
        logfire.info("Finetuning blocks: {blocks}", blocks=finetune_blocks)

        for finetune_block_key, finetune in finetune_blocks:
            if finetune:
                logfire.debug("Deepcopying block {block} from FM for finetuning", block=finetune_block_key)
                self.shared_blocks[finetune_block_key] = deepcopy(self.shared_blocks[finetune_block_key])
            self.shared_blocks[finetune_block_key].freeze_all_parameters(freeze=not finetune)

        if self.mtl_optimizer is None:
            # At this point the shared blocks will be wired up.
            # Changes like deepcopying blocks for fine-tuning will only work if done before this point.
            self.init_optimizer()

        self.load_checkpoint(
            self.args.load_prefix,
            load_optim_state=True,
            load_meta=True,
            load_tasks=True,
        )

        until_epoch = (
            self.args.max_epochs if num_loops is None else min(self.args.max_epochs, self.state.epoch + num_loops)
        )

        return self._fit(until_epoch)
