from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import json5
import logfire
import wandb
from m3_sdk.DistributedPath import DistributedPath
from pydantic import Field
from pydantic_settings import BaseSettings

from mmm.BaseModel import BaseModel

# Dataloading
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.DataSplit import DataSplit
from mmm.event_selectors import CombinedEventSelector, FixedEventSelector, RecurringEventSelector
from mmm.logging.wandb_ext import remove_wandb_special_chars
from mmm.mmm_types.GroupUsage import GroupUsage
from mmm.mtl_modules.shared_blocks.FCOSDecoder import FCOSDecoder
from mmm.mtl_modules.shared_blocks.MTLDecoder import MTLDecoder
from mmm.mtl_modules.shared_blocks.PyramidDecoder import PyramidDecoder

# Blocks
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder
from mmm.mtl_modules.tasks.ClassificationTask import ClassificationTask
from mmm.mtl_modules.tasks.mmDetectionTask import MMDetectionTask

# Tasks
from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.mtl_modules.tasks.MultilabelClassificationTask import MultilabelClassificationTask
from mmm.mtl_modules.tasks.RegressionTask import RegressionTask
from mmm.mtl_modules.tasks.SemSegTask import SemSegTask
from mmm.neural.activations import ActivationFn, ActivationFunctionConfig
from mmm.neural.modules.convnext import TorchVisionConvnext
from mmm.neural.modules.simple_cnn import MiniConvNet
from mmm.neural.modules.simple_linear import SimpleLinearNet
from mmm.neural.modules.swinformer import TorchVisionSwinformer
from mmm.neural.modules.TimmEncoder import TimmEncoder
from mmm.neural.modules.TorchVisionCNN import TorchVisionCNN
from mmm.neural.pooling import GlobalPooling, GlobalPoolingConfig
from mmm.optimization.MTLOptimizer import (
    CosineAnnealingLRSchedulerConfig,
    DecaySchedulerConfig,
    ExponentialLRConfig,
    MTLOptimizer,
    OptimizerAdamWConfig,
    OptimizerSGDConfig,
    PolySchedulerConfig,
    ReduceLROnPlateauConfig,
    SchedulerType,
)
from mmm.settings import mtl_settings

# Training
from mmm.trainer.Loop import LoopConfig, TrainLoopConfig, ValLoopConfig
from mmm.trainer.MTLTrainer import EarlyStoppingConfig, MTLTrainer

TrainValCohortConfig = TrainValCohort.Config
# Training
MTLOptimizerConfig = MTLOptimizer.Config


# Blocks
MiniConvNetConfig = MiniConvNet.Config
SimpleLinearNetConfig = SimpleLinearNet.Config
TorchVisionSwinformerConfig = TorchVisionSwinformer.Config
FCOSDecoderConfig = FCOSDecoder.Config
AEDecoderConfig = MTLDecoder.Config
PyramidEncoderConfig = PyramidEncoder.Config
PyramidDecoderConfig = PyramidDecoder.Config
TorchVisionConvnextConfig = TorchVisionConvnext.Config

# Tasks
MTLTaskConfig = MTLTask.Config
ClassificationTaskConfig = ClassificationTask.Config
SemSegTaskConfig = SemSegTask.Config
MMDetectionTaskConfig = MMDetectionTask.Config
MultilabelClassificationTaskConfig = MultilabelClassificationTask.Config


class EnvByConvention:
    """
    Loads info about your environment by the conventions.
    """

    def __init__(self, env_name: str = "training", job_config_folder: str = "./job_configs/") -> None:
        """
        `env_name` might be "notebookname" for a notebook with filename `notebookname.ipynb`
        """
        self.env_name = env_name
        self.data_root = Path(os.getenv("ML_DATA_ROOT", default="/data_root/"))
        self.data_cache = Path(os.getenv("ML_DATA_CACHE", default="/dl_cache/"))
        self.data_output = Path(os.getenv("ML_DATA_OUTPUT", default="/data_output/"))
        self.interactive_environment = os.getenv("LOCAL_DEV_ENV", default="False") == "True"
        self.job_config_folder: Path = Path(job_config_folder)
        self.rank = int(os.getenv("RANK", default="0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", default=0))
        self.world_size = int(os.getenv("WORLD_SIZE", default="1"))

    def get_default_jobconfig_path(self) -> Path:
        return self.job_config_folder / f"{self.env_name}.jsonc"

    def get_schema_path(self) -> Path:
        return Path(f"./.vscode/{self.env_name}.schema.json")

    def if_torchrun_prepare(self) -> EnvByConvention:
        """
        Determines if the torchrun environment is set up. If so, will prepare dist training if applicable.
        """
        import torch
        import torch.distributed as dist

        from mmm.utils import find_missing_torchrun_envvars

        disttraining = len(find_missing_torchrun_envvars()) == 0
        if not disttraining:
            logging.info(f"Assuming local training because {find_missing_torchrun_envvars()} are not set")
        else:
            with logfire.span(
                "Initializing torch.distributed on rank {rank}, {local_rank}, world_size {world_size}",
                rank=self.rank,
                local_rank=self.local_rank,
                world_size=self.world_size,
            ):
                dist.init_process_group(backend="nccl")
        torch.cuda.set_device(self.local_rank)
        return self

    def __repr__(self) -> str:
        res = f"{self.data_root=}\n{self.data_output=}\n"
        res += f"{self.data_cache=}\n{self.interactive_environment=}\n"
        return res


class ExperimentHyperParameters(BaseSettings):
    experiment_name: str = Field(
        default="default",
        description="Multiple runs can have the same name. \
        Group name can be explicitly set by dot notation. \
        Example: 'name', 'group.name'. You can usually correct the name in W&B without consequences. \
        ",
    )
    resumable: bool = Field(
        default_factory=lambda: os.getenv("LOCAL_DEV_ENV", default="False") == "False",
        description="If true, \
              this job will automatically resume if it is stopped or crashes.\
              It will infer the ID from the experiment's name, you will have to find it from the W&B GUI.",
    )

    # Optional
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_notes: str | None = Field(default="", description="Make some notes about your experiment here")
    wandb_save_code: bool = Field(
        default=False,
        description="If False, WANDB will still save the commit. If true, will save the entry file's code.",
    )
    wandb_job_type: str | None = None

    clear_cache_on_node: str | None = Field(
        default=None,
        description="If this matches the node name (regex), env.data_cache will be cleared.",
    )

    # Top-level configs starting with "example_" do not get sent to W&B
    example_cnn_optim: MTLOptimizer.Config = Field(
        default=MTLOptimizerConfig(optim_config=OptimizerAdamWConfig(lr=2.5e-4)),
        description="Optimizer settings that work well with CNNs such as Resnet and Densenet.",
    )
    example_vit_optim: MTLOptimizer.Config = Field(
        default=MTLOptimizerConfig(optim_config=OptimizerAdamWConfig(lr=1e-4)),
        description="Optimizer settings that work well with vision transformers such as swin transformer.",
    )
    example_schedulers_cosinedecay: list[SchedulerType] = [
        DecaySchedulerConfig(last_epoch=100),
        CosineAnnealingLRSchedulerConfig(last_epoch=100),
    ]

    example_encoder_densenet_imgnet: PyramidEncoder.Config = PyramidEncoder.Config(model=TorchVisionCNN.Config(pretrained=True))  # type: ignore

    example_encoder_swintiny_imgnet: PyramidEncoder.Config = PyramidEncoder.Config(
        model=TorchVisionSwinformer.Config(pretrained=True),
    )  # type: ignore

    example_encoder_convnext_imgnet: PyramidEncoder.Config = PyramidEncoder.Config(
        model=TimmEncoder.Config(
            pretrained=True,
            variant="convnext_tiny",
        ),
    )
    # Enable the $schema keyword for autocompletion in VSCode to be set to any string
    # Prefix with example_ to avoid sending to W&B
    example_schema_dollar: str | None = Field(default=None, alias="$schema")

    @staticmethod
    def load_config_string(env: EnvByConvention) -> str:
        import json

        if not env.interactive_environment:
            config_env_name = f"MLOPS_JSON_{env.env_name}"
            assert config_env_name in list(
                os.environ.keys()
            ), f"""
For non-interactive jobs, the job config must be provided using the environment variable {config_env_name}.
If this is an interactive job, set environment variable LOCAL_DEV_ENV=True.
            """
            return os.getenv(config_env_name, default="{}")
        else:
            config_path = env.get_default_jobconfig_path()
            if not config_path.exists():
                config_path.parent.mkdir(exist_ok=True)
                config_path.write_text(f'{{"$schema": "{env.get_schema_path().absolute()}"}}')
                logging.info(
                    f"""
    For {env.interactive_environment=} jobs, the job config must be provided in {env.job_config_folder}.
    You can override the default location using the {type(env).__name__} constructor.
    The file was automatically created for you at {config_path.absolute()}.
                """
                )
            with open(config_path, "r") as f:
                return json.dumps(json5.load(f))

    @classmethod
    def load_config(cls, env: EnvByConvention):
        from mmm.utils import load_config_from_str

        config = load_config_from_str(cls, ExperimentHyperParameters.load_config_string(env))
        return config

    @classmethod
    def update_schema(cls, env: EnvByConvention):
        if not env.get_schema_path().exists():
            config_file_path = env.get_default_jobconfig_path()
            logging.info(
                f"""
Creating schema file at {env.get_schema_path().absolute()}.
You can use this to get autocompletion in VSCode by appending to json.schemas like
{{
    "fileMatch": ["{config_file_path}"],
    "url": "{env.get_schema_path()}"
}}
"""
            )
        env.get_schema_path().parent.mkdir(exist_ok=True, parents=True)
        env.get_schema_path().write_text(json.dumps(cls.model_json_schema(), indent=2))
        logging.info(f"Schema file written to {env.get_schema_path().absolute()}")

    @staticmethod
    def _delete_dir(direc: Path):
        for p in direc.iterdir():
            if p.is_dir():
                logging.debug(f"Removing directory {p}")
                ExperimentHyperParameters._delete_dir(p)
            else:
                p.unlink()
        direc.rmdir()

    def get_unique_id(self, rank: int, world_size: int, experiment_name: str) -> str:
        """
        rank refers to the global rank.
        """
        if world_size == 1:
            uid = f"{remove_wandb_special_chars(experiment_name)}"
        else:
            uid = f"{remove_wandb_special_chars(experiment_name)}{rank}"

        if uid == "default":
            uid = f"{uid}-{wandb.util.generate_id()}"
        return uid

    def get_run_description(self, rank: int, world_size: int) -> str | None:
        if "." in self.experiment_name:
            # n = self.get_unique_id(rank, world_size).split(".")[1]
            assert len(self.experiment_name.split(".")) == 2, f"{self.experiment_name=} must have max one dot"
            n = self.get_unique_id(rank, world_size, self.experiment_name.split(".")[1])
        else:
            n = self.get_unique_id(rank, world_size, self.experiment_name)

        return f"{n[:None if world_size == 1 else -1]} {rank+1}/{world_size}"

    def get_groupname(self, rank: int, world_size: int):
        if "." in self.experiment_name:
            return remove_wandb_special_chars(self.experiment_name.split(".")[0])
        if self.resumable:
            return remove_wandb_special_chars(self.experiment_name)
        else:
            return "temporary"

    def model_dump_for_wandb(self):
        return {k: v for k, v in self.model_dump().items() if not k.startswith("example_")}

    def init_experiment(self, env: EnvByConvention, always_restart: bool = False):
        if wandb.run is None or always_restart:
            wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                id=self.get_unique_id(env.rank, env.world_size, self.experiment_name),
                group=self.get_groupname(env.rank, env.world_size),
                job_type=self.wandb_job_type,
                name=self.get_run_description(env.rank, env.world_size),
                save_code=self.wandb_save_code,
                config=self.model_dump_for_wandb(),
                # If started not resumable and then it overwrites the data of the runs with the same ID
                resume=None if not self.resumable else "allow",
            )
            if mtl_settings.default_log_folder is None:
                mtl_settings.default_log_folder = DistributedPath.from_string(wandb_run.dir) / "m3logs"
                mtl_settings.default_log_folder.upath().mkdir(exist_ok=True, parents=True)
            wandb.log(
                {
                    "report/config": wandb.Html(
                        f"""
                <h3>The custom config for this training:</h3>
                <pre><code>{json.dumps(json.loads(self.load_config_string(env)), indent=2)}</pre></code>
                """
                    )
                }
            )
            logfire.info(
                "Run {run_url} resumed? {run_resumed}",
                run_url=wandb_run.url,
                run_resumed=wandb_run.resumed,
            )
            if wandb_run.resumed and not self.resumable:
                raise Exception("Resume not allowed by config")
        else:
            logfire.info(
                "wandb already initialized, not starting a new run because {always_restart}",
                always_restart=always_restart,
            )
            wandb_run = wandb.run
        return wandb_run
