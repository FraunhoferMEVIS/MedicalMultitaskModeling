from __future__ import annotations
import wandb
import re
import logging
import json
import json5
from typing import List
from pathlib import Path
import os
from typing import Optional

from pydantic import Field, model_validator
from mmm.BaseModel import BaseModel

# Dataloading
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.logging.wandb_ext import remove_wandb_special_chars
from mmm.DataSplit import DataSplit

# Training
from mmm.trainer.Loop import LoopConfig, TrainLoopConfig, ValLoopConfig
from mmm.trainer.MTLTrainer import MTLTrainer, EarlyStoppingConfig
from mmm.optimization.MTLOptimizer import (
    MTLOptimizer,
    OptimizerAdamWConfig,
    OptimizerSGDConfig,
    SchedulerType,
)

from mmm.optimization.MTLOptimizer import (
    PolySchedulerConfig,
    ExponentialLRConfig,
    CosineAnnealingLRSchedulerConfig,
    DecaySchedulerConfig,
    ReduceLROnPlateauConfig,
)
from mmm.event_selectors import (
    FixedEventSelector,
    CombinedEventSelector,
    RecurringEventSelector,
)

# Blocks
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder
from mmm.mtl_modules.shared_blocks.PyramidDecoder import PyramidDecoder
from mmm.mtl_modules.shared_blocks.FCOSDecoder import FCOSDecoder
from mmm.mtl_modules.shared_blocks.MTLDecoder import MTLDecoder
from mmm.neural.modules.smp_modules import SMPPyramidAttentionDecoderConfig
from mmm.neural.modules.TorchVisionCNN import TorchVisionCNN

from mmm.neural.modules.swinformer import TorchVisionSwinformer
from mmm.neural.modules.simple_cnn import MiniConvNet
from mmm.neural.modules.convnext import TorchVisionConvnext
from mmm.neural.modules.TimmEncoder import TimmEncoder
from mmm.neural.activations import ActivationFunctionConfig, ActivationFn
from mmm.neural.pooling import GlobalPoolingConfig, GlobalPooling

# Tasks
from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.mtl_modules.tasks.ClassificationTask import ClassificationTask
from mmm.mtl_modules.tasks.RegressionTask import RegressionTask
from mmm.mtl_modules.tasks.SemSegTask import SemSegTask
from mmm.mtl_modules.tasks.mmDetectionTask import MMDetectionTask
from mmm.mtl_modules.tasks.MultilabelClassificationTask import (
    MultilabelClassificationTask,
)

TrainValCohortConfig = TrainValCohort.Config
# Training
MTLOptimizerConfig = MTLOptimizer.Config

# Neural
from mmm.neural.pooling import GlobalPoolingConfig, GlobalPooling

# Blocks
MiniConvNetConfig = MiniConvNet.Config
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

    def __init__(self, env_name: str = "training") -> None:
        """
        `env_name` might be "notebookname" for a notebook with filename `notebookname.ipynb`
        """
        from pathlib import Path

        # This is needed to override the logging of IPython
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        self.env_name = env_name
        self.data_root = Path(os.getenv("ML_DATA_ROOT", default="/data_root/"))
        self.data_cache = Path(os.getenv("ML_DATA_CACHE", default="/dl_cache/"))
        self.data_output = Path(os.getenv("ML_DATA_OUTPUT", default="/data_output/"))
        self.interactive_environment = os.getenv("LOCAL_DEV_ENV", default="False") == "True"
        self.job_config_folder: Path = Path("./job_configs/")
        self.rank = int(os.getenv("RANK", default="0"))
        self.world_size = int(os.getenv("WORLD_SIZE", default="1"))

    def get_default_jobconfig_path(self) -> Path:
        return self.job_config_folder / f"{self.env_name}.jsonc"

    def get_schema_path(self) -> Path:
        return Path(f"./.vscode/{self.env_name}.schema.json")

    def if_torchrun_prepare(self) -> EnvByConvention:
        """
        Determines if the torchrun environment is set up. If so, will prepare dist training if applicable.
        """
        from mmm.utils import find_missing_torchrun_envvars
        import torch
        import torch.distributed as dist

        disttraining = len(find_missing_torchrun_envvars()) == 0
        if not disttraining:
            logging.info(f"Assuming local training because {find_missing_torchrun_envvars()} are not set")
        else:
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", default=0)))
        return self

    def __repr__(self) -> str:
        res = f"{self.data_root=}\n{self.data_output=}\n" f"{self.data_cache=}\n{self.interactive_environment=}\n"
        return res


class ExperimentHyperParameters(BaseModel):
    """ """

    experiment_name: str = Field(
        default="default",
        help="Multiple runs can have the same name. \
        Group name can be explicitly set by dot notation. \
        Example: 'name', 'group.name'. You can usually correct the name in W&B without consequences. \
        ",
    )
    resumable: bool = Field(
        default_factory=lambda: os.getenv("LOCAL_DEV_ENV", default="False") == "False",
        help="If true, \
              this job will automatically resume if it is stopped or crashes.\
              It will infer the ID from the experiment's name, you will have to find it from the W&B GUI.",
    )

    # Optional
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_notes: str | None = Field(default="", description="Make some notes about your experiment here")
    wandb_host: str | None = Field(
        default=None,
        description="Useful to identify the physical machine that your job runs on",
    )
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
    example_schedulers_cosinedecay: List[SchedulerType] = [
        DecaySchedulerConfig(last_epoch=100),
        CosineAnnealingLRSchedulerConfig(last_epoch=100),
    ]

    example_encoder_densenet_imgnet: PyramidEncoder.Config = PyramidEncoder.Config(
        model=TorchVisionCNN.Config(pretrained=True)
    )  # type: ignore

    example_encoder_swintiny_imgnet: PyramidEncoder.Config = PyramidEncoder.Config(
        model=TorchVisionSwinformer.Config(pretrained=True),
    )  # type: ignore

    example_encoder_convnext_imgnet: PyramidEncoder.Config = PyramidEncoder.Config(
        model=TimmEncoder.Config(
            pretrained=True,
            variant="convnext_tiny",
        ),
    )

    @staticmethod
    def load_config_string(env: EnvByConvention) -> str:
        import json

        if not env.interactive_environment:
            config_env_name = f"MLOPS_JSON_{env.env_name}"
            assert config_env_name in list(
                os.environ.keys()
            ), f"""
For non-interactive jobs, the job config must be provided using the environment variable {config_env_name}.
            """
            return os.getenv(config_env_name, default="{}")
        else:
            config_path = env.get_default_jobconfig_path()
            if not config_path.exists():
                config_path.parent.mkdir(exist_ok=True)
                config_path.write_text(r"{}")
                logging.info(
                    f"""
    For {env.interactive_environment=} jobs, the job config must be provided in {env.job_config_folder}.
    You can override the default location using the {type(env).__name__} constructor.
    The file was automatically created for you at {config_path}.
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
Creating schema file at {env.get_schema_path()}.
You can use this to get autocompletion in VSCode by appending to json.schemas like
{{
    "fileMatch": ["{config_file_path}"],
    "url": "{env.get_schema_path()}"
}}
"""
            )
        env.get_schema_path().write_text(json.dumps(cls.model_json_schema()))

    @staticmethod
    def _delete_dir(direc: Path):
        for p in direc.iterdir():
            if p.is_dir():
                logging.debug(f"Removing directory {p}")
                ExperimentHyperParameters._delete_dir(p)
            else:
                p.unlink()
        direc.rmdir()

    def clear_cache_on_nodes_like(self, env: EnvByConvention, machine_id: Optional[str]):
        if self.clear_cache_on_node is not None and machine_id is not None:
            if not env.data_cache.exists():
                logging.warn(f"not delete cache, because {env.data_cache} does not exist")
                return
            if re.match(self.clear_cache_on_node, machine_id):
                import shutil

                assert "mtltorch" in env.data_cache.parts
                free_before = shutil.disk_usage(env.data_cache)[2]
                logging.info(f"Clearing cache at {env.data_cache}")
                for p in env.data_cache.iterdir():
                    if p.is_dir():
                        print(f"Removing directory {p}")
                        self._delete_dir(p)
                    else:
                        p.unlink()
                free_after = shutil.disk_usage(env.data_cache)[2]
                logging.info(f"Freed {(free_after-free_before)/1024/1024/1024:.2f} GB")

            else:
                logging.warn(
                    f"Cache will only be cleared on nodes matching {self.clear_cache_on_node}, this is {machine_id}"
                )

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

    def init_experiment(self, env: EnvByConvention, always_restart: bool = False):
        if wandb.run is None or always_restart:
            if machine_id := os.getenv("NODE_NAME", default=self.wandb_host):
                os.environ["WANDB_HOST"] = machine_id
            wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                id=self.get_unique_id(env.rank, env.world_size, self.experiment_name),
                group=self.get_groupname(env.rank, env.world_size),
                job_type=self.wandb_job_type,
                name=self.get_run_description(env.rank, env.world_size),
                save_code=self.wandb_save_code,
                config={k: v for k, v in self.model_dump().items() if not k.startswith("example_")},
                # If started not resumable and then it overwrites the data of the runs with the same ID
                resume=None if not self.resumable else "allow",
            )
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
            logging.info(f"Run resumed? {wandb_run.resumed=}")
            if wandb.run.resumed and not self.resumable:
                raise Exception("Resume not allowed by config")

            self.clear_cache_on_nodes_like(env, machine_id)
        else:
            logging.info(f"wandb already initialized, not starting a new run because {always_restart=}")
            wandb_run = wandb.run
        return wandb_run
