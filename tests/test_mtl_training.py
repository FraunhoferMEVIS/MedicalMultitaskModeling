import pytest
from typing import Callable, List, Tuple
import torch
from mmm.mtl_modules.MTLModule import MTLModule
from mmm.optimization.MTLOptimizer import MTLOptimizer

from mmm.trainer.MTLTrainer import MTLTrainer
from mmm.utils import recursive_equality
from mmm.trainer.Loop import TrainLoopConfig, ValLoopConfig


@pytest.fixture
def backbone_trainer_factory(
    tmp_path, task_factory, default_encoder_factory
) -> Callable[[MTLOptimizer.Config], MTLTrainer]:
    def create_trainer(optim_config: MTLOptimizer.Config):
        # Create a small backbone
        backbone = default_encoder_factory()

        # And a trainer with that backbone
        trainer = MTLTrainer(
            MTLTrainer.Config(
                optim=optim_config,
                max_epochs=1,
                mtl_train_loop=TrainLoopConfig(max_steps=1),
                mtl_val_loop=ValLoopConfig(max_steps=1),
            ),
            checkpoint_cache_folder=tmp_path,
        )
        trainer.add_shared_block(backbone)

        # Add 2 mtl tasks to the trainer
        for t_name in ["t1", "t2"]:
            trainer.add_mtl_task(task_factory(t_name))

        trainer.init_optimizer()
        return trainer

    return create_trainer


@pytest.fixture
def mtl_semseg_trainer_factory(
    tmp_path,
    wandb_run,
    shape_segtask_factory,
    default_optim_config,
    default_encoder_factory,
    default_decoder_factory,
    torch_device,
):
    def build_mtl_semseg_trainer(max_train_steps=5, max_val_steps=2, max_epochs=25):
        optim_config = MTLOptimizer.Config(optim_config=default_optim_config(), lr_scheduler_configs=[])
        enc = default_encoder_factory()
        dec = default_decoder_factory(enc)
        trainer = MTLTrainer(
            MTLTrainer.Config(
                max_epochs=max_epochs,
                mtl_train_loop=TrainLoopConfig(max_steps=max_train_steps),
                mtl_val_loop=ValLoopConfig(max_steps=max_val_steps),
                optim=optim_config,
                checkpoint_cache_folder=tmp_path,
                train_device=torch_device,
            ),
            experiment_name="mtl_semseg_trainer_factory_fixture",
        )
        trainer.add_shared_blocks([enc, dec])

        trainer.add_mtl_task(shape_segtask_factory(dec, task_name="shapesegtrain"))

        for t in trainer.mtl_tasks:
            t.cohort.prepare_epoch(0)

        trainer.init_optimizer()

        return trainer

    return build_mtl_semseg_trainer


def test_checkpointing_metainfo(mtl_semseg_trainer_factory: Callable):
    """
    Caveat: DeepDiff is used for comparisons between optimizers and cannot check for tensor equality.
    As a result, only the loading of any tensor through the checkpoint is ensured for the moment.
    """
    backbone_trainer: MTLTrainer = mtl_semseg_trainer_factory(max_train_steps=1, max_val_steps=1, max_epochs=1)
    backbone_trainer2: MTLTrainer = mtl_semseg_trainer_factory(max_train_steps=1, max_val_steps=1, max_epochs=1)

    # Change the state of the first trainer
    backbone_trainer.state.epoch = 54
    backbone_trainer.state.step_counters["test_prefix"] = 542

    backbone_trainer.create_checkpoint("metainfotest")
    backbone_trainer2.load_checkpoint(backbone_trainer.get_recent_checkpoint_path("metainfotest"))

    assert backbone_trainer.state.epoch == backbone_trainer2.state.epoch
    assert (
        backbone_trainer.state.step_counters["test_prefix"]
        == backbone_trainer2.state.step_counters["test_prefix"]
        == 542
    )


def test_checkpointing(mtl_semseg_trainer_factory: Callable):
    """
    Caveat: DeepDiff is used for comparisons between optimizers and cannot check for tensor equality.
    As a result, only the loading of any tensor through the checkpoint is ensured for the moment.
    """
    backbone_trainer: MTLTrainer = mtl_semseg_trainer_factory(max_train_steps=1, max_val_steps=1, max_epochs=1)
    # TEST_CHECKPOINT_PATH: Path = backbone_trainer.checkpoint_cache_parent / "test_checkpoint"
    backbone_trainer2: MTLTrainer = mtl_semseg_trainer_factory(max_train_steps=1, max_val_steps=1, max_epochs=1)
    backbone_trainer.run_mtl_train_epoch()

    # Save the trainer
    backbone_trainer.create_checkpoint("checkpointtest")

    # Assert that the two trainers are different
    def compare(should_be_equal: bool):
        shared_pairs = list(
            zip(
                backbone_trainer.shared_blocks.values(),
                backbone_trainer2.shared_blocks.values(),
            )
        )
        task_pairs = list([(t1, t2) for t1, t2 in zip(backbone_trainer.mtl_tasks, backbone_trainer2.mtl_tasks)])
        # eval_task_pairs = list([(t1, t2) for t1, t2 in zip(backbone_trainer.eval_tasks, backbone_trainer2.eval_tasks)])
        all_modules: List[Tuple[MTLModule, MTLModule]] = shared_pairs + task_pairs  # type: ignore

        assert backbone_trainer is not backbone_trainer2

        # Optimizer
        assert backbone_trainer2.mtl_optimizer is not None
        # Trainer with optimizer
        assert (
            recursive_equality(
                backbone_trainer.mtl_optimizer.optimizer.state_dict(),
                backbone_trainer2.mtl_optimizer.optimizer.state_dict(),
            )
            == should_be_equal
        )
        # else:
        #     # Optimizers of MTLModules are used
        #     for module1, module2 in shared_pairs + task_pairs:  # test only those optimizers used in training
        #         assert (module1.optim.optimizer is not None) and (module2.optim.optimizer is not None)
        #         assert recursive_equality(
        #             module1.optim.optimizer.state_dict(),
        #             module2.optim.optimizer.state_dict()) == should_be_equal

        # parameters
        for b1, b2 in all_modules:
            for (p1_n, p1), (p2_n, p2) in zip(b1.named_parameters(), b2.named_parameters()):
                assert p1_n == p2_n, "The test assumes that parameters are loaded in the same order"
                assert p1 is not p2, "Both trainers seem to hold the exact same parameter instance"
                assert torch.equal(p1, p2) or (not should_be_equal), f"Checkpointing forgot {p1_n} parameter"

    compare(should_be_equal=False)

    # Load second trainer from checkpoint
    backbone_trainer2.load_checkpoint(backbone_trainer.get_recent_checkpoint_path("checkpointtest"))

    # Assert that the original trainer and the loaded trainer have exactly the same state
    compare(should_be_equal=True)
