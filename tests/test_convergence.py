"""
The convergence tests are in a separate file for filtering them into a GPU-enabled environment.
"""

import numpy as np

from mmm.trainer.MTLTrainer import MTLTrainer

from .test_mtl_training import (
    mtl_semseg_trainer_factory,
    shape_segtask_factory,
    shape_semseg_cohort,
    default_optim_config,
    default_encoder_factory,
    default_squeezer_factory,
    default_decoder_factory,
)


def test_train_epoch(mtl_semseg_trainer_factory):
    """
    Currently, sometimes the test might fail due to bad luck.
    """
    TRAIN_STEPS, IMPROVEMENT = 40, 0.95
    mtl_semseg_trainer: MTLTrainer = mtl_semseg_trainer_factory(max_train_steps=TRAIN_STEPS)

    mtl_semseg_trainer.run_mtl_train_epoch()
    train_losses = mtl_semseg_trainer.mtl_tasks[0]._step_losses
    assert len(train_losses) == TRAIN_STEPS

    # Make sure the training loss got significantly smaller by checking with x% of the random loss
    assert np.mean(train_losses[TRAIN_STEPS // 4 :]) < np.mean(train_losses[: TRAIN_STEPS // 4]) * IMPROVEMENT
