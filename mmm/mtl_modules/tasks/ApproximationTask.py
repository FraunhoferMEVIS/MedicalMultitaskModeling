from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from typing_extensions import Annotated

from mmm.data_loading.EmbeddingDataset import EmbeddingDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.logging.type_ext import StepMetricDict
from mmm.logging.wandb_ext import build_wandb_image
from mmm.mmm_types.GroupUsage import GroupUsage
from mmm.mtl_modules.shared_blocks.Grouper import Grouper
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.mtl_modules.shared_blocks.Squeezer import Squeezer
from mmm.neural import LossConfigs, MSELossConfig
from mmm.utils import flatten_list_of_dicts

from .MTLTask import MTLTask


class ApproximationTask(MTLTask):
    """
    Special case of a MTLTask which deals with approximation of other networks' output.

    Target embeddings should best be scaled to be similar in magnitude and variance to a torch.rand()
    """

    class Config(MTLTask.Config):
        encoder_key: str = "encoder"
        squeezer_key: str = "squeezer"
        grouper: GroupUsage = Field(
            default=GroupUsage(grouper_key=""),
            description="If the key is set, assumes a grouper to exist in the shared modules.",
        )
        approximator_dim: int = Field(1024, description="The size of the embedding to be approximated")

        approx_dropout: float = 0.1
        approx_loss: None | Annotated[LossConfigs, Field(discriminator="loss_type")] = MSELossConfig()
        approx_lossweight: float = 0.1
        approx_metrics: None | list[Literal["mae", "mse", "cosine_sim"]] = Field(
            default=None,
            description="If none, the task will decide which metrics make sense",
        )

        discr_loss: None | Literal["bce"] = "bce"
        discr_metrics: None | list[Literal["acc"]] = None
        discr_dropout: float = 0.2
        discr_lossweight: float = 0.9
        discr_sampleweights: tuple[float, float, float] = Field(
            (0.2, 1.0, 1.0), description="Weight for treating predictions as negatives, negatives, positives"
        )

    def __init__(
        self,
        for_squeezer: Squeezer,
        args: Config,
        cohort: TrainValCohort[EmbeddingDataset],
    ):
        super().__init__(args, cohort)
        self.args: ApproximationTask.Config  # Make sure IDE knows about the task specific fields
        self.hidden_dim = for_squeezer.get_hidden_dim()

        self.task_modules = self._create_pretraining_head()
        self.flatten = nn.Flatten(1)

        self.approx_loss: nn.Module = None if self.args.approx_loss is None else self.args.approx_loss.build_instance()
        self._grouper_metainfo = None

    def _create_pretraining_head(self):
        res = {}

        res["approximation_head"] = nn.Sequential(
            nn.Dropout(p=self.args.approx_dropout), nn.GELU(), nn.Linear(self.hidden_dim, self.args.approximator_dim)
        )

        if self.args.discr_loss is not None:
            # discr_hidden_dim = min(self.args.approximator_dim, self.hidden_dim)
            # res["image_proj"] = nn.Sequential(
            #     nn.Dropout(p=self.args.discr_dropout),
            #     nn.GELU(),
            #     nn.Linear(self.hidden_dim, discr_hidden_dim),
            # )
            res["embedding_proj"] = nn.Sequential(
                nn.Dropout(p=self.args.discr_dropout),
                nn.GELU(),
                nn.Linear(self.args.approximator_dim, self.hidden_dim),
            )
            res["discrimination_head"] = nn.Sequential(
                nn.Dropout(p=self.args.discr_dropout),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Dropout(p=self.args.discr_dropout),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )

        return nn.ModuleDict(res)

    def prepare_batch(self, batch: Dict[str, Any]) -> Any:
        batch["image"] = batch["image"].to(self.torch_device)
        batch["embeddings"] = [t.to(self.torch_device) for t in batch["embeddings"]]
        if "negatives" in batch:
            batch["negatives"] = [t.to(self.torch_device) for t in batch["negatives"]]
        return batch

    def forward(self, inputs, shared_blocks: Dict[str, SharedBlock]):
        x, (fe, te), supercase_indexes, contexts = inputs
        pyr = shared_blocks[self.args.encoder_key](x)
        pyr, squeezed = shared_blocks["squeezer"](pyr)
        squeezed = self.flatten(squeezed)

        if self.args.grouper.grouper_key:
            if hasattr(self.args, "positions") and self.args.positions is not None:
                positions = [c[self.args.positions[0]] for c in contexts]
            else:
                positions = None
            squeezed, self._grouper_metainfo = shared_blocks[self.args.grouper.grouper_key](
                squeezed, supercase_indexes, self.args.grouper, positions=positions
            )

        # Always apply the approximation_head, it is also used for discrimination
        pred_approx = self.task_modules["approximation_head"](squeezed)

        if self.args.discr_loss is not None:
            # image_projected = self.task_modules["image_proj"](squeezed)
            # Repeat the image projection to match the number of discriminator samples
            fe_preds = [item_pred.expand(f_emb.shape[0], -1) for item_pred, f_emb in zip(squeezed, fe)]
            te_preds = [item_pred.expand(t_emb.shape[0], -1) for item_pred, t_emb in zip(squeezed, te)]
            pred_like_candidates = torch.cat([squeezed] + fe_preds + te_preds)

            # Put all embedding candidates, both false and true, in one Tensor [#embeddings, embedding-dim]
            embedding_candidates = torch.cat([pred_approx.detach(), torch.cat(fe), torch.cat(te)])
            embedding_projected = self.task_modules["embedding_proj"](embedding_candidates)
            joined_hidden = torch.cat([pred_like_candidates, embedding_projected], dim=-1)

            pred_discr = self.task_modules["discrimination_head"](joined_hidden)
        else:
            pred_discr = None
        return pred_approx, pred_discr

    def training_step(self, batch: Dict[str, Any], shared_blocks: SharedModules):
        x = batch["image"]
        false_embeddings, true_embeddings = batch.get("negatives", None), batch["embeddings"]
        metas = batch.get("meta", [{} for _ in range(batch["image"].shape[0])])
        supercase_indices = Grouper.extract_ids_from_batch(
            [x.get("group_id", f"{i}") for i, x in enumerate(metas)],
            for_task_name=self.get_name(),
        ).to(self.torch_device)
        # [batch_size, embedding-size], [#embedding-candidates, 1]
        pred_approx, pred_discr = shared_blocks.forward(
            (x, (false_embeddings, true_embeddings), supercase_indices, [x.get("context", None) for x in metas]),
            self.forward,
        )

        step_results: StepMetricDict = {}
        live_vis = {}

        # if there are multiple possible answers, train for the closest correct answer
        with torch.no_grad():
            best_embeddings = []
            for batch_idx in range(x.shape[0]):
                candiate_embeddings = true_embeddings[batch_idx]
                if candiate_embeddings.shape[0] == 1:
                    best_embeddings.append(candiate_embeddings[0])
                else:
                    # Select the closest target by absolute distance
                    # pred = pred_approx[batch_idx].expand(candiate_embeddings.shape[0], -1)
                    diffs = (candiate_embeddings - pred_approx[batch_idx]).norm(dim=1)
                    best_idx = diffs.argmin().item()
                    best_embeddings.append(candiate_embeddings[best_idx])
        target_embs = torch.stack(best_embeddings)
        approx_loss = self.approx_loss(pred_approx, target_embs) * 6.0
        step_results["targets"] = target_embs.cpu().numpy()
        step_results["preds"] = pred_approx.detach().cpu().float().numpy()
        live_vis["loss_approx"] = approx_loss.item()

        pred_neg_targets = [0] * x.shape[0]
        neg_targets = [0] * sum(len(e) for e in false_embeddings)
        pos_targets = [1] * sum(len(e) for e in true_embeddings)
        disc_target = (
            torch.tensor(pred_neg_targets + neg_targets + pos_targets).float().to(self.torch_device).unsqueeze(1)
        )

        loss_weights = (
            ([self.args.discr_sampleweights[0]] * len(pred_neg_targets))
            + ([self.args.discr_sampleweights[1]] * len(neg_targets))
            + ([self.args.discr_sampleweights[2]] * len(pos_targets))
        )
        discr_loss = (
            F.binary_cross_entropy_with_logits(
                pred_discr,
                disc_target,
                weight=torch.tensor(loss_weights).float().to(self.torch_device).unsqueeze(1),
            )
            * 1.44269
        )
        assert torch.unique(disc_target[: len(pred_neg_targets)]).item() == 0.0
        step_results["pred_neg_discr_preds"] = (
            torch.sigmoid(pred_discr[: len(pred_neg_targets)]).detach().cpu().float().numpy()
        )
        assert torch.unique(disc_target[len(pred_neg_targets) : len(pred_neg_targets) + len(neg_targets)]).item() == 0.0
        step_results["neg_discr_preds"] = (
            torch.sigmoid(pred_discr[len(pred_neg_targets) : len(pred_neg_targets) + len(neg_targets)])
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        assert torch.unique(disc_target[len(pred_neg_targets) + len(neg_targets) :]).item() == 1.0
        step_results["pos_discr_preds"] = (
            torch.sigmoid(pred_discr[len(pred_neg_targets) + len(neg_targets) :]).detach().cpu().float().numpy()
        )
        live_vis["loss_discr"] = discr_loss.item()
        batch_loss = approx_loss * self.args.approx_lossweight + discr_loss * self.args.discr_lossweight

        self.add_step_result(batch_loss.item(), step_results)

        live_vis.update(self._visualize_preds(x, step_results, metas, supercase_indices))

        return batch_loss, live_vis

    def _visualize_preds(
        self, training_ims, step_metrics: Dict, metas: List[Dict], supercase_indices
    ) -> Dict[str, Any]:
        if (vis_n := min(self.ask_for_visualization(), training_ims.size(0))) <= 0:
            return {}

        if (visualizer := self.cohort.datasets[int(not self.training)].batch_visualizer) is not None:
            return visualizer(
                training_ims.detach().cpu(),  # [B, C, H, W]
                step_metrics,
                metas,
                supercase_indices,
                self._grouper_metainfo,
            )

    def log_epoch_metrics(self) -> Tuple[Dict[str, Any], str]:
        metrics = flatten_list_of_dicts(self._step_metrics)
        approx_metrics = ["mae", "cosine_sim"] if self.args.approx_metrics is None else self.args.approx_metrics
        discr_metrics = ["acc"] if self.args.discr_metrics is None else self.args.discr_metrics

        _, print_str = super().log_epoch_metrics()
        log_dict: Dict[str, Any] = {}

        if "mae" in approx_metrics:
            log_dict["mae"] = mean_absolute_error(metrics["targets"], metrics["preds"])

        if "mse" in approx_metrics:
            log_dict["mse"] = mean_squared_error(metrics["targets"], metrics["preds"])

        if "cosine_sim" in approx_metrics:
            sims = []
            for i in range(len(metrics["targets"])):
                sims.append(cosine_similarity(metrics["targets"][i].reshape(1, -1), metrics["preds"][i].reshape(1, -1)))
            sim = np.array(sims).mean()
            log_dict["cosine_sim"] = sim

        if "acc" in discr_metrics:
            log_dict["pred_neg_acc"] = accuracy_score(
                np.zeros_like(metrics["pred_neg_discr_preds"]), metrics["pred_neg_discr_preds"] > 0.5
            )
            log_dict["neg_acc"] = accuracy_score(
                np.zeros_like(metrics["neg_discr_preds"]), metrics["neg_discr_preds"] > 0.5
            )
            log_dict["pos_acc"] = accuracy_score(
                np.ones_like(metrics["pos_discr_preds"]), metrics["pos_discr_preds"] > 0.5
            )

        return log_dict, print_str
