import logging
from typing import Literal, Tuple

import segmentation_models_pytorch.losses as smp_losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field

from mmm.BaseModel import BaseModel

from .TorchModule import TorchModule


class CrossEntropyLossConfig(TorchModule):
    loss_type: Literal["cross_entropy"] = "cross_entropy"

    def build_instance(self, *args, **kwargs) -> nn.Module:
        return nn.CrossEntropyLoss()


class NLLLossConfig(TorchModule):
    loss_type: Literal["negative_log_likelihood"] = "negative_log_likelihood"

    def build_instance(self, *args, **kwargs) -> nn.Module:
        return NLLLoss()


class NLLLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # NLLLoss excpets softmaxed pred. LogSoftmax used bcs of pytorch
        # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.loss_fc = nn.NLLLoss()
        self.soft = nn.LogSoftmax(dim=1)

    def __call__(self, y_pred, y_true):
        return self.loss_fc(self.soft(y_pred), y_true)


class MSELossConfig(TorchModule):
    loss_type: Literal["mean_squared_error"] = "mean_squared_error"
    value_range: Tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Expected range (min, max) of the values passed into the loss function. Used to scale the loss to [0, 1].",
    )

    def build_instance(self, *args, **kwargs) -> nn.Module:
        return MSELoss(self)


class SmoothL1Loss(nn.Module):
    """
    The maximum of the value range is used to scale the loss. It is ok to set the maximum lower than the actual maximum.
    """

    class Config(TorchModule):
        loss_type: Literal["smooth_l1"] = "smooth_l1"
        value_range: Tuple[float, float] = Field(
            default=(0.0, 1.0),
        )

        def build_instance(self, *args, **kwargs) -> nn.Module:
            return SmoothL1Loss(self)

    def __init__(self, args: MSELossConfig) -> None:
        super().__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()

        range_min, range_max = args.value_range
        self.factor = 1.0 / (range_max - range_min)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return self.factor * self.smooth_l1_loss(inputs, targets)


class RMSELossConfig(TorchModule):
    loss_type: Literal["root_mean_squared_error"] = "root_mean_squared_error"
    value_range: Tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Expected range (min, max) of the values passed into the loss function. Used to scale the loss to [0, 1].",
    )

    def build_instance(self, *args, **kwargs) -> nn.Module:
        return RMSELoss(self)


class FocalLossConfig(TorchModule):
    """
    Focal loss helps to focus your training on the difficult examples.
    """

    loss_type: Literal["focal"] = "focal"
    alpha: float = 1.0
    gamma: float = 2.0
    logits: bool = True
    reduce: bool = True

    def build_instance(self, *args, **kwargs) -> nn.Module:
        return FocalLoss(self)


class MSELoss(nn.Module):
    def __init__(self, args: MSELossConfig) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss()

        range_min, range_max = args.value_range
        self.factor = 1.0 / (range_max - range_min)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # return self.mse_loss(inputs * self.factor, targets * self.factor)
        return torch.mean(torch.square(targets - inputs))


class RMSELoss(nn.Module):
    def __init__(self, args: RMSELossConfig) -> None:
        super().__init__()
        self.mse_loss = MSELoss(MSELossConfig(value_range=args.value_range))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return torch.sqrt(self.mse_loss(inputs, targets))


class FocalLoss(nn.Module):
    """
    Loss based on cross entropy that emphasizes those outputs that have a large difference to the targets.
    Focal loss is a simple trick which can be used to train networks when class imbalance is present.
    Focusing parameter gamma: Increase to emphasize hard examples and put less effort into optimizing easy ones.

    For a drop in replacement of nn.CrossEntropyLoss, use the default values
    """

    def __init__(self, args: FocalLossConfig):
        super().__init__()
        self.args = args
        # self.alpha, self.gamma, self.logits, self.reduce = alpha, gamma, logits, reduce
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # targets = targets.long()
        if self.args.logits:
            bce_loss = self.ce_loss(inputs, targets)
        else:
            bce_loss = F.cross_entropy(inputs, targets, reduce=None)
        pt = torch.exp(-bce_loss)
        f_loss = self.args.alpha * (1 - pt) ** self.args.gamma * bce_loss

        if self.args.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss


class SMPFocalLoss:
    class Config(BaseModel):
        gamma: float = 2.0

    def __init__(self, cfg: Config, mode: Literal["multiclass", "multilabel"], ignore_index) -> None:
        self.crit = smp_losses.FocalLoss(mode=mode, ignore_index=ignore_index, gamma=cfg.gamma)

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        l = self.crit(preds, targets)
        if l.isnan():
            logging.warning(f"Focal loss is nan for {preds.shape=} and {targets.shape=}")
            return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
        else:
            return l


class SMPDiceFocalLoss2D:
    class Config(BaseModel):
        focal: SMPFocalLoss.Config = SMPFocalLoss.Config()

    def __init__(
        self,
        cfg: Config,
        ignore_index,
        mode: Literal["multiclass", "multilabel"] = "multiclass",
    ) -> None:
        self.crit1 = SMPFocalLoss(cfg.focal, mode=mode, ignore_index=ignore_index)
        self.crit2 = smp_losses.DiceLoss(mode=mode, ignore_index=ignore_index)

    def __call__(self, preds, mask):
        f_loss = self.crit1(preds, mask)
        # Dice loss is 0 for perfect prediction, 1 for worst possible prediction
        d_loss = self.crit2(preds, mask)
        return {"focal": f_loss, "dice": d_loss}


class SurvivalLossConfig(TorchModule):
    loss_type: Literal["nll_surv", "cox_reg", "bce_surv", "nnet_surv"] = "cox_reg"
    # How much the still alive examples count
    alpha: float = 0.2

    def build_instance(self) -> nn.Module:
        if self.loss_type == "nll_surv":
            return NLLSurvLoss(self.alpha)
        elif self.loss_type == "bce_surv":
            return BCESurvivalLoss(self.alpha)
        elif self.loss_type == "nnet_surv":
            return NnetSurvivalLoss(self.alpha)
        else:
            return CoxSurvivalLoss(self.alpha)


class CoxSurvivalLoss(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.eps: float = 1e-8
        self.continuous: bool = True

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, event: torch.Tensor):
        """
        Implements the loss function from DeepSurv
        y_pred: time prediction of event [B,n]. Should estimate the log-risk function of the Cox PH model
        y_true: actual time of event [B,n]
        event: if example is censored or not [B,n]
        """

        log_loss = torch.exp(y_pred)
        log_loss = torch.sum(log_loss, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((y_pred - log_loss) * event) / torch.sum(event)
        return neg_log_loss


class NLLSurvLoss(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.eps: float = 1e-8
        self.continuous: bool = False

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, event: torch.Tensor):
        """
        expects dicts with keys: censorship, surv, where hazard = nn.Sigmoid(surv_bin)
        the shapes of the tensors behind the keys should be:
        censorship: [B,1] where 1 means uncensored (event occours) and 0 means censored (no event)
        y_pred: [B,n_bins] Logits of bins predicted
        y_true: [B,1] true bin
        """
        # # calculate estimated hazards
        hazards = torch.sigmoid(y_pred)

        # # survival is a cumulative product of 1-hazard
        survival = torch.cumprod(1 - hazards, dim=1)

        # # S(-1) = 0, all patients are alive from (-inf, 0) by definition
        survival_padded = torch.cat([torch.ones_like(event).view(-1, 1), survival], 1)

        survival_before_event = torch.gather(survival_padded, dim=1, index=y_true.view(-1, 1)).clamp(min=self.eps)
        hazard_at_event = torch.gather(hazards, dim=1, index=y_true.view(-1, 1)).clamp(min=self.eps)
        survival_at_event = torch.gather(survival_padded, dim=1, index=(y_true + 1).view(-1, 1)).clamp(min=self.eps)

        uncensored_loss = -event * ((torch.log(survival_before_event) + torch.log(hazard_at_event)))
        censored_loss = -(1 - event) * torch.log(survival_at_event)

        loss = (uncensored_loss + censored_loss) + (self.alpha * uncensored_loss)

        return loss.sum()


class BCESurvivalLoss(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.eps: float = 1e-8
        self.continuous: bool = False

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, event: torch.Tensor):
        """
        expects dicts with keys: censorship, surv, where hazard = nn.Sigmoid(surv_bin)
        the shapes of the tensors behind the keys should be:
        censorship: [B,1] where 1 means uncensored (event occours) and 0 means censored (no event)
        y_pred: [B,n_bins] Logits of bins predicted
        y_true: [B,1] true bin
        """
        # # calculate estimated hazards
        hazards = torch.sigmoid(y_pred)

        # # survival is a cumulative product of 1-hazard
        survival = torch.cumprod(1 - hazards, dim=1)

        # # S(-1) = 0, all patients are alive from (-inf, 0) by definition
        survival_padded = torch.cat([torch.ones_like(event).view(-1, 1), survival], 1)

        survival_before_event = torch.gather(survival_padded, dim=1, index=y_true.view(-1, 1)).clamp(min=self.eps)
        hazard_at_event = torch.gather(hazards, dim=1, index=y_true.view(-1, 1)).clamp(min=self.eps)
        survival_at_event = torch.gather(survival_padded, dim=1, index=(y_true + 1).view(-1, 1)).clamp(min=self.eps)

        uncensored_loss = -event * ((torch.log(survival_before_event) + torch.log(hazard_at_event)))
        censored_loss = -(1 - event) * torch.log(survival_at_event) - event * torch.log(1 - survival_at_event)

        loss = (uncensored_loss + censored_loss) + (self.alpha * uncensored_loss)
        return loss.sum()


class NnetSurvivalLoss(nn.Module):
    """
    Loss is adapted from https://peerj.com/articles/6257/
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.eps: float = 1e-8
        self.continuous: bool = False

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, event: torch.Tensor):
        """
        expects dicts with keys: censorship, surv, where hazard = nn.Sigmoid(surv_bin)
        the shapes of the tensors behind the keys should be:
        censorship: [B,1] where 1 means uncensored (event occours) and 0 means censored (no event)
        y_pred: [B,n_bins] Logits of bins predicted
        y_true: [B,1] true bin
        """

        # # calculate estimated hazards
        hazards = torch.sigmoid(y_pred)

        # # survival is a cumulative product of 1-hazard
        survival = torch.cumprod(1 - hazards, dim=1)

        n_bins = y_pred.shape[1]
        # To adapt to the Nnet-surv loss we need to rearrange the y_true parts
        adapted_y_true = torch.zeros((y_true.shape[0], n_bins * 2))
        for i in range(y_true.shape[0]):
            adapted_y_true[i, n_bins : n_bins * 2] = F.one_hot(y_true[i], n_bins)
            adapted_y_true[i, 0:n_bins] = torch.zeros(n_bins).index_fill_(
                0, torch.from_numpy(np.arange(y_true[i].shape[0])), 1
            )

        adapted_y_true = adapted_y_true.to(survival.device)
        cens_uncens = 1.0 + adapted_y_true[:, 0:n_bins] * (survival - 1.0)  # component for all individuals
        uncens = 1.0 - adapted_y_true[:, n_bins : 2 * n_bins] * survival  # component for only uncensored individuals
        concatenated = torch.cat((cens_uncens, uncens), dim=-1)
        clipped = torch.clamp(concatenated, min=torch.finfo(concatenated.dtype).eps)
        loss = -torch.log(clipped)

        return loss.sum()


class KLDivLossConfig(TorchModule):
    loss_type: Literal["Kullback-Leibler"] = "Kullback-Leibler"

    def build_instance(self, *args, **kwargs) -> nn.Module:
        return nn.KLDivLoss()
