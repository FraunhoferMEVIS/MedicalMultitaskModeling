import logging
from typing import Any, Literal, Tuple
from typing_extensions import Unpack
from pydantic import Field
from pydantic.config import ConfigDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.models import vgg16, VGG16_Weights, resnet18

from .TorchModule import TorchModule
import segmentation_models_pytorch.losses as smp_losses

from mmm.torch_ext import replace_childen_recursive
from mmm.BaseModel import BaseModel


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
        return self.mse_loss(inputs * self.factor, targets * self.factor)


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
            return 0
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
    loss_type: Literal["nll_surv", "cox_reg"] = "cox_reg"
    # How much the still alive examples count
    alpha: float = 0.2

    def build_instance(self) -> nn.Module:
        if self.loss_type == "nll_surv":
            return NLLSurvLoss(self.alpha)
        else:
            return CoxSurvivalLoss(self.alpha)


class CoxSurvivalLoss(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.eps: float = 1e-8
        self.continuous: bool = True

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, censor: torch.Tensor):
        """
        Implements the loss function from DeepSurv
        y_pred: time prediction of event [B,n]. Should estimate the log-risk function of the Cox PH model
        y_true: actual time of event [B,n]
        censor: if example is censored or not [B,n]
        """

        log_loss = torch.exp(y_pred)
        log_loss = torch.sum(log_loss, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((y_pred - log_loss) * censor) / torch.sum(censor)
        return neg_log_loss


class NLLSurvLoss(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.eps: float = 1e-8
        self.continuous: bool = False

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, censor: torch.Tensor):
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
        survival_padded = torch.cat([torch.ones_like(censor).view(-1, 1), survival], 1)

        survival_before_event = torch.gather(survival_padded, dim=1, index=y_true.view(-1, 1)).clamp(min=self.eps)
        hazard_at_event = torch.gather(hazards, dim=1, index=y_true.view(-1, 1)).clamp(min=self.eps)
        survival_at_event = torch.gather(survival_padded, dim=1, index=(y_true + 1).view(-1, 1)).clamp(min=self.eps)

        uncensored_loss = -censor * ((torch.log(survival_before_event) + torch.log(hazard_at_event)))
        censored_loss = -(1 - censor) * torch.log(survival_at_event)

        loss = (uncensored_loss + censored_loss) + (self.alpha * uncensored_loss)
        return loss.sum()
