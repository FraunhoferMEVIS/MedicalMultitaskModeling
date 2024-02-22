import logging
import os
from pathlib import Path
from typing import Optional, Literal
import torch
import torch.nn as nn
from mmm.BaseModel import BaseModel
from torchinfo import ModelStatistics, summary
from torchinfo.layer_info import LayerInfo


class MTLModule(nn.Module):
    """
    Baseclass for all tasks and all shared blocks.
    """

    class Config(BaseModel):
        module_name: str

    def __init__(self, args: Config) -> None:
        super().__init__()
        self.args = args
        self._epoch = -1
        self._prefix = ""
        self.torch_device: str = "cpu"
        self.training_state: Literal["frozen", "trainable"] = "trainable"
        assert True not in [
            s in self.args.module_name for s in [".", "/"]
        ], f"Do not use wandb special characters in name {self.args.module_name}"

    def set_device(self, device: Literal["cpu", "cuda"]):
        """
        Replacement for torch's to method which also tracks the device.
        """
        if device == "cuda":
            # For distributed training, the user script needs to call torch.cuda.set_device
            self = self.cuda()
        elif device == "cpu":
            self = self.cpu()
        else:
            raise ValueError(f"Unknown torch device {device} in MTLModule.set_device")
        self.torch_device = device
        return self

    @torch.no_grad()
    def reinit_parameters(self) -> None:
        """
        Generic resetting of torch.nn modules.
        """

        # Child modules
        def init_weights(m):
            if hasattr(m, "reset_parameters"):
                print(f"Resetting {m}")
                m.reset_parameters()

        self.apply(init_weights)

    def get_name(self) -> str:
        """The name that the task was given in its config."""
        return self.args.module_name

    def load_checkpoint(self, folder_path: Path):
        checkpoint = torch.load(folder_path / "module.ckpt", map_location=torch.device(self.torch_device))
        try:
            self.load_state_dict(checkpoint["model_state"])
        except RuntimeError as e:
            logging.error(f"Could not load checkpoint {folder_path} due to {e}. Retrying without strict mode")
            self.load_state_dict(checkpoint["model_state"], strict=False)

    def save_checkpoint(self, folder_path: Path):
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        save_dict = {"model_state": self.state_dict()}
        torch.save(save_dict, folder_path / "module.ckpt")

    def prepare_epoch(self, epoch: int, prefix: str, training_mode: Optional[bool] = None):
        """
        Called in the beginning of an epoch. Does not need to be overwritten by shared building blocks.
        """
        self._epoch = epoch
        self._prefix = prefix
        if training_mode is not None:
            if training_mode:
                # trainable parameters of norms would still update even with requires_grad==False:
                self.train(self.training_state == "trainable")
            else:
                self.eval()

    def freeze_all_parameters(self, freeze: bool = True):
        for param in self.parameters():
            param.requires_grad = not freeze
        self.training_state = "frozen" if freeze else "trainable"

    def __repr_html__(self) -> str:
        """
        Returns a html string describing this MTLModule.
        """
        stats: ModelStatistics = summary(model=self, verbose=0, depth=25)
        return f"""
                <b>Device:</b>{self.torch_device}<br />

                <pre><code>{stats.__repr__()}</pre></code>
                <pre><code>{self}</pre></code>

                """

    def get_short_status(self) -> str:
        return "-"
