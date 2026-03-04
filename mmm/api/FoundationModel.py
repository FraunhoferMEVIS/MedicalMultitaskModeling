from __future__ import annotations

from abc import ABC
from typing import Any

import torch
import torch.nn as nn
from m3_sdk.DistributedPath import DistributedPath
from m3_sdk.types import CompressType

from mmm.api.models import Repr


class FoundationModel(nn.Module, ABC):
    """
    This class serves as a wrapper for models to integrate them into the evaluation framework.
    Model sharing is expected through pickling (e.g., using `torch.save`).
    """

    @classmethod
    def load_from_disk(cls, file_path: DistributedPath, to_device: str) -> FoundationModel:
        """
        Load a model from disk. The model should be saved with `save_to_disk`.
        """
        raise NotImplementedError

    @staticmethod
    def save_to_disk(file_path: DistributedPath, neural_modules: Any):
        """
        Should result in a single file at least if the file_path ends with .zip
        """
        torch.save(neural_modules, file_path.upath())

    def get_identifier(self) -> str:
        """
        Should be unique w.r.t. weights and type of model.
        """
        raise NotImplementedError

    def get_device(self) -> str:
        """
        Return the device identifier of the model.
        """
        raise NotImplementedError

    def get_sharedblock_keys(self) -> list[str]:
        raise NotImplementedError

    def collate_instances(self, instances: list[dict[str, Any]], max_edge_len=512) -> dict[str, Any]:
        """
        Instances are a list of dictionaries. The output can be processed by the model via batch processing.
        """
        from mmm.data_loading.MTLDataset import mtl_collate
        from mmm.transforms import UnifySizes

        if not hasattr(self, "unify_sizes"):
            self.unify_sizes = UnifySizes(divisable_by=32, max_edge_len=max_edge_len)

        unified = self.unify_sizes(instances)
        return mtl_collate(unified)

    @torch.inference_mode()
    def compress_instances(
        self,
        instances: dict[str, Any],
        for_representation: list[CompressType],
    ) -> dict[CompressType, Repr]:
        """Compresses a batch of input images into neural representations for all specified label types.

        Batches are given as instances. Each instance is a dictionary with the following keys:

        instance['meta']['ls_task'] contains the LabelStudio task where this instance comes from.
        instance['image'] contains the image data.

        Args:
            model_input (torch.Tensor | list[Image]): if a tensor is provided, no further processing is done.
                                                      If a list of images is provided, they are processed with `image_processor`.
            for_label_types: Labeltypes, for which the compressed representation should be returned.

        Returns:
            dict[str, torch.Tensor]: mapping of labeltype to compressed representation.
        """
        raise NotImplementedError
