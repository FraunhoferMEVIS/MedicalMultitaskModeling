from __future__ import annotations
from typing import Any, List, Optional, Callable, Dict, Tuple, TypeVar, Iterator

import torch
from torch.utils.data import Dataset

from .MTLDataset import MTLDataset, SrcCaseType, InvalidCaseError


class MultipleInstanceDataset(MTLDataset):
    """
    - bag: list of dicts, where each dict must contain the keys specified by instance_case_keys
    - label: single label for the bag given by the integer index of the class name in class_names
    """

    def __init__(
        self,
        src_ds: Dataset[SrcCaseType],
        provide_latent_rep: bool,
        class_names: List[str],
        src_transform: Optional[Callable[[SrcCaseType], Dict[str, Any]]] = None,
        batch_transform: Optional[Callable[[SrcCaseType], Dict[str, Any]]] = None,
        verify_cases: int = 0,
    ) -> None:
        self.class_names = class_names
        self.provide_latent_rep = provide_latent_rep
        self.src_transform = src_transform
        self.batch_transform = batch_transform
        self.instance_keys = ["image", "position"]
        super().__init__(src_ds, ["bag", "label", "meta"], [], None, None, None, verify_cases)

    def verify_case_by_index(self, index: int) -> Dict[str, Any]:
        case = super().verify_case_by_index(index)
        assert isinstance(case["bag"], list)
        if not self.provide_latent_rep:
            for instance in case["bag"]:
                missing_keys = [k for k in self.instance_keys if k not in instance]
                if missing_keys:
                    raise InvalidCaseError(
                        self,
                        -1,
                        [f"{k} missing in case {instance}" for k in missing_keys],
                    )
        else:
            assert isinstance(case["bag"], torch.Tensor), "Bag must be a tensor if provided as latent representation."

        return case

    def get_untransformed_case(self, index: int) -> Any:
        case = self.src_ds[self._indices[index]]
        if not self.provide_latent_rep:
            if self.src_transform is not None:
                for idx, img in enumerate(case["bag"]):
                    case["bag"][idx] = self.src_transform(img)
        return self.data_stripper(case)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        src_case = self.get_untransformed_case(index) if self.cache_folder is None else self.get_case_from_cache(index)

        # The cohort may use these to apply random augmentations
        if not self.provide_latent_rep:
            if self.batch_transform is not None:
                for idx, img in enumerate(src_case["bag"]):
                    src_case["bag"][idx] = self.batch_transform(img)

        return src_case

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for d in iter(self.src_ds):
            if not self.provide_latent_rep:
                if self.src_transform is not None:
                    for idx, img in enumerate(d["bag"]):
                        d["bag"][idx] = self.src_transform(img)

                if self.batch_transform is not None:
                    for idx, img in enumerate(d["bag"]):
                        d["bag"][idx] = self.batch_transform(img)

            d = self.data_stripper(d)  # type: ignore
            yield d
