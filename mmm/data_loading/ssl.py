"""
Collection of Cohorts and Datasets for Self-supervised Tasks
"""

from __future__ import annotations

import torch
from typing import Any, Callable, Dict, List, Iterator, Optional
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np
import PIL

from torch.utils.data import Dataset, IterableDataset
from mmm.transforms import ApplyToKey
from mmm.logging.st_ext import side_by_side
from mmm.data_loading.MTLDataset import MTLDataset
from mmm.data_loading.ClassificationDataset import ClassificationDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.torch_ext import combine_datasets
from mmm.augmentations import SimCLRPatchAug, get_contrastive_2D_augs


class SSLDSWrapper(IterableDataset):
    def __init__(self, src) -> None:
        super().__init__()
        self.src = src
        self.src.batch_transform = None
        self.to_tensor = transforms.ToTensor()

    def channel_first(self, img: np.ndarray) -> np.ndarray:
        shape = img.shape
        # print(shape)
        if shape[0] == 3:
            return img
        elif shape[2] == 3:
            return np.transpose(img, (2, 1, 0))
        elif shape[1] == 3:
            return np.transpose(img, (1, 0, 2))
        else:
            channel_idx = np.argmin(shape)
            if shape[channel_idx] == 1:
                if shape[0] == 1:
                    return np.stack([img, img, img], axis=0)
                else:
                    img = np.stack([img, img, img], axis=2).squeeze()
                    return np.transpose(img, (2, 1, 0))
            else:
                raise Exception()

    def __iter__(self):
        """
        The price you pay for different Dataloding strategies
        """
        for item in self.src:
            if isinstance(item, Dict):
                img = item["image"]
            elif isinstance(item, tuple):
                img = item[0]
            else:
                continue
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            elif isinstance(img, PIL.Image.Image):
                img = np.asarray(img)
            else:
                continue
            if img.max() > 1.0:
                img = img / 255
            # img = torch.from_numpy(self.channel_first(img.squeeze()))
            img = torch.from_numpy(img)
            yield {"image": img.float(), "target": img.float()}


class SSLDataset(MTLDataset):
    def __init__(
        self,
        src_ds: Dataset,
        src_transform: Optional[Callable],
        input_transform: Optional[Callable],
        target_transform: Optional[Callable],
    ) -> None:
        super().__init__(
            src_ds,
            src_transform=None,
            optional_case_keys=["meta"],
            mandatory_case_keys=["image", "target"],
        )
        self.src_ds: Dataset = src_ds
        self.src_transform = src_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

    def set_classes_for_visualization(self, classes: List[str]):
        self.vis_classes = [str(i) for i in range(len(self.src_ds))]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        d = super().__getitem__(index)

        if self.src_transform is not None:
            d = self.src_transform(d)

        if self.target_transform is not None:
            d["target"] = self.target_transform(d["target"])

        if self.input_transform is not None:
            d["image"] = self.input_transform(d["image"])

        return d

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for d in iter(self.src_ds):
            if self.src_transform is not None:
                d = self.src_transform(d)
            if self.input_transform is not None:
                d["image"] = self.input_transform(d["image"])
            if self.target_transform is not None:
                d["target"] = self.target_transform(d["target"])

            yield d

    def st_case_viewer(self, case: Dict[str, Any], index: int) -> None:
        import streamlit as st

        st.title("SSL view")
        side_by_side(case["image"], case["target"])
        st.write(case)


class AETrainValCohort(TrainValCohort[SSLDataset]):
    class Config(TrainValCohort.Config):
        shared_img_size: int = 224

    def __init__(self, args: Config, cohort_list) -> None:
        src_transform = transforms.Compose(
            [
                ApplyToKey(
                    transforms.Resize((args.shared_img_size, args.shared_img_size)),
                    key="image",
                ),
                ApplyToKey(
                    transforms.Resize((args.shared_img_size, args.shared_img_size)),
                    key="target",
                ),
            ]
        )

        train_ds = get_combined_SSL_dataset(list=[c.datasets[0] for c in cohort_list], src_transform=src_transform)

        val_ds = get_combined_SSL_dataset(
            list=[c.datasets[1] for c in cohort_list if c.datasets[1] is not None],
            src_transform=src_transform,
        )

        super().__init__(args, train_ds, val_ds)


class BTTrainValCohort(TrainValCohort[SSLDataset]):
    class Config(TrainValCohort.Config):
        shared_img_size: int = 224

    def __init__(self, args: Config, cohort_list) -> None:
        src_transform = transforms.Compose(
            [
                ApplyToKey(
                    transforms.Resize((args.shared_img_size, args.shared_img_size)),
                    key="image",
                ),
                ApplyToKey(
                    transforms.Resize((args.shared_img_size, args.shared_img_size)),
                    key="target",
                ),
            ]
        )

        train_ds = get_combined_SSL_dataset(
            list=[c.datasets[0] for c in cohort_list],
            src_transform=src_transform,
            input_transform=transforms.Compose(get_contrastive_2D_augs()),
            target_transform=transforms.Compose(get_contrastive_2D_augs()),
        )

        val_ds = get_combined_SSL_dataset(
            list=[c.datasets[1] for c in cohort_list if c.datasets[1] is not None],
            src_transform=src_transform,
            input_transform=transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.08, 0.1))]),
            target_transform=transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.08, 0.1))]),
        )

        super().__init__(args, train_ds, val_ds)


def get_combined_SSL_dataset(list: List[MTLDataset], src_transform, input_transform=None, target_transform=None):
    ds = combine_datasets(list)
    return SSLDataset(
        src_ds=SSLDSWrapper(ds),
        src_transform=src_transform,
        input_transform=input_transform,
        target_transform=target_transform,
    )


# def cl_from_cohort(clf_cohort: TrainValCohort,
#                    cl_func: Callable
#                    ) -> TrainValCohort[ContrastiveImagePairDataset]:
#     val_ds = ContrastiveImagePairDataset(
#         clf_cohort.datasets[1],
#         cl_func) if clf_cohort.datasets[1] is not None else None
#     cl_cohort = TrainValCohort(
#         clf_cohort.args,
#         train_ds=ContrastiveImagePairDataset(clf_cohort.datasets[0], cl_func),
#         val_ds=val_ds
#     )
#     if cl_cohort.datasets[0].get_dataset_style() == DatasetStyle.MapStyle:
#         cl_cohort.datasets[0].set_classes_for_visualization(
#             [f"case_{i}" for i in range(len(cl_cohort.datasets[0]))])
#         if cl_cohort.datasets[1] is not None:
#             cl_cohort.datasets[1].set_classes_for_visualization(
#                 [f"case_{i}" for i in range(len(cl_cohort.datasets[1]))])

#     return cl_cohort


if __name__ == "__main__":
    import os
    from mmm.logging.st_ext import multi_cohort_explorer
    import streamlit as st
    import pydantic
    from pathlib import Path
    from mmm.data_loading.medical.histo_patchclassification import (
        Kather100kClassificationCohort,
        Kather100kClassificationCohortConfig,
    )
    from mmm.data_loading.medical.conic import ConicSemSegTrainValCohort

    def create_ae_cohort(cfg, lst):
        return AETrainValCohort(cfg, lst)

    def create_bt_cohort(cfg, lst):
        return BTTrainValCohort(cfg, lst)

    root = Path(os.getenv("ML_HISTO_ROOT", default="/histo_root/"))
    kather_conic = create_ae_cohort(
        cfg=AETrainValCohort.Config(batch_size=10),
        lst=[
            Kather100kClassificationCohort(root, Kather100kClassificationCohortConfig()),
            ConicSemSegTrainValCohort(root, ConicSemSegTrainValCohort.Config()),
        ],
    )
    BT = create_bt_cohort(
        cfg=BTTrainValCohort.Config(batch_size=16, num_workers=2),
        lst=[
            Kather100kClassificationCohort(root, Kather100kClassificationCohortConfig()),
            ConicSemSegTrainValCohort(root, ConicSemSegTrainValCohort.Config()),
        ],
    )

    multi_cohort_explorer(
        {
            # "kather_conic": lambda: kather_conic ,
            "BT": lambda: BT
        }
    )
