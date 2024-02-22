"""
Convenience utilities for testing without plausible training.

For a multi-task simulation dataset which can be trained on, take a look at the shape datasets.
"""

from typing import Any, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset

from mmm.data_loading.ClassificationDataset import ClassificationDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.mtl_modules.tasks.ClassificationTask import ClassificationTask


class ClassificationMockupDataset(Dataset):
    """
    Dataset of length N.
    """

    def __init__(self, N: int, image_shape=(1, 64, 64), return_real_task=False) -> None:
        super().__init__()
        self.N, self.image_shape = N, image_shape
        self.images = torch.ones(N, *image_shape)
        self.classes = torch.ones(N, dtype=torch.long)
        self.return_real_task = return_real_task
        self.class_names = ["No", "Yes"]
        for i in range(N, 2):
            self.images[i, 0] = torch.ones(image_shape[1], image_shape[2]) * 5
            self.classes[i] = 5

    def __len__(self):
        return self.N

    def __getitem__(self, index) -> Dict[str, Any]:
        if self.return_real_task:
            return {
                "image": self.images[index],
                "class": self.classes[index].unsqueeze(0),
            }
        else:
            class_code = int(index > (self.N // 2))
            return {
                "image": torch.ones(self.image_shape) * index,
                "class": torch.Tensor([class_code]),
            }

    @staticmethod
    def build_cohort(train_n: int, val_n: Optional[int]) -> TrainValCohort:
        train_src = ClassificationMockupDataset(train_n, image_shape=(1, 4, 4))
        val_src = ClassificationMockupDataset(val_n) if val_n is not None else None
        return TrainValCohort(
            TrainValCohort.Config(batch_size=(2, 2), shuffle_loaders=(False, False), num_workers=0),
            ClassificationDataset(train_src, class_names=train_src.class_names),
            (ClassificationDataset(val_src, class_names=val_src.class_names) if val_src is not None else None),
        )

    @classmethod
    def build_classification_task(cls, task_name: str, train_n: int, val_n: Optional[int]) -> ClassificationTask:
        cohort = cls.build_cohort(train_n, val_n)

        return ClassificationTask(
            64,
            # ["a", "b"],
            ClassificationTask.Config(
                module_name=task_name,
            ),
            cohort,
        )
