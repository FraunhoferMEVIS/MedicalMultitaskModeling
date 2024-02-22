from enum import Enum


class TaskPurpose(str, Enum):
    pretraining = "pretraining"
    downstream = "downstream"
