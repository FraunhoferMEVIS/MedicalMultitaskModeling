from __future__ import annotations
from enum import Enum


class DataSplit(Enum):
    """
    By the time of writing, the library is only opinionated about training.
    As a result, only train and validation split handling is directly integrated.
    If you want to integrate testing or other custom code directly into the training,
    you can use the trainer's hooks.

    In this enum the actual values matter because other components are allowed to use them for indexing.
    """

    train = 0
    val = 1

    @staticmethod
    def from_index(i: int) -> DataSplit:
        return DataSplit.val if i != 0 else DataSplit.train
