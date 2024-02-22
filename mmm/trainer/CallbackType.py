from enum import Enum


class CallbackType(Enum):
    test_downstream_task = "test_downstream_task"
    each_epoch = "each_epoch"
