from enum import Enum

from mmm.BaseModel import BaseModel


class MaskingStrategy(str, Enum):
    fullattention = "fullattention"
    redundancy = "redundancy"


class GroupingStrategy(str, Enum):
    # One embedding for the whole group is returned.
    single = "single"
    # Exists because some grouping strategies (CLAM) returns only most important instances, and grouped instance.
    mixed = "mixed"  # probably fewer than full group, but more than 1.
    # all instances of the group, useful for segmentation
    full = "full"


class GroupUsage(BaseModel):
    """
    Used by tasks to specify assumptions about the grouping of their data.
    """

    grouper_key: str = "grouper"
    masking: MaskingStrategy = MaskingStrategy.fullattention
    grouping: GroupingStrategy = GroupingStrategy.full


class IncompatibleUsageError(Exception):
    pass
