"""
Configurable datatypes for selecting iterations in a loop for something to happen.
"""

from abc import abstractmethod
from typing import List, Literal, Union
from mmm.BaseModel import BaseModel
from pydantic import validator

import random
import math


class EventSelectorBase(BaseModel):
    @abstractmethod
    def is_event(self, iteration: int) -> bool:
        raise NotImplementedError()


class FixedEventSelector(EventSelectorBase):
    selector_type: Literal["fixed"] = "fixed"
    at_iterations: List[int] = [0]

    def is_event(self, iteration: int):
        return iteration in self.at_iterations


class RecurringEventSelector(EventSelectorBase):
    selector_type: Literal["recurring"] = "recurring"
    every_n: int = 1
    starting_at: int = 0

    def is_event(self, iteration: int):
        return iteration % self.every_n == 0 and iteration >= self.starting_at


class CodedEventSelector(EventSelectorBase):
    """Enables code injection, do not allow in apps which use external configs!"""

    selector_type: Literal["coded"] = "coded"
    python_program: str = r"i % random.randint(2, 8) == 0"

    @validator("python_program")
    def must_be_valid_program(cls, _):
        res = cls._execute_prog(0)
        assert isinstance(res, bool), "program must execute to bool"

    def _execute_prog(self, iteration: int):
        return eval(self.python_program, {"i": iteration, "random": random, "math": math})

    def is_event(self, iteration: int):
        return self._execute_prog(iteration)


class CombinedEventSelector(EventSelectorBase):
    selector_type: Literal["combined"] = "combined"
    events: List[Union[FixedEventSelector, RecurringEventSelector, CodedEventSelector]]

    def is_event(self, iteration: int):
        subevent_results = [v.is_event(iteration) for v in self.events]
        return True in subevent_results


EventSelector = Union[
    FixedEventSelector,
    RecurringEventSelector,
    CodedEventSelector,
    CombinedEventSelector,
]
