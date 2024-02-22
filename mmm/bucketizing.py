from __future__ import annotations
import re
from typing import Callable, Dict, Optional, List, Literal, Tuple, TypeVar
from mmm.BaseModel import BaseModel
from pydantic import Field


class Bucket(BaseModel):
    expression: str
    description: Optional[str] = None
    expression_type: Literal["perfectmatch", "substring", "regex"] = "regex"

    def matches(self, s: str) -> bool:
        if self.expression_type == "perfectmatch":
            return s == self.expression
        elif self.expression_type == "substring":
            return self.expression in s
        elif self.expression_type == "regex":
            return bool(re.match(self.expression, s))
        else:
            raise Exception(f"Unknown expression type {self.expression_type}")

    def transform_to_class_name(self) -> str:
        if self.description:
            return self.description
        else:
            return self.expression

    def __str__(self) -> str:
        return self.transform_to_class_name()


class AmbiguousMatchException(Exception):
    pass


class BucketConfig(BaseModel):
    exactly_one_match: bool = Field(
        default=True,
        description="If true, only one bucket is allowed to match. Otherwise, the first match is returned.",
    )
    buckets: List[Bucket]

    def get_class_names(self) -> List[str]:
        descs = [b.transform_to_class_name() for b in self.buckets]
        assert len(set(descs)) == len(descs), f"Duplicate class names in {descs}"
        return descs

    def get_matches_for(self, desc: str) -> List[Bucket]:
        return [bucket for bucket in self.buckets if bucket.matches(desc)]

    def get_bucket(self, class_description: str) -> Tuple[Bucket, int]:
        matches = self.get_matches_for(class_description)
        assert len(matches) > 0, f"No match for {class_description}"

        if self.exactly_one_match and len(matches) != 1:
            raise AmbiguousMatchException(matches)

        return matches[0], self.buckets.index(matches[0])

    def __str__(self) -> str:
        return ", ".join([str(bucket) for bucket in self.buckets])

    def __eq__(self, other: BucketConfig):
        if not isinstance(other, BucketConfig):  # type: ignore
            return False

        return str(self) == str(other)

    def get_bucket_name(self, class_description: str) -> Tuple[str, int]:
        bucket, cls_id = self.get_bucket(class_description)
        return bucket.transform_to_class_name(), cls_id
