from typing import List
import pytest
from mmm.bucketizing import BucketConfig, Bucket, AmbiguousMatchException


@pytest.fixture
def buckets() -> List[Bucket]:
    buckets = [
        Bucket(expression=r"a", expression_type="regex"),
        Bucket(expression=r"aa", expression_type="regex"),
        Bucket(expression=r"baa", expression_type="regex"),
    ]
    return buckets


def test_regex_matching(buckets):
    matches = [bucket.matches("baab") for bucket in buckets]
    assert matches == [False, False, True]


def test_bucketconfig_regex_ambiguousexception(buckets):
    c = BucketConfig(buckets=buckets, exactly_one_match=True)
    with pytest.raises(AmbiguousMatchException):
        c.get_bucket_name("aa")


def test_bucketconfig_regex(buckets):
    c = BucketConfig(buckets=buckets, exactly_one_match=True)
    assert c.get_bucket_name("baab") == ("baa", 2)
