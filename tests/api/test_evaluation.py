import pytest
from mmm.api.models import MSubject
from mmm.api.evaluation import CrossValidation, ByMetaInfo

cases = [
    MSubject(id="1", meta={"center": "A"}, data={}),
    MSubject(id="2", meta={"center": "B"}, data={}),
    MSubject(id="3", meta={"center": "C"}, data={}),
    MSubject(id="4", meta={"center": "A"}, data={}),
    MSubject(id="5", meta={"center": "A"}, data={}),
]


# Create some splits which cover all cases and do not overlap between train and test sets
@pytest.fixture(
    ids=["ByMetaInfoNoVal", "ByMetaInfo", "CrossValidation"],
    params=[
        ByMetaInfo(train_criterion=("center", "A"), val_criterion=None, test_criterion=("center", "(B|C)")),
        ByMetaInfo(train_criterion=("center", "A"), val_criterion=("center", "B"), test_criterion=("center", "C")),
        CrossValidation(n_splits=2, shuffle=False, splitting_seed=42),
    ],
)
def split(request):
    return request.param


def test_splitting(split):
    splits = split.compute_splits(cases)

    for split_name, tr, val, ts in splits:
        assert tr is not None, "Train set should be defined"
        assert ts is not None, "Test set should be defined"
        if val is None:
            assert len(tr) + len(ts) == len(cases), "Train and test sets should cover all cases"
        assert len(tr + ts) == len(set(tr + ts)), "Train and test sets should not overlap"
