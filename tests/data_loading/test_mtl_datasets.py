import pytest
from mmm.data_loading.RegressionDataset import RegressionDataset
from mmm.data_loading.synthetic.mockup import ClassificationMockupDataset
from mmm.data_loading.MTLDataset import InvalidCaseError


def test_verification_fail():
    ds = RegressionDataset(ClassificationMockupDataset(10))
    with pytest.raises(InvalidCaseError):
        # This should fail because the target is missing
        ds.verify_case_by_index(0)


def test_verification_success():
    ds = RegressionDataset(
        ClassificationMockupDataset(10),
        src_transform=lambda x: {"image": x["image"], "target": 0.5},
    )
    ds.verify_case_by_index(0)
