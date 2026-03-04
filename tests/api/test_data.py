from typing import Generator
import pytest

from mmm.api.data import SubjDataset
from mmm.api.WorkerState import ws
from m3_sdk import types
import mmm.api.models as m3_models

from tests.test_data import image_url


@pytest.fixture(params=types.CompressType)
def compress_type(request) -> Generator[types.CompressType, None, None]:
    yield request.param


def test_singleimage_representation(image_url, compress_type: types.CompressType):
    if compress_type is types.CompressType.ctxtoken:
        pytest.skip("Context token compression is not implemented")
    subj: m3_models.MSubject = m3_models.MSubject(id="testsubject", data={"image": image_url})
    assert len(list(m3_models.MSubject.extract_instances(subj))) == 1
    reprs = [
        r
        for _, repr_batch in SubjDataset(SubjDataset.Config(), data=[subj]).tokenize(ws.fm, compress_type, batchsize=5)
        for r in repr_batch
    ]
    assert len(reprs) == 1
