import os
import pytest
from torch.utils.data import DataLoader
from mmm.data_loading.CachingSubCaseDSV2 import CachingSubCaseDSV2, CacheItem, M3LoaderInfo


@pytest.mark.parametrize("batch_size", [1, 3], ids=["bs_1", "bs_3"])
@pytest.mark.parametrize("cache_items_per_hit", [1, 3, 15], ids=["cache_1", "cache_3", "cache_15"])
@pytest.mark.parametrize("min_max_cachehits", [(1, 1), (5, 5)], ids=["one", "five"])
def test_min_max_cachehits(batch_size, cache_items_per_hit, min_max_cachehits):
    if os.getenv("MMM_TEST_JOB", None) is not None:
        pytest.skip("DB should to be made optional")
    data = list(range(10))
    loaded = []

    def cacher(x):
        loaded.append(x)
        return CacheItem(original=x, name=str(x))

    ds = CachingSubCaseDSV2(
        CachingSubCaseDSV2.Config(cache_items_per_hit=cache_items_per_hit, min_max_cachehits=min_max_cachehits),
        data,
        cacher=cacher,
        cache_loader=lambda cache_item, remaining_batch_size: [cache_item.original],
    )
    ds.mmm_info = M3LoaderInfo(task_name="test_min_max_cachehits", batch_size=batch_size, split_name="split")
    trained_on = []
    for batch in DataLoader(ds, batch_size=batch_size, collate_fn=lambda x: x):
        trained_on.extend(batch)

    assert set(loaded) == set(data)
    if min_max_cachehits[0] == min_max_cachehits[1] == 1:
        assert set(trained_on) == set(data)
