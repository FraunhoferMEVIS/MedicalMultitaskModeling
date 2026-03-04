import itertools
import json
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import logfire
import numpy as np
import wandb
from pydantic import BaseModel, Field
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from mmm.data_loading.MTLDataset import M3LoaderInfo
from mmm.settings import mtl_settings

SuperCaseType = TypeVar("SuperCaseType")
SubCaseType = TypeVar("SubCaseType")
MAX_DEBUG_LOGS = 1000


@dataclass
class CacheItem:
    # The type here depends on the usecase
    # For example, for WSI it might be a list of patches
    # For 3D images, it might be the volume before 3D patch extraction
    original: Any
    name: str | None = None  # Optional name for debugging

    def __hash__(self) -> int:  # requires for usage in sets
        return id(self)  # Use object's id as hash

    def __eq__(self, other) -> bool:  # the default __eq__ compares dataclasses by each of their fields
        return self is other  # Equality based on identity


T = TypeVar("T")


class CountedList(Generic[T]):
    """
    A list-like container that keeps track of the number of times each item has been accessed.

    Enables efficient vectorized operations on the counts using numpy.
    """

    def __init__(self, initial_buffer_size: int = 1):
        # -1 indicates an empty slot
        self.counts = np.zeros(initial_buffer_size, dtype=np.int64) - 1
        self.items: list[T | None] = [None] * initial_buffer_size

    def average_count(self) -> float:
        valid_counts = self.counts[self.counts != -1]
        if valid_counts.size == 0:
            return 0.0
        return np.mean(valid_counts).item()

    def append(self, item: T):
        empty_indices = np.where(self.counts == -1)[0]
        if empty_indices.size == 0:
            # No empty slot, need to expand
            self.counts = np.append(self.counts, -1)
            self.items.append(None)
            first_empty_index = len(self.counts) - 1
        else:
            first_empty_index = empty_indices[0]

        self.counts[first_empty_index] = 0
        self.items[first_empty_index] = item

    def extend(self, items: Sequence[T]):
        for item in items:
            self.append(item)

    def max(self) -> tuple[T, int] | None:
        if len(self) == 0:
            return None
        idx = np.argmax(self.counts)
        assert (res := self.items[idx]) is not None, "No items available in CountedList"
        return res, self.counts[idx].item()

    def sample(self, increase_counter: bool = True, above: int | None = None) -> tuple[T, int] | None:
        if above is not None:
            valid = np.where(self.counts > above)[0]
        else:
            valid = np.where(self.counts != -1)[0]
        if not valid.size > 0:
            return None
        idx = random.choice(valid)
        if increase_counter:
            self.counts[idx] += 1
        return self.items[idx], self.counts[idx].item()

    def __len__(self) -> int:
        return np.sum((self.counts != -1).astype(int)).item()

    def remove(self, item: T):
        idx = self.items.index(item)
        self.counts[idx] = -1
        self.items[idx] = None

    def __repr__(self) -> str:
        return f"CountedList({[item for item in self.items if item is not None]})"


class CachingSubCaseDSV2(IterableDataset, Generic[SubCaseType]):
    """
    - For DDP, repeats the same supercase indices on each rank
    - For deterministic yielding of each supercase, use min_max_cachehits=(1, 1)
    """

    class Config(BaseModel):
        cache_items_per_hit: int = Field(
            default=10,
            description="""The number of cache hits per loaded Cacheitem is tuned automatically
if a range is given in min_max_cachehits.

""",
        )
        min_max_cachehits: tuple[int, int] = (1, 20)

    def get_dataloader_defaults(self):
        return dict(prefetch_factor=4)

    def __init__(
        self,
        cfg: Config,
        supercases: Sequence[SuperCaseType] | Dataset[SuperCaseType],
        cacher: Callable[[SuperCaseType], CacheItem | list[CacheItem]],
        cache_loader: Callable[[CacheItem, int], Sequence[SubCaseType]],
    ):
        self.cfg = cfg
        self.supercases = supercases
        self.cache: CountedList[CacheItem] = CountedList(
            initial_buffer_size=self.cfg.min_max_cachehits[1] * self.cfg.cache_items_per_hit
        )

        # Set by framework
        self.mmm_info: M3LoaderInfo | None = None

        self.cacher = cacher
        self.cache_loader = cache_loader

        self.cachehits_per_load = (
            self.cfg.min_max_cachehits[1] - self.cfg.min_max_cachehits[0]
        ) // 2 + self.cfg.min_max_cachehits[0]
        self.tune_for: int = 1

    def compute_supercase_indices(self) -> Iterable[int]:
        dataset_length = len(self.supercases)  # type: ignore
        res: list[int] = list(range(dataset_length))
        random.shuffle(res)
        return res

    def delete_item(self) -> CacheItem | None:
        if (
            item_above_hits := self.cache.sample(increase_counter=False, above=self.cachehits_per_load - 1)
        ) is not None:
            removal_item, hits = item_above_hits
        else:
            highest = self.cache.max()
            assert highest is not None, "No items available to remove from cache"
            removal_item, hits = highest
            if hits <= 0:
                logfire.warning(
                    "No items with cachehits in cache {maxhits}. Skipping deletion at cachesize {cache_size}",
                    maxhits=np.max(self.cache.counts).item(),
                    cache_size=len(self.cache),
                )
                return None

        with mtl_settings.kv.pipeline() as pipe:
            assert self.mmm_info is not None
            k = self.mmm_info.get_cache_key(f"cache_deletions")
            pipe.lpush(
                k,
                json.dumps(
                    {
                        "timestamp": time.perf_counter(),
                        "cachehits": hits,
                        "name": removal_item.name,
                        "current_cache_size": len(self.cache),
                        "current_target_hits": self.cachehits_per_load,
                        "worker_id": w.id if (w := get_worker_info()) is not None else "main",
                    }
                ),
            )
            pipe.ltrim(k, 0, MAX_DEBUG_LOGS - 1)
            pipe.execute()

        self.cache.remove(removal_item)
        return removal_item

    def tune_cachehits_per_load(self):
        assert self.mmm_info is not None, "Cohort settings must be set before tuning cachehits_per_load."

        def adjust(up: bool, load_times: list[float]):
            # Going up in case of high load times is always fast independent of tune_for
            self.tune_for += len(load_times)
            mtl_settings.kv.delete(self.mmm_info.get_cache_key("batch_extraction_s"))

            if up:
                if self.cachehits_per_load == self.cfg.min_max_cachehits[1]:
                    logfire.warning(
                        "Average load time above threshold, but cachehits_per_load already at max, not increasing"
                    )
                else:
                    self.cachehits_per_load += 1
                    logfire.debug(
                        "Increasing cachehits_per_load to {cachehits_per_load}",
                        cachehits_per_load=self.cachehits_per_load,
                    )
            else:
                self.cachehits_per_load = max(self.cachehits_per_load - 1, self.cfg.min_max_cachehits[0])

            wandb.log(
                {
                    f"{self.mmm_info.task_name}/{self.mmm_info.split_name}_cachehits": self.cachehits_per_load,
                    f"{self.mmm_info.task_name}/{self.mmm_info.split_name}_tune_for": self.tune_for,
                },
            )
            mtl_settings.kv.set(self.mmm_info.get_cache_key("cache_hits_per_load"), self.cachehits_per_load)

        worker_id = worker_info.id if (worker_info := get_worker_info()) is not None else 0
        # Only tune if there is a range of cachehits to tune over
        if self.cfg.min_max_cachehits[0] != self.cfg.min_max_cachehits[1]:
            if worker_id == 0:
                load_times = [
                    float(t.decode())
                    for t in mtl_settings.kv.lrange(
                        name=self.mmm_info.get_cache_key("batch_extraction_s"), start=0, end=-1
                    )  # type: ignore
                ]

                if max(load_times, default=0) > 1.0:
                    # There is a large outlier, increase cachehits_per_load immediately!
                    with logfire.span(
                        "Detected large outlier in load times {max_load_time}", max_load_time=max(load_times, default=0)
                    ):
                        adjust(up=True, load_times=load_times)
                elif len(load_times) > self.tune_for:
                    avg_load_time = np.mean(load_times).item()
                    # Increase the samples taken for tuning over time
                    with logfire.span(
                        "Average load time: {avg_load_time}", avg_load_time=avg_load_time, load_times=load_times
                    ):
                        adjust(up=avg_load_time > 0.05, load_times=load_times)
            else:
                self.load_cache_tune()

    def load_cache_tune(self):
        assert self.mmm_info is not None, "Cohort settings must be set before iteration."
        if (saved := mtl_settings.kv.get(self.mmm_info.get_cache_key("cache_hits_per_load"))) is not None:
            self.cachehits_per_load = int(saved.decode())  # type: ignore
            logfire.info(
                "Loading cachehits_per_load from {key}: {cachehits_per_load}",
                key=self.mmm_info.get_cache_key("cache_hits_per_load"),
                cachehits_per_load=self.cachehits_per_load,
            )

    def compute_target_cachesize(self) -> int:
        return self.cachehits_per_load * self.cfg.cache_items_per_hit

    def __iter__(self):
        assert self.mmm_info is not None, "Cohort settings must be set before iteration."
        running_batch, remaining_batchsize = [], self.mmm_info.batch_size
        assert remaining_batchsize is not None, "Batch size must be set before iteration."
        self.load_cache_tune()

        if (worker_info := get_worker_info()) is not None:
            worker_id: int = worker_info.id
            num_workers: int = worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1

        for supercase_idx in self.compute_supercase_indices():
            self.tune_cachehits_per_load()
            # Delete one item if there are items above the max_cachehits threshold
            # while self.cache.len_above(self.cfg.min_max_cachehits[1] - 1) > 0:
            #     self.delete_item()

            while len(self.cache) >= self.compute_target_cachesize():
                if self.delete_item() is None:
                    break  # Deletion is not a good idea anymore

            # Add new cache item (after deletion to avoid immediately deleting the new one)
            with logfire.span(
                "Caching supercase {supercase_idx}",
                supercase_idx=supercase_idx,
            ):
                if isinstance(new_stuff := self.cacher(self.supercases[supercase_idx]), list):
                    self.cache.extend(new_stuff)
                else:
                    self.cache.append(new_stuff)

            for cache_takes in itertools.count():
                if (avg_count := self.cache.average_count()) >= self.cachehits_per_load:
                    mtl_settings.kv.publish(
                        channel=self.mmm_info.get_cache_key(f"cache_take_debug{worker_id}"),
                        message=json.dumps(
                            {
                                "cache_takes": cache_takes,
                                "average_cache_hits": avg_count,
                                "cache_size": len(self.cache),
                                "worker_id": worker_id,
                                "cachehits_per_load": self.cachehits_per_load,
                                "num_workers": num_workers,
                            }
                        ),
                    )
                    break

                # Continue building running batch from cache
                if (sample_result := self.cache.sample()) is not None:
                    next_cache_item, hits = sample_result

                    if hits >= self.cfg.min_max_cachehits[1]:
                        logfire.warning(
                            "Removing {name}: it has {hits} hits, allowed: {min_hits}-{max_hits}, size={cache_size}",
                            name=next_cache_item.name,
                            hits=hits,
                            min_hits=self.cfg.min_max_cachehits[0],
                            max_hits=self.cfg.min_max_cachehits[1],
                            cache_size=len(self.cache),
                            hits_per_load=self.cachehits_per_load,
                        )
                        self.cache.remove(next_cache_item)

                    with logfire.span(
                        "Loading {name} with {hits} hits from cache, remaining_batchsize={remaining_batchsize}",
                        name=next_cache_item.name,
                        hits=hits,
                        remaining_batchsize=remaining_batchsize,
                        cachehits_per_load=self.cachehits_per_load,
                        _level="debug",
                    ):
                        new_batch_items = self.cache_loader(next_cache_item, remaining_batchsize)
                    remaining_batchsize -= len(new_batch_items)
                    running_batch.extend(new_batch_items)

                    if remaining_batchsize == 0:
                        assert (remaining_batchsize := self.mmm_info.batch_size) is not None
                    elif remaining_batchsize < 0:
                        raise ValueError(f"{self=} returned more items than requested: {len(new_batch_items)=}")

                else:
                    logfire.warning(
                        "Cache with size {cache_size} is empty, cannot sample more items ({remaining_batchsize} left)",
                        cache_size=len(self.cache),
                        remaining_batchsize=remaining_batchsize,
                    )
                    break

                while running_batch:
                    yield running_batch.pop(0)

        while running_batch:
            yield running_batch.pop(0)
