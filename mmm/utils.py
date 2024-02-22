from pathlib import Path
import logging
from functools import partial
import os
import pprint
import json
import hashlib

from typing import List, TypeVar, Type, Dict, Tuple, Literal
import numpy as np
import torch
import torch.nn.functional as nnF

from mmm.BaseModel import BaseModel

from shutil import rmtree
from pathlib import Path

from mmm.BaseModel import BaseModel
from mmm.logging.type_ext import StepMetricDict


def check_streamlit():
    """
    Function to check whether python code is run within streamlit or not.
    """
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if not get_script_run_ctx():
            use_streamlit = False
        else:
            use_streamlit = True
    except ModuleNotFoundError:
        use_streamlit = False
    return use_streamlit


def convert_tile_into_patchbatch(region: torch.Tensor, patch_size=224, stride=224) -> Tuple[torch.Tensor, int, int]:
    """
    Converts a large tensor into a batch of smaller patches.

    If the region is not divisable by the stride, it will be padded with zeros *in-place*.

    That means that the input tensor might have a different shape after applying this function.

    Returns:
        torch.Tensor: A tensor of shape [B, C, patch_size, patch_size] where B is the number of patches.
        int: The number of patches in the width dimension (columns).
        int: The number of patches in the height dimension (rows).

    >>> import torch
    >>> from mmm.utils import convert_tile_into_patchbatch
    >>> testtensor = torch.Tensor(
    ...     [
    ...         [1, 1],
    ...         [1, 1],
    ...         [2, 2],
    ...         [2, 2],
    ...         [3, 3],
    ...         [3, 3],
    ...     ]
    ... )
    >>> patches, c, r = convert_tile_into_patchbatch(testtensor.unsqueeze(0), patch_size=2, stride=2)
    >>> assert False not in (patches[0, 1] == torch.Tensor([[2., 2.],[2., 2.]]))
    >>> c
    3
    >>> r
    1

    """

    region = region.unsqueeze(0)

    # Calculate padding size for height and width
    # h, w = region.shape[-2:]
    # pad_h = (stride - h % stride) % stride
    # pad_w = (stride - w % stride) % stride
    # assert pad_h == 0 and pad_w == 0
    # if pad_h > 0 or pad_w > 0:
    #     region = nnF.pad(region, (0, pad_w, 0, pad_h), mode='constant', value=0)

    # torch.Size([1, C, patches_by_height(rows), patches_by_width(columns), patch_size, patch_size])
    # patches = region.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = region.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # return patches
    rows, columns = patches.shape[2], patches.shape[3]

    # Transform into a batch of patches [B, C, patch_size, patch_size]
    patchbatch = patches.reshape(1, 3, -1, patch_size, patch_size).squeeze(dim=0)
    # patchbatch
    return patchbatch.permute(1, 0, 2, 3), rows, columns


def make_divisable_by(x: int, by: int = 32):
    """Increases number to be divisable by `by`"""
    return x + ((by - x) % by)


def unique_str_hash(*args, **kwargs):
    """
    Currently, the order of keyword argument matter
    """
    argumenttuple = tuple(args) + tuple([x for keyvalue in kwargs.items() for x in keyvalue])
    argumenthasher = hashlib.md5()
    for arg in argumenttuple:
        # Different types might need to be hashed differently
        if isinstance(arg, Path):
            argumenthasher.update(arg.absolute().__str__().encode())
        else:
            argumenthasher.update(str(arg).encode())
    return argumenthasher.hexdigest()


def get_default_cachepath(folder_name: str = "default", implicitly_create=False, based_on="ML_DATA_CACHE") -> Path:
    assert based_on in os.environ, f"Set the cache_path manually or set it using the {based_on} env var"
    cache_path = Path(os.getenv(based_on, default=f"./{folder_name}"))
    res = cache_path / folder_name
    if implicitly_create and not res.exists():
        res.mkdir(parents=True)
    return res


def disk_cacher(cache_path: Path | Literal["local", "shared"] = "local", disable: bool = False):
    """
    Can cache the result of pure functions.
    In other words, the result needs to be unique w.r.t. the unique combination of function name and its parameters.

    If cache_path is "local", then the local cache path by convention will be used.
    If cache_path is "shared", then the shared cache path by convention will be used.

    It will put a hash.pkl file with the pickled function result in the cache path.
    The hash results from the stringified parameters and the stringified values of the keyword parameters.

    If its fine for you to cache into RAM, use `functools.lru_cache` instead.
    """
    if cache_path == "local":
        cache_path = get_default_cachepath("pure_functions", based_on="ML_DATA_CACHE")
    elif cache_path == "shared":
        cache_path = get_default_cachepath("cache/pure_functions", based_on="ML_DATA_OUTPUT")

    def function_wrapper(func):
        def wrapper(attempt: int, *args, **kwargs):
            function_cache_dir = cache_path / func.__name__
            if not function_cache_dir.exists():
                function_cache_dir.mkdir(parents=True)

            # If args already known: return cached result
            func_hash = unique_str_hash(*args, **kwargs)
            p = function_cache_dir / f"{func_hash}.pkl"
            if p.exists() and not disable:
                try:
                    with open(p, "rb") as f:
                        return torch.load(f)
                except EOFError as e:
                    # This cache seems to be invalid, remove it!
                    logging.warn(f"Removing and recomputing {p} because of exception {e}, {attempt=}")
                    os.remove(p)
                    # And compute it again
                    if attempt > 0:
                        raise Exception(f"Using cache failed with {e} for {p=} {args=} {kwargs=}")
                    else:
                        return wrapper(attempt + 1, *args, **kwargs)

            # Else: execute function and return cache its result:
            function_result = func(*args, **kwargs)
            # and save
            with open(p, "wb") as f:
                torch.save(function_result, f)
            return function_result

        return partial(wrapper, 0)

    return function_wrapper


def remove_folder_blocking_if_exists(folder: Path):
    if folder.exists():
        rmtree(folder)
        while folder.exists():
            pass  # Wait for rmtree to finish cleaning up


T = TypeVar("T", bound=BaseModel)


def load_config_from_str(basemodel: Type[T], config_str: str, verbose=True) -> T:
    if config_str:
        logging.debug(config_str)
        config = basemodel(**json.loads(config_str))
    else:
        config = basemodel()

    if verbose:
        # Only import if user wants to know the difference of the config to its default!
        from deepdiff import DeepDiff

        differences = DeepDiff(basemodel().dict(), config.dict())
        if differences:
            diff_dict = differences.to_dict()
            print(f"Using adapted default config with {list(diff_dict.keys())}! (see logging.debug for details)")
            logging.debug(diff_dict)
        else:
            print(f"Using default config!")

    return config


def load_config_from_json5(basemodel: Type[T], file_path: Path, verbose=True) -> T:
    import json5

    with open(file_path, "r") as f:
        return load_config_from_str(basemodel, json.dumps(json5.load(f)), verbose=verbose)


def load_config_from_env(basemodel: Type[T], config_env_name: str, verbose=True) -> T:
    """
    If an environment variable with name `config_env_name` exists,
    will return the basemodel with adaptions made from the json-content of that environment variable.
    Otherwise, returns the default model.

    If `verbose`, will print the differences to the base configuration.
    """
    if config_env_name in list(os.environ.keys()):
        json_string = os.getenv(config_env_name)
        assert json_string is not None, f"{config_env_name} environment variable was not found"
    else:
        json_string = ""
    return load_config_from_str(basemodel, json_string, verbose=verbose)


def flatten_list_of_dicts(step_metrics: List[StepMetricDict]) -> StepMetricDict:
    """
    Concatenates arrays for every top-level key. For example,

    >>> import numpy as np
    >>> from mmm.utils import flatten_list_of_dicts
    >>> flatten_list_of_dicts([
    ...     {"a": np.array([1, 2]), "b": np.array([1, 2])},
    ...     {"a": np.array([3, 4])}
    ... ])
    {'a': array([1, 2, 3, 4]), 'b': array([1, 2])}
    """
    metric_arrs: Dict[str, List[np.ndarray]] = {}
    for d in step_metrics:
        for k, v in d.items():
            if k not in metric_arrs:
                metric_arrs[k] = []

            metric_arrs[k].append(v)

    metrics = {k: np.concatenate(arrs) for k, arrs in metric_arrs.items()}
    return metrics


def recursive_equality(e1, e2, approx=False) -> bool:
    """
    Compares objects. Supports torch.Tensor.
    """
    if type(e1) is not type(e2):
        return False

    # If e1 has a __len__ method, the lengths of e1 and e2 must be equal
    try:
        if len(e1) != len(e2):
            return False
    except TypeError:
        pass

    if isinstance(e1, dict):
        if set(e1.keys()) != set(e2.keys()):
            return False
        # Assumes equal order
        return False not in [recursive_equality(v1, v2) for v1, v2 in zip(e1.values(), e2.values())]
    elif isinstance(e1, list) or isinstance(e1, tuple):
        return False not in [recursive_equality(x1, x2) for x1, x2, in zip(e1, e2)]
    elif isinstance(e1, torch.Tensor):
        if approx:
            return torch.allclose(e1, e2)
        else:
            return torch.equal(e1, e2)
    elif isinstance(e1, np.ndarray):
        if approx:
            return np.allclose(e1, e2)
        else:
            return np.array_equal(e1, e2)
    elif e1 != e1:
        # Only true for nan values
        return np.isnan(e1) and np.isnan(e2)
    else:
        return e1 == e2


# Plotting


def create_sample_efficiency_plot_from_wandb(
    run_id: str,
    project_name: str,
    downstream_tasks: List,
    entity: str = "tissue-concepts",
    metric: str = "acc",
):
    """
    Function to create a sample-efficiency plot between different downstream tasks

    run_id: id of the wandb run to import data from
    project_name: project in which the run is stored

    downstream_tasks: ['str', 'str'] list fo tasks. Example: ["breakhis", "bach"]
    entitiy: wandb entitiy. Ususally tissue-concepts
    metric: metric to be plottet at y-axis

    return: plotly figure object
    """
    import wandb
    import re
    import plotly.graph_objects as go

    api = wandb.Api()
    run = api.run(f"{entity}/{project_name}/{run_id}/")

    summary = run.summary
    summary_keys = list(summary.keys())

    fig = go.Figure()

    for task in downstream_tasks:
        r = re.compile(f"epoch_tl.bestval(.*){task}(.*)_{metric}")
        found = list(filter(r.match, summary_keys))
        data = [[int(entry.split(f"{task}")[-1].split("_")[0]), summary[entry]] for entry in found]

        data.sort()
        x = [a[0] for a in data]
        y = [a[1] for a in data]

        fig.add_trace(go.Scatter(x=x, y=y, name=task))

    fig.update_layout(xaxis_title="Percent Samples", yaxis_title=f"best overall {metric}")

    return fig


def find_missing_torchrun_envvars() -> list[str]:
    torchrun_envvars = [
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
    ]
    return [var for var in torchrun_envvars if var not in os.environ]
