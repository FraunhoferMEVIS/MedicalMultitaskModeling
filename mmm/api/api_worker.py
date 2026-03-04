"""
The module contains stateless tasks.
They are designed to be compatible with task runners such as Celery.
"""

from mmm.api.functions.compression import CacheInstances, CacheSubjects
from mmm.api.functions.deeplearning import Finetune, Predict
from mmm.settings import mtl_settings

from .celery import app, ws


def get_kv(kv):
    """Helper to pass the key-value store around. Used for automatic testing."""
    return mtl_settings.kv if kv is None else kv


@app.task
def cache_subjects(inps: CacheSubjects.Args, kv=None) -> CacheSubjects.Results:
    return CacheSubjects.invoke(inps, ws, get_kv(kv))


@app.task
def get_task_ids(kv=None) -> list[str]:
    inside_model = ws.fm.get_task_keys()
    within_redis = get_kv(kv).keys(f"{mtl_settings.adapter_prefix}:*:model")
    return list(set(inside_model + within_redis))


@app.task
def get_model_info(kv=None) -> dict:
    return {
        "model_id": ws.fm.get_identifier(),
        "shared_blocks": ws.fm.get_sharedblock_keys(),
        "pretrained": ws.fm.get_task_keys(),
        "finetunings": get_kv(kv).keys(f"{mtl_settings.adapter_prefix}:*:model"),
    }


@app.task
def cache_instances(inps: CacheInstances.Args, kv=None) -> CacheInstances.Results:
    return CacheInstances.invoke(inps, ws, get_kv(kv))


@app.task
def predict(predict_args: Predict.Args, kv=None) -> Predict.Results:
    return Predict.invoke(predict_args, ws, get_kv(kv))


@app.task
def finetune(inps: Finetune.Args, kv=None) -> Finetune.Results:
    return Finetune.invoke(inps, ws, get_kv(kv))
