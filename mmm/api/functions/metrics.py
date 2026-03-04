import json
import logging
from copy import deepcopy
from typing import Any

import logfire
import redis

from mmm.api.functions.ApiFunction import ApiFunction
from mmm.api.models import MSubject
from mmm.api.mtl_adapter import ADAPTERS, LabelingConfig, MTLAdapterConfig, process_for_metrics
from mmm.api.WorkerState import WorkerState
from mmm.logging.wandb_ext import remove_wandb_special_chars
from mmm.mmm_types.LabelType import LabelType
from mmm.settings import mtl_settings


class Metric(ApiFunction):
    """
    Computes the metrics for a dataset of subjects. By default, it computes all metrics.

    For multiple annotations it always uses the latest annotation.
    If thats not possible it uses the last one in the list.
    """

    class Args(ApiFunction.Args):
        dataset: list[str | bytes] | str
        for_models: list[str]
        label_config: LabelingConfig
        to_wandb: None | dict = None
        wandb_metric_prefix: str = "test"
        adapter_cfg: MTLAdapterConfig | None = None

    class Results(ApiFunction.Results):
        metrics: dict[str, Any]

    @staticmethod
    def invoke(args: Args, ws: WorkerState, kv: redis.Redis) -> Results:
        if isinstance(args.dataset, str):
            dataset = [
                f"{mtl_settings.subj_prefix}:{subj_id.decode()}" for subj_id in mtl_settings.kv.smembers(args.dataset)
            ]
        else:
            dataset = args.dataset

        subjects: list[MSubject] = [MSubject(**json.loads(mtl_settings.kv.get(subj))) for subj in dataset]

        results = {}  # model -> label -> metric -> value
        with logfire.span("Computing metrics for subjects_len={subjects_len}", subjects_len=len(subjects)) as span:
            for for_model in args.for_models:
                subjects_for_label = process_for_metrics(
                    subjects, for_model_version=for_model, label_config=args.label_config
                )
                for label_name, label_subjects in subjects_for_label.items():
                    logging.info(f"Using {len(label_subjects)=} for {label_name=}")
                    adapter = ADAPTERS[LabelType.from_string(args.label_config.get_parsed()[label_name]["type"])]
                    adapter_cfg = adapter.Config() if args.adapter_cfg is None else args.adapter_cfg
                    results.setdefault(for_model, {})[label_name] = adapter.compute_metrics(
                        adapter_cfg, label_subjects, args.label_config.get_parsed()[label_name]
                    )

        if args.to_wandb is not None:
            import wandb

            wandb.init(**args.to_wandb, settings=wandb.Settings(quiet=True))

            for model_name, model_metrics in results.items():
                for label_name, label_metrics in model_metrics.items():
                    for metric_name, metric_value in label_metrics.items():
                        model_category = model_name.split("|")[-1] if "|" in model_name else model_name
                        wandb.log(
                            {
                                f"{args.wandb_metric_prefix}-{remove_wandb_special_chars(model_category)}-{label_name}/{metric_name}": metric_value
                            },
                            commit=False,
                        )
            wandb.finish()

        return Metric.Results(metrics=results)
