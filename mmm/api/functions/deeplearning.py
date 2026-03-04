import json
from copy import deepcopy
from typing import Hashable, Literal

import logfire
import numpy as np
import redis
import wandb
from m3_sdk.DistributedPath import DistributedPath
from m3_sdk.models import TrainingDoneEvent
from m3_sdk.types import CompressType
from pydantic import Field

from mmm.api.data import KVReprCohort, SubjDataset
from mmm.api.Finetuner import FineTuner
from mmm.api.functions.ApiFunction import ApiFunction
from mmm.api.models import MSubject, Prediction
from mmm.api.mtl_adapter import ADAPTERS, LabelingConfig, MTLAdapterConfig
from mmm.api.WorkerState import WorkerState
from mmm.mmm_types.LabelType import LabelType
from mmm.settings import mtl_settings


def _suggest_modulename_for_cohort(cohort: KVReprCohort, lbl_type: str) -> str:
    return f"{cohort.cfg.for_data.for_input}|{cohort.cfg.for_data.for_label}|{lbl_type}"


def _load_subjects(subjects: list[MSubject | str | bytes], kv) -> list[MSubject]:
    res = []

    for subj in subjects:
        if isinstance(subj, (str, bytes, np.str_)):
            # The subject is a key to a dictionary in Redis
            res.append(MSubject.model_validate_json(kv.get(subj)))
        else:
            res.append(subj if isinstance(subj, MSubject) else MSubject.model_validate(subj))

    if len(set([s.id for s in res])) != len(subjects):
        raise ValueError("Subjects must have unique ids")

    return res


class Finetune(ApiFunction):
    class Args(ApiFunction.Args):
        cohorts: list[KVReprCohort.Config]
        finetuning_id: str = Field(description="multiple tasks result in the same model")
        cfg: FineTuner.Config
        wandb_args: dict | None = None
        wandb_artifacts: list[str] = Field(default_factory=lambda: [], description="List of wandb artifact URIs")
        num_loops: None | int = Field(
            default=None, description="the number of loops to train the model, None for as many as cfg specifies"
        )
        eventqueue: str = Field(default="", description="The event queue where training events are sent")
        lock_for_distributed: bool = Field(
            default=True, description="If True, a lock is created in Redis to prevent multiple identical training jobs."
        )

    class Results(ApiFunction.Results):
        state: Literal["done", "ongoing", "worked"]

    @staticmethod
    def invoke(args: Args, ws: WorkerState, kv: redis.Redis) -> Results:
        """
        Finetunes an adapter for the worker's foundation model given a dataset of cached subjects.

        If a model training for finetuning_id is already running, the task will return None.
        """
        lock_key = f"{mtl_settings.ongoing_training_prefix}:{args.finetuning_id}" if args.lock_for_distributed else ""
        if lock_key and kv.exists(lock_key):
            logfire.info(
                "Model {finetuning_id} is already being trained, skipping finetuning",
                finetuning_id=args.finetuning_id,
            )
            return Finetune.Results(state="ongoing")

        if FineTuner.is_done(args.finetuning_id, args.cfg, kv):
            return Finetune.Results(state="done")

        trainer: FineTuner = FineTuner(args.cfg, ws.fm, args.finetuning_id, lock_name=lock_key, kv=kv)
        trainer.eventqueue = args.eventqueue if args.eventqueue else args.finetuning_id

        for cohort_cfg in args.cohorts:
            with logfire.span("Setting up cohort for label {label}", label=cohort_cfg.for_data.for_label):
                cohort: KVReprCohort = KVReprCohort(cohort_cfg, for_model=ws.fm, kv=kv)
                lbl_type: str = cohort.get_labeling_config().determine_type_of_label(lbl := cohort.get_label())
                mtltask = ADAPTERS[LabelType.from_string(lbl_type)].build_task(
                    ADAPTERS[LabelType.from_string(lbl_type)].Config(),
                    cohort,
                    ws.fm,
                    _suggest_modulename_for_cohort(cohort, lbl_type),
                    cohort.get_extra(),
                )
                trainer.add_mtl_task(mtltask)

        logfire.info("Training {tasks}", tasks=[t.get_name() for t in trainer.mtl_tasks])
        wandb_args = dict(mode="disabled") if args.wandb_args is None else deepcopy(args.wandb_args)
        wandb_args["name"] = f"Finetune {wandb_args.get('name', '')}"
        wandb_run = wandb.init(**wandb_args, settings=wandb.Settings(quiet=True))
        original_default_log_folder = mtl_settings.default_log_folder
        if args.wandb_args is not None:
            mtl_settings.default_log_folder = DistributedPath.from_string(wandb_run.dir) / "m3logs"
            logfire.info(
                "W&B logging to {wandb_url}, locally: {logfolder}",
                wandb_url=wandb_run.url,
                logfolder=mtl_settings.default_log_folder,
            )
            mtl_settings.default_log_folder.upath().mkdir(exist_ok=True, parents=True)

        for artifact in args.wandb_artifacts:
            wandb_run.use_artifact(artifact)

        training_result = trainer.fit(num_loops=args.num_loops)
        for task in trainer.mtl_tasks:
            task.cohort.terminate_workers()

        if args.wandb_args is not None:
            wandb.finish()
            mtl_settings.default_log_folder = original_default_log_folder
        # Delete the lock
        if lock_key:
            kv.delete(lock_key)
        if args.eventqueue:
            mtl_settings.kv.publish(
                channel=args.eventqueue,
                message=TrainingDoneEvent().model_dump_json(),
            )
        return Finetune.Results(state="worked")


class Predict(ApiFunction):
    """
    Enables making prediction with pretrained heads via `for_labels` or with finetuned models via `finetuning_id`.
    """

    class Args(ApiFunction.Args):
        finetuning_id: str = Field(
            default="",
            description="""If given, will use shared blocks from this finetuning 
instead of their counterparts in the pretrained foundation model.""",
        )
        subjects: list[MSubject | str | bytes] = Field(
            ...,
            examples=[
                [{"data": {"image": "http://address:8080/input.jpg"}}],
                ["subject:Manfred", "subject:Lisa"],
                [b"subject:Manfred", b"subject:Lisa"],
            ],
        )
        t: float = 0.0
        for_labels: None | dict[str, str] = Field(
            default=None,
            description="""If None, finetuning_id has to point to a model with tasks with names
such as 'input_name|label_name|labeltype'. Otherwise, should be a mapping from a label name in the label_config
to a task name.""",
        )
        commit_to_subject_key: bool = Field(default=False, description="Whether to save the predictions back to the DB")
        label_config: LabelingConfig
        checkpoint: str = Field(default="latest", description="Only relevant for finetuned models")
        adapter_cfg: MTLAdapterConfig | None = None

        compress_type: CompressType = CompressType.rgbimage
        compress_batchsize: int = 1
        compress_num_workers: int = 0

    class Results(ApiFunction.Results):
        predictions: list[Prediction]

    @staticmethod
    @logfire.instrument("Predicting for model {args.finetuning_id}")
    def invoke(args: Args, ws: WorkerState, kv: redis.Redis) -> Results:
        from io import BytesIO

        import torch

        from mmm.api.models import Prediction, Repr
        from mmm.mmm_types.LabelType import LabelType
        from mmm.mtl_modules.tasks.MTLTask import MTLTask

        subjects = _load_subjects(args.subjects, kv)
        if args.finetuning_id:
            exportdict = torch.load(
                BytesIO(kv.get(f"{mtl_settings.adapter_prefix}:{args.finetuning_id}:{args.checkpoint}:model")),
                weights_only=False,
            )
            # Match the tasks from exportdict
            for_labels = (
                {
                    k.split("|")[1]: k
                    for k, v in exportdict.items()  # k -> input_name|label_name|labeltype
                    if isinstance(v, MTLTask)
                }
                if args.for_labels is None
                else args.for_labels
            )
        else:
            assert args.for_labels is not None, "If no finetuning_id is given, for_labels has to be set"
            exportdict, for_labels = {}, args.for_labels

        with logfire.span("Building model {finetuning_id}", finetuning_id=args.finetuning_id):
            shared_blocks_finetuned = {
                k: v.set_device(ws.fm.device) for k, v in exportdict.items() if not isinstance(v, MTLTask)
            }
            not_finetuned_keys = [k for k in ws.fm.get_sharedblock_keys() if k not in shared_blocks_finetuned]
            logfire.info(
                "Loaded finetuned shared blocks for prediction {finetuned} and not fine-tuned: {not_finetuned}",
                finetuned=list(shared_blocks_finetuned.keys()),
                not_finetuned=not_finetuned_keys,
            )
            for k in not_finetuned_keys:
                shared_blocks_finetuned[k] = ws.fm[k]

            # Finetuned tasks should use the finetuned shared blocks
            mtl_tasks: dict[str, MTLTask] = {
                k: (v.set_device(ws.fm.device), shared_blocks_finetuned)
                for k, v in exportdict.items()
                if isinstance(v, MTLTask)
            }
            # For pretrained task heads, the original shared blocks have to be used
            mtl_tasks.update({k: (ws.fm[k], ws.fm) for k in ws.fm.get_task_keys()})

        all_preds: dict[str, Prediction] = {}
        # Compress all subjects into a dictionary of the form subject_id->data_key->context->repr
        inputs: dict[str, dict[str, dict[Hashable, Repr]]] = {}

        # Already compressed instances are part of the set compressed:for_type
        set_name = f"compressed:{args.compress_type.value}"
        with kv.pipeline() as pipe:
            for subject in subjects:
                pipe.sismember(set_name, subject.id if subject.id else "")
            already_compressed = pipe.execute()
        already_compressed = [
            subject.id for subject, is_compressed in zip(subjects, already_compressed) if is_compressed
        ]
        if to_compress := [subject for subject in subjects if subject.id not in already_compressed]:
            for i, batch in SubjDataset(SubjDataset.Config(), data=to_compress).tokenize(
                ws.fm,
                for_type=args.compress_type,
                batchsize=args.compress_batchsize,
                num_workers=args.compress_num_workers,
            ):
                for repr in batch:
                    subject = repr.meta["msubject"]
                    data_key = repr.meta["data_key"]
                    inputs.setdefault(subject.id, {}).setdefault(data_key, {})[repr.meta["context"]] = repr
        logfire.info("{n_done} subj compressed, {n_do} to do", n_done=len(already_compressed), n_do=len(to_compress))

        with kv.pipeline() as pipe:
            for subject_id in already_compressed:
                # Load all representations which are compressed at "repr:subject.id:compress_type"
                pipe.hgetall(f"repr:{subject_id}:{args.compress_type.value}")
            representations = pipe.execute()

        for subject_id, reprs in zip(already_compressed, representations):
            for context_bytes, repr_bytes in reprs.items():
                context, repr = Repr.resolve_context(context_bytes), Repr.from_bytes(repr_bytes)
                data_key = repr.meta["data_key"]
                inputs.setdefault(subject_id, {}).setdefault(data_key, {})[context] = repr

        @logfire.instrument("Predicting for subject {for_subject.id}")
        def process_subject(for_subject):
            results = []
            for label_key, task_name in for_labels.items():
                label_type = args.label_config.get_parsed()[label_key]["type"]
                task, shared_blocks = mtl_tasks[task_name]
                data_key = args.label_config.get_parsed()[label_key]["to_name"][0]
                with torch.inference_mode():
                    adapter = ADAPTERS[LabelType.from_string(label_type)]
                    adapter_cfg = adapter.Config() if args.adapter_cfg is None else args.adapter_cfg
                    rs = adapter.predict(
                        adapter_cfg,
                        ws.fm,
                        shared_blocks,
                        task,
                        for_subject,
                        args.t,
                        label_key,
                        data_key,
                        list(inputs[for_subject.id][data_key].values()),
                        args.label_config,
                    )
                results.extend(rs)
            if scores := [r.score for r in results if r.score is not None]:
                score = sum(scores) / len(scores)
            else:
                score = None
            inputs.pop(for_subject.id)

            res = Prediction(
                result=results,
                score=score,
                model_version=f"{args.finetuning_id}|{args.checkpoint}"
                if args.finetuning_id
                else ws.fm.get_identifier(),
            )
            if args.commit_to_subject_key:
                subject_key = f"{mtl_settings.subj_prefix}:{for_subject.id}"
                subject_dict = json.loads(kv.get(subject_key))
                subject_dict["predictions"] = subject_dict.get("predictions", []) + [res.model_dump()]
                kv.set(subject_key, json.dumps(subject_dict))
            return res

        for subj in subjects:
            if subj.id not in all_preds:
                all_preds[subj.id] = process_subject(subj)
            else:
                logfire.warning("Subject {subject_id} already predicted, skipping", subject_id=subj.id)

        return Predict.Results(predictions=[all_preds[subject.id] for subject in subjects])
