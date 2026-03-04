from __future__ import annotations

import json
import random
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Literal, cast

import logfire
import torchvision.transforms as transforms
from m3_sdk.DistributedPath import DistributedPath
from m3_sdk.types import CompressType
from pydantic import Field
from torch.utils.data import Dataset
from typing_extensions import Annotated

from mmm.api.FoundationModel import FoundationModel
from mmm.api.models import Annotation, MSubject, Repr, _extract_instances_for_input
from mmm.api.mtl_adapter import ADAPTERS, LabelingConfig, MTLAdapter
from mmm.BaseModel import BaseModel
from mmm.data_loading.MTLDataset import MTLDataset, flatten_list, mtl_batch_collate, mtl_collate
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.interactive import pipes
from mmm.mmm_types.LabelType import LabelType
from mmm.settings import mtl_settings
from mmm.torch_ext import CachingSubCaseDS, DeterministicSamplerSignalLast
from mmm.transforms import ApplyToList


class SubjDataset(Dataset[MSubject]):
    """
    If you only want to iterate through the subjects that have a certain label, set `set_active_label`.
    """

    class Config(BaseModel):
        # labeling_config: LabelingConfig | None = None
        pass

    def __init__(self, cfg: Config, data: list[dict | MSubject], labeling_config: LabelingConfig | None = None):
        self.cfg = cfg
        self.labeling = labeling_config

        assert len(data) > 0, "No data provided"

        # Pre-compute which labels occur in which subjects.
        # In other words, which cases should be considered for which label.
        self.labels: dict[str, list[int]] = {}
        self.all_cases: list[MSubject] = []
        self.subject_id_to_subject: dict = {}
        self.ids: list[str | int] = []
        for d in data:
            subj: MSubject = d if isinstance(d, MSubject) else MSubject(**d)
            self.ids.append(subj.id)
            self.subject_id_to_subject[subj.id] = subj
            self.all_cases.append(subj)

            if subj.annotations:
                for annotation in subj.annotations:
                    for result in annotation.result:
                        self.labels.setdefault(result.from_name, []).append(subj.id)

        # Delete duplicates which happen due to multiple annotations per subject.
        self.labels = {k: list(set(v)) for k, v in self.labels.items()}

        self.active_cases = self.all_cases

    def __len__(self):
        return len(self.active_cases)

    def __getitem__(self, index) -> MSubject:
        return self.active_cases[index]

    def set_active_label(self, label_key: str):
        """
        Only retains cases that have the specified label.
        """
        self.active_cases = [x for x in self.all_cases if x.id in self.labels[label_key]]
        return self

    def get_deterministic_cacheds(
        self,
        with_labels: list[str] | None,
        max_instances_per_input: None | int = None,
    ):
        supercase_loader = partial(
            MSubject.extract_instances,
            labeling=self.labeling,
            with_labels=with_labels,
            max_instances_per_input=max_instances_per_input,
        )
        return CachingSubCaseDS(
            supercase_ds=self,
            supercase_loader=supercase_loader,  # type: ignore
            cache_sampler=DeterministicSamplerSignalLast(),
            cfg=CachingSubCaseDS.Config(split_across_workers=True, drain_each_epoch=True, subcase_cache_size=1),
        )

    @staticmethod
    def build_dataloader(cache_ds, **dataloader_kwargs):
        from torch.utils.data import DataLoader

        if "batch_size" not in dataloader_kwargs:
            dataloader_kwargs["batch_size"] = 1

        if "num_workers" not in dataloader_kwargs:
            dataloader_kwargs["num_workers"] = 0

        return DataLoader(
            cache_ds,
            shuffle=False,
            **dataloader_kwargs,
        )

    def tokenize_multitype(
        self,
        model: FoundationModel,
        for_types: list[CompressType],
        batchsize: int,
        with_labels: list[str] | None = None,
        progress_log_callback: Callable[[int, dict[str, Any]], None] | None = None,
        num_workers: int = 0,
        subject_hook: Callable | None = None,
        max_instances_per_input: None | int = None,
        **dataloader_kwargs,
    ):
        """
        subject_hook yields the neural compression of the subject once all its instances have been processed.
        """
        # Increasing num workers will speed up the process, but will disallow debugging
        cache_ds = self.get_deterministic_cacheds(
            with_labels=with_labels,
            max_instances_per_input=max_instances_per_input,
        )
        dataloader = self.build_dataloader(
            cache_ds,
            batch_size=batchsize,
            num_workers=num_workers,
            collate_fn=transforms.Compose([ApplyToList(Repr.to_mmm_image), model.collate_instances]),
            **dataloader_kwargs,
        )
        all_pending_supercases = set(self.ids)

        start_time, data_loading_time_s, case_processing_time_s = time.perf_counter(), 0, 0

        for batch_idx, batch in enumerate(dataloader):
            data_loading_time_s = time.perf_counter() - start_time

            reprs = model.compress_instances(batch, for_representation=for_types)
            model_time_s = time.perf_counter() - start_time - data_loading_time_s

            # list of tuples, where each tuple entry is a repr of one of the requested compress types
            representations = []
            for r_idx in range(len(batch["meta"])):
                reprs_i = []
                buffer = batch["meta"][r_idx].pop("buffer", None)  # remove buffer from meta
                for for_type in for_types:
                    repr_meta = batch["meta"][r_idx].copy() if len(for_types) > 1 else batch["meta"][r_idx]
                    repr_meta["compress_type"] = for_type.value
                    r = Repr(tensor=reprs[for_type][r_idx], meta=repr_meta)
                    r._buffer = buffer  # reattach buffer if it was present
                    reprs_i.append(r)

                representations.append(tuple(reprs_i))

            yield batch_idx, representations
            all_pending_supercases -= set([m["msubject"].id for m in batch["meta"]])

            # for supercase in [m["msubject"] for m in batch["meta"] if "sampler_last_subcase" in m]:
            for supercase in [m["msubject"] for m in batch["meta"] if "sampler_last_subcase" in m]:
                if subject_hook:
                    subject_hook(supercase)

            current_progress = len(self.ids) - len(all_pending_supercases)

            case_processing_time_s = time.perf_counter() - start_time - data_loading_time_s - model_time_s
            if progress_log_callback:
                progress_log_callback(
                    batch_idx,
                    {
                        "data_loading_time_s": data_loading_time_s,
                        "case_processing_time_s": case_processing_time_s,
                        "model_time_s": model_time_s,
                        "current_progress": current_progress,
                        "prepindex": batch_idx,
                    },
                )
            start_time = time.perf_counter()

    def tokenize(
        self,
        model: FoundationModel,
        for_type: CompressType,
        batchsize: int,
        with_labels: list[str] | None = None,
        progress_log_callback: Callable[[int, dict[str, Any]], None] | None = None,
        num_workers: int = 0,
        subject_hook: Callable | None = None,
        max_instances_per_input: None | int = None,
        **dataloader_kwargs,
    ):
        for i, representations in self.tokenize_multitype(
            model=model,
            for_types=[for_type],
            batchsize=batchsize,
            with_labels=with_labels,
            progress_log_callback=progress_log_callback,
            num_workers=num_workers,
            subject_hook=subject_hook,
            max_instances_per_input=max_instances_per_input,
            **dataloader_kwargs,
        ):
            yield i, [repr_tuple[0] for repr_tuple in representations]


class ReprDataset(Dataset[dict[str, Any]]):
    """
    Converts the target task data format to the pretraining multitask task data format.

    The target task format describes multiple labels and inputs per subject.
    For pretraining, one input and label per subject is selected.
    """

    class Config(BaseModel):
        for_input: str
        for_label: str

    def __init__(self, cfg: Config, subj_dataset: SubjDataset, adapter_cfg: MTLAdapter.Config | None = None):
        self.cfg = cfg
        self.subj_dataset = subj_dataset
        self.relevant_ids = self.subj_dataset.labels[self.cfg.for_label]
        assert self.subj_dataset.labeling is not None, f"{self.subj_dataset.labeling=} must be provided for ReprDataset"
        self.label = self.subj_dataset.labeling.get_parsed()[self.cfg.for_label]
        self.extra = self.subj_dataset.labeling.get_extra(label_name=self.cfg.for_label)
        self.labeltype = LabelType.from_string(self.subj_dataset.labeling.determine_type_of_label(self.label))
        self.adapter_cfg = adapter_cfg if adapter_cfg is not None else ADAPTERS[self.labeltype].Config()

    def __len__(self):
        return len(self.relevant_ids)

    def representation_to_mmm(self, repr: Repr, train: bool = False) -> dict[str, Any]:
        res = {
            "image": repr.tensor,
            "meta": {
                "group_id": repr.group_id,
                "repr_meta": {k: v for k, v in repr.meta.items() if k not in ["context", "msubject"]},
                "context": repr.meta["context"],
            },
        }
        subject: MSubject = repr.meta["msubject"]
        if subject.meta:
            res["meta"]["subjectmeta"] = subject.meta

        assert (anno := subject.get_last_updated_annotation()) is not None, f"Subject {subject.id} has no annotations"

        gt_repr: Repr = cast(Annotation, anno).extract_gt(
            self.cfg.for_label, subject, for_instance=repr, labeling=self.subj_dataset.labeling
        )
        return ADAPTERS[self.labeltype].extract_label(
            self.adapter_cfg, res, gt_repr, self.label, train=train, extra=self.extra
        )

    def __getitem__(self, index):
        case_id = self.relevant_ids[index]
        subj: MSubject = self.subj_dataset.subject_id_to_subject[case_id]
        return _extract_instances_for_input(subj, labeling=self.subj_dataset.labeling, input_tag=self.cfg.for_input)


class ReprCohort(TrainValCohort):
    """
    Cohort that can be used for multitask learning, based on the interoperability format.

    This class is intended to be subclassed to customize its default behavior.
    The default behavior is implemented in several methods that can be overridden.
    """

    class Config(TrainValCohort.Config):
        for_data: ReprDataset.Config

    def __init__(self, cfg: Config, subjects: SubjDataset | None, for_model: FoundationModel | None = None) -> None:
        self.cfg = cfg
        self.subj_data = subjects if subjects is not None else None
        self.for_model: FoundationModel | None = for_model

        self.train_source, self.test_source = self.build_source_data()

        train_ds = self.build_mtldataset(self.train_source, train=True)
        val_ds = self.build_mtldataset(self.test_source, train=False)

        super().__init__(cfg, train_ds, val_ds)

    def get_labeling_config(self) -> LabelingConfig:
        return self.subj_data.labeling

    def get_domain_augmentation(self):
        by_domain: dict = {
            "gigapixel": pipes.get_histo_augs,
            "tomographic": pipes.get_mri2d_augs,
            "xray": pipes.get_xray_augs,
            None: pipes.get_weak_default_augs,
        }
        logfire.info(
            "Using domain augmentation for domain {domain} with replay={replay}",
            domain=self.get_extra().domain,
            replay=self.get_extra().replay_augmentations_for_groups,
        )
        return pipes.Alb(
            by_domain[self.get_extra().domain](),
            replay_for_groups=self.get_extra().replay_augmentations_for_groups == True,
        )

    def get_batch_transform(self, train: bool):
        return pipes.ApplyToList(self.get_domain_augmentation()) if train else None

    def collate_fn(self) -> Callable:
        if self.for_model is None:
            from mmm.transforms import UnifySizes

            return transforms.Compose([flatten_list, UnifySizes(divisable_by=32, max_edge_len=512), mtl_collate])
        else:
            return transforms.Compose([flatten_list, self.for_model.collate_instances])

    def get_extra(self):
        if not hasattr(self, "_extra") or self._extra is None:
            self._extra = self.get_labeling_config().get_extra(label_name=self.cfg.for_data.for_label)

        return self._extra

    def get_label(self):
        return self.get_labeling_config().get_parsed()[self.cfg.for_data.for_label]

    def build_mtldataset(self, src_ds: ReprDataset | KVDataset, train: bool) -> MTLDataset:
        labeltype: str = self.get_labeling_config().determine_type_of_label(lbl := self.get_label())

        return ADAPTERS[LabelType.from_string(labeltype)].build_dataset(
            ADAPTERS[LabelType.from_string(labeltype)].Config(),
            src_ds,
            lbl,
            train,
            src_transform=self.get_src_transform(src_ds, train),
            batch_transform=self.get_batch_transform(train),
            collate_fn=self.collate_fn(),
        )

    def get_src_transform(self, src_ds: ReprDataset, train: bool):
        def load_subject(g):
            return list(map(partial(src_ds.representation_to_mmm, train=train), g))

        return load_subject

    def build_source_data(self) -> tuple[ReprDataset, ReprDataset]:
        d = ReprDataset(self.cfg.for_data, self.subj_data)
        return d, d


class KVDataset(Dataset):
    """
    - compressed subjects are hashes
    - by convention, the hashes are prefixed like: "your-prefix:subject-id"
    - the dataset returns all contexts for a given subject
    - the actual values can be retrieved using a list of contexts
    - difference to representation dataset: all representations have fast random access (no caching layer needed)
    """

    def __init__(
        self, adapter_cfg, subject_keys: list[bytes], kv, deserializer: None | Callable = None, length: int = 1
    ):
        logfire.info(
            "Initializing KVDataset with {num_subjects} subjects",
            num_subjects=len(subject_keys),
            subject_keys=subject_keys,
            adapter_config=adapter_cfg,
        )
        self._redis_client = None  # only initialized in __getitem__ for fork safety
        self.kv = kv
        self.subject_keys = subject_keys
        self.deserializer = deserializer
        self.length = length
        self.relevant_ids = list(range(len(self)))
        self.adapter_cfg = adapter_cfg

    def __len__(self):
        return len(self.subject_keys) * self.length

    def retrieve_values_for_contexts(self, hash_key: bytes, fields: list[bytes]) -> list:
        """
        Retrieves all has field values for a given hash and a list of contexts (fields) from the key-value client.

        In a speed test:
        - a list comprehension took 33.8s for 100 subjects.
        - a pipeline took 7.4s for 100 subjects.
        - hmget took 33.8s for 100 subjects.
        """

        with self._redis_client.pipeline() as pipe:
            for ctx in fields:
                pipe.hget(hash_key, ctx)
            if self.deserializer is not None:
                return [self.deserializer(val) for val in pipe.execute()]
            else:
                return pipe.execute()

    def __getitem__(self, idx) -> tuple[bytes, list[bytes]]:
        idx = idx % len(self.subject_keys)
        if self._redis_client is None:
            # For fork safety, we might need to initialize the redis client in the __getitem__ method
            # If training is stuck for unknown reasons, try to create a new client here instead of reusing
            self._redis_client = self.kv

        subject_repr_hash = self.subject_keys[idx]  # values such as repr:subject_id:token
        contexts = self._redis_client.hkeys(subject_repr_hash)
        return subject_repr_hash, contexts, self

    def representation_to_mmm(self, hash_key: bytes, ctx: bytes, repr: Repr, group_id=None):
        """
        group_id may be supplied for optimization. If None, it will be extracted from the hash_key.
        """
        return {
            "image": repr.tensor,
            "meta": {
                "repr_hash": hash_key,
                "repr_meta": {k: v for k, v in repr.meta.items() if k not in ["context", "msubject"]},
                "context": Repr.resolve_context(ctx),
                "group_id": hash_key.decode().split(":")[1] if group_id is None else group_id,
            },
        }

    def _load_cached_gt(self, d: dict[str, Any], ctx: bytes, for_label: str):
        repr_prefix, subj_id, repr_type = d["meta"]["repr_hash"].decode().split(":")
        gt_key = f"gt:{subj_id}:{for_label}:{repr_type}"
        gt_repr = Repr.from_bytes(self._redis_client.hget(gt_key, ctx))
        return gt_repr


class KVReprCohort(ReprCohort):
    class Config(ReprCohort.Config):
        # Information required for discovering training and validation samples
        train_dataset: str = Field(description="Redis set with all subject ids for this cohort")
        validation_dataset: str | None = Field(
            default=None, description="Redis set with all validation subject ids for this cohort"
        )
        compress_type: CompressType = CompressType.rgbimage
        length: int | tuple[int, int] = Field(
            default=(1000, 1),
            description="For tiny datasets, the length can be duplicated to avoid recreating multiprocessing workers",
        )
        labeling_config: LabelingConfig

    def __init__(self, cfg: KVReprCohort.Config, for_model=None, kv=None):
        self.kv = kv if kv is not None else mtl_settings.kv
        self.cfg = cfg
        self.use_cached_gt = self.cfg.compress_type is CompressType.rgbimage
        self.adapter_cfg = ADAPTERS[
            LabelType.from_string(self.get_labeling_config().determine_type_of_label(self.get_label()))
        ].Config()
        super().__init__(cfg, None, for_model)
        self.cfg: KVReprCohort.Config

    def get_labeling_config(self):
        return self.cfg.labeling_config

    def get_batch_transform(self, train):
        if self.cfg.compress_type is CompressType.rgbimage:
            return super().get_batch_transform(train)
        else:
            return None

    def collate_fn(self):
        if self.cfg.compress_type is CompressType.rgbimage:
            return super().collate_fn()
        else:
            return mtl_batch_collate

    def get_selection_strategy(self) -> ReprSelector | None:
        return self.get_extra().repr_selector

    def get_src_transform(self, src_ds: KVDataset, train: bool):
        def mmm_instance_from_repr(
            mtl_adapter, label, subject_key, subject: MSubject, anno: Annotation, h, ctx, repr: Repr, sample_suffix: str
        ):
            # For subjects with many representations, this function is called many times.
            # In consequence, only light weight operations should be done here!
            d = src_ds.representation_to_mmm(h, ctx, repr, group_id=subject_key if not self.use_cached_gt else None)
            d["meta"]["group_id"] += sample_suffix  # ensure group_id is unique for each sample
            if self.use_cached_gt:
                gt_repr = src_ds._load_cached_gt(d, ctx, self.cfg.for_data.for_label)
            else:
                gt_repr = anno.extract_gt(
                    label_name=self.cfg.for_data.for_label,
                    subject=subject,
                    labeling=self.get_labeling_config(),
                    for_instance=repr,
                )

            return mtl_adapter.extract_label(self.adapter_cfg, d, gt_repr, label, train=train, extra=self.get_extra())

        def load_subject(subject_data):
            repr_hash, contexts, redis_ds = subject_data
            if self.get_selection_strategy() is not None:
                contexts = self.get_selection_strategy().apply(contexts)

            labeling_config = self.get_labeling_config()
            label_type = labeling_config.determine_type_of_label(
                (label := labeling_config.get_parsed()[self.cfg.for_data.for_label])
            )
            mtl_adapter = ADAPTERS[LabelType.from_string(label_type)]
            if not self.use_cached_gt:
                # Extract GT from the subject. This could be cached as well if optimization is needed.
                subject_key = repr_hash.decode().split(":")[1]
                subject = MSubject.model_validate_json(
                    self.kv.get(f"{mtl_settings.subj_prefix}:{subject_key}").decode()  # type: ignore
                )
                anno = subject.get_last_updated_annotation()
            else:
                subject_key, subject, anno = None, None, None

            id_suffix = f"_{uuid.uuid4().hex[:8]}"

            return [
                mmm_instance_from_repr(mtl_adapter, label, subject_key, subject, anno, repr_hash, ctx, repr, id_suffix)
                for ctx, repr in zip(contexts, redis_ds.retrieve_values_for_contexts(repr_hash, contexts))
            ]

        return load_subject

    def build_kv_dataset(
        self, train: bool, dataset_key: str | None = None, subject_keys: None | list[bytes] = None
    ) -> KVDataset:
        if dataset_key is not None:
            assert self.kv.exists(dataset_key), f"Dataset {dataset_key} not found"
            assert subject_keys is None, f"Cannot specify subject keys when dataset key is given in {self}"
        else:
            assert dataset_key is None, f"Cannot specify both dataset key and subject keys in {self}"
        all_subject_keys: list[bytes] = self.kv.smembers(dataset_key) if subject_keys is None else subject_keys
        assert len(all_subject_keys) > 0, f"Dataset {dataset_key} has no subjects"
        if self.use_cached_gt:
            subjects_with_gt = [
                s_key
                for s in all_subject_keys
                if self.kv.exists(
                    f"gt:{(s_key := s.decode())}:{self.cfg.for_data.for_label}:{self.cfg.compress_type.value}"
                )
            ]
        else:
            logfire.info(
                "Ground truth is not cached. Need to check {num_subjects} subjects for those with GT.",
                num_subjects=len(all_subject_keys),
            )
            full_subjects = [
                MSubject.model_validate_json(
                    self.kv.get(f"{mtl_settings.subj_prefix}:{s.decode()}").decode()  # type: ignore
                )
                for s in all_subject_keys
            ]
            relevant_annotations = [
                (s, anno)
                for s in full_subjects
                if (anno := s.get_last_updated_annotation()) is not None
                # and self.cfg.pretraining.for_label in anno.result
                and self.cfg.for_data.for_label in [r.from_name for r in anno.result]
            ]

            subjects_with_gt = [s.id for s, _ in relevant_annotations]

        repr_keys = [f"repr:{subject_key}:{self.cfg.compress_type.value}".encode() for subject_key in subjects_with_gt]
        assert len(repr_keys) > 0, f"Dataset {dataset_key} has no ground truth for {self.cfg.for_data.for_label}"
        return KVDataset(
            adapter_cfg=self.adapter_cfg,
            subject_keys=repr_keys,
            kv=self.kv,
            deserializer=Repr.from_bytes,
            length=self.cfg.length if isinstance(self.cfg.length, int) else self.cfg.length[int(not train)],
        )

    def build_source_data(self):
        if self.cfg.validation_dataset is None:
            from mmm.data_loading.utils import train_val_split

            # Get a sorted list to ensure reproducibility as long as no new subjects are added during training
            all_subject_keys = sorted(self.kv.smembers(self.cfg.train_dataset))  # set[bytes]. ids, not subject keys

            train_indices, val_indices = train_val_split(list(range(len(all_subject_keys))), perc=0.7, seed=42)
            train_ds = self.build_kv_dataset(
                subject_keys=(train_keys := [all_subject_keys[i] for i in train_indices]), train=True
            )
            val_ds = self.build_kv_dataset(
                subject_keys=(val_keys := [all_subject_keys[i] for i in val_indices]), train=False
            )
            logfire.info(
                "No validation dataset specified, using {num_train}, {num_val} split (70/30).",
                num_train=len(train_indices),
                num_val=len(val_indices),
                train_keys=train_keys,
                val_keys=val_keys,
            )
        else:
            train_ds = self.build_kv_dataset(dataset_key=self.cfg.train_dataset, train=True)
            val_ds = self.build_kv_dataset(dataset_key=self.cfg.validation_dataset, train=False)
        return train_ds, val_ds


if __name__ == "__main__":
    from mmm.api.evaluation import TaskDefinition
    from mmm.logging.st_ext import st, stw

    def visualize_dataset_kv(task_definition: TaskDefinition):
        all_sets = list(
            filter(
                lambda s: not s.startswith("_"),
                map(lambda b: b.decode(), mtl_settings.kv.scan_iter(_type="set", count=500)),
            )
        )
        train_set = st.selectbox(label="Train dataset", options=all_sets, index=0)
        val_set = st.selectbox(label="Validation dataset", options=all_sets)
        selected_label = st.selectbox(
            label="Selected label",
            options=sorted(task_definition.get_labeling_config().get_parsed().keys()),
            index=0,
        )
        compress_type: str = st.selectbox(label="Compress type", options=CompressType, index=0)
        cohort_cfg = KVReprCohort.Config(
            batch_size=2,
            num_workers=0,
            compress_type=CompressType(compress_type),
            labeling_config=task_definition.get_labeling_config(),
            for_data=ReprDataset.Config(
                for_input=task_definition.get_labeling_config().get_parsed()[selected_label]["to_name"][0],
                for_label=selected_label,
            ),
            train_dataset=train_set,
            validation_dataset=val_set,
        )
        stw(KVReprCohort(cohort_cfg, kv=mtl_settings.kv))

    @st.cache_resource
    def build_cohort(selected_label: str):
        task_definition = st.session_state["task_definition"]
        cohort_cfg = ReprCohort.Config(
            batch_size=2,
            num_workers=0,
            labeling_config=task_definition.get_labeling_config(),
            for_data=ReprDataset.Config(
                for_input=task_definition.get_labeling_config().get_parsed()[selected_label]["to_name"][0],
                for_label=selected_label,
            ),
        )
        return ReprCohort(cohort_cfg, task_definition.get_subjects())

    def visualize_dataset(task_definition: TaskDefinition):
        selected_label = st.selectbox(
            label="Selected label",
            options=sorted(task_definition.get_labeling_config().get_parsed().keys()),
            index=0,
        )

        stw(build_cohort(selected_label))

    with st.sidebar:
        pages = {"KV Cache Dataset": visualize_dataset_kv, "Local Dataset": visualize_dataset}
        page_name = st.selectbox("Select page", options=pages.keys(), index=0)
        if "task_definition" in st.session_state:
            st.json(st.session_state["task_definition"].model_dump(), expanded=False)

    if "task_definition" not in st.session_state:
        with st.form(key="task_definition_form"):
            st.write("Select task definition either by uploading a JSON file or by pasting it into the text area.")
            search_parent = st.text_input("Search parent", value="/mmm/data_interface_tasks/leopard_eval/")
            task_definition_path = st.selectbox(
                label="Task definition", options=Path(search_parent).rglob("task-definition.json")
            )
            task_definition_str = task_definition_path.read_text()
            if st.form_submit_button("Submit"):
                task_definition = TaskDefinition(**json.loads(task_definition_str))
                task_definition.make_relative_paths_absolute(
                    DistributedPath.from_string(task_definition_path.parent.absolute().__str__())
                )
                st.session_state["task_definition"] = task_definition
                st.rerun()

    else:  # "task_definition" in st.session_state:
        pages[page_name](st.session_state["task_definition"])
