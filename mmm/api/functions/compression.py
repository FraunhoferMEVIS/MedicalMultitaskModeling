import logfire
import redis
from m3_sdk.types import CompressType
from pydantic import Field

from mmm.api.functions.ApiFunction import ApiFunction
from mmm.api.models import MSubject
from mmm.api.WorkerState import WorkerState
from mmm.mmm_types.LabelType import LabelType
from mmm.settings import mtl_settings


class CacheSubjects(ApiFunction):
    class Args(ApiFunction.Args):
        subjects: list[MSubject] = Field(
            ...,
            examples=[
                [MSubject(id="Manfred", data={"image": "/local/path/to/image.jpg"})],
                [MSubject(id="Manfred", data={"image": "http://address:8080/image.jpg"})],
            ],
        )
        overwrite: bool = Field(default=True, description="If True, overwrites the existing subjects in the database")

    class Results(ApiFunction.Results):
        subject_keys: list[bytes]

    @staticmethod
    def invoke(args: Args, ws: WorkerState, kv: redis.Redis) -> Results:
        res = []

        all_keys, key_to_subj = [], {}
        for subject in args.subjects:
            assert subject.id, f"Subject {subject} does not have an ID"
            all_keys.append(k := f"{mtl_settings.subj_prefix}:{subject.id}")
            key_to_subj[k] = subject

        if args.overwrite:
            existing_keys = []
            new_keys = all_keys
        else:
            existing_keys = [subj_key for subj_key in all_keys if kv.exists(subj_key)]
            new_keys = set(all_keys) - set(existing_keys)

        with logfire.span("Adding subjects", cached_keys=new_keys, preexisting_keys=existing_keys) as span:
            with kv.pipeline() as pipe:
                for new_key in new_keys:
                    pipe.set(
                        new_key,
                        key_to_subj[new_key].model_dump_json(exclude_none=True),
                    )
                    res.append(new_key)
                pipe.execute()
            if new_keys:
                logfire.info("Cached {num_subjects} new subjects", num_subjects=len(new_keys))
        return CacheSubjects.Results(subject_keys=all_keys)


class CacheInstances(ApiFunction):
    """
    Caches the instances of the subjects and compresses them according to the compress type.
    Once all instances of a subject are processed,
    the subject is added to the set of compressed subjects for that type of compression.

    Representations are stored as fields in the Redis hash of the subject.

    If the subject naturally results in a single image instance, the context is "...".

    If dense annotations are available, they should be cached as well using `with_labels`.
    For sparse annotations (e.g. classification), it makes more sense to compute them on the fly."
    """

    class Args(ApiFunction.Args):
        for_type: CompressType = Field(..., description="The single type of compression to use")
        subject_keys: list[bytes] = Field(
            ...,
            description="The keys of the subjects to cache, returned from `cache_subjects`",
            examples=[["subject:Manfred", "subject:Lisa"]],
        )
        with_labels: list[str] | None = Field(
            default=None,
            description="The labels to cache. If None, only the instance is cached. If empty, no labels are cached.",
            examples=[["foreground_mask", "tumor_mask"]],
        )
        batch_size: int = 1
        num_workers: int = 0
        wandb_init_settings: dict | None = Field(
            default=None,
            description="If provided, initializes a W&B run for this compression",
            examples=[{"project": "medical-multitask-modeling", "name": "compression-run-001"}],
        )
        only_gt_for_pixeltoken: bool = Field(
            default=True, description="pixeltokens store the features and the ground truth in the same tensor"
        )
        skip_if_exists: bool = Field(
            default=False,
            description="If True, skips the subject if it is already part of the set for that type",
        )

    class Results(ApiFunction.Results):
        num_fields_added: int
        num_instances_processed: int

    @staticmethod
    def invoke(args: Args, ws: WorkerState, kv: redis.Redis) -> Results:
        if not args.subject_keys:
            return CacheInstances.Results(num_fields_added=0, num_instances_processed=0)

        import torch
        import torchvision.transforms.functional as F
        import wandb

        from mmm.api.data import SubjDataset
        from mmm.api.models import MSubject, Repr

        # Load all subjects using a pipeline
        with kv.pipeline() as pipe:
            for subj in args.subject_keys:
                pipe.get(subj)
            subjects: list[MSubject] = [MSubject.model_validate_json(subj) for subj in pipe.execute()]

        if args.skip_if_exists:
            compressed_subjects = kv.smembers(f"compressed:{args.for_type.value}")
            uncompressed_subjects = [s for s in subjects if f"{s.id}".encode() not in compressed_subjects]

            logfire.info(
                "Skipping {num_skipped} subjects (already compressed)",
                num_skipped=len(subjects) - len(uncompressed_subjects),
                compressed_subjects=compressed_subjects,
                uncompressed_subjects=uncompressed_subjects,
            )
            if not uncompressed_subjects:
                return CacheInstances.Results(num_fields_added=0, num_instances_processed=0)
            else:
                subjects = uncompressed_subjects

        if args.wandb_init_settings is None:
            wandb.init(mode="disabled")
        else:
            wandb.init(**args.wandb_init_settings)

        def progress_log_callback(batch_idx: int, log_dict: dict):
            wandb.log(log_dict)

        def subject_hook(subject: MSubject):
            kv.sadd(f"compressed:{args.for_type.value}", subject.id)

        # Sometimes single subjects result in many batches (gigapixel imaging)
        # Sometimes many subjects result in a single batch (e.g. for CXR)
        # The dataset enables to load full batches for both cases
        subject_dataset = SubjDataset(SubjDataset.Config(), data=subjects)
        with logfire.span("Starting to compress {num_subj} subjects", num_subj=len(subjects)) as span:
            num_fields_added, num_instances_processed = 0, 0
            for _, batch in subject_dataset.tokenize(
                ws.fm,
                args.for_type,
                batchsize=args.batch_size,
                with_labels=args.with_labels,
                num_workers=args.num_workers,
                # Extract the labels in worker instead of the main process by putting the work into the dataset
                progress_log_callback=progress_log_callback,
                subject_hook=subject_hook,
            ):
                with kv.pipeline() as pipe:
                    for repr in batch:
                        repr: Repr
                        repr.tensor = repr.tensor.cpu()
                        if "labels" in repr.meta:
                            labels_repr: dict[str, Repr] = repr.meta.pop("labels")
                            for label_name, label_repr in labels_repr.items():
                                if args.for_type is CompressType.rgbimage and label_repr.meta["label_type"] in [
                                    t.name for t in LabelType.get_segmentation_types()
                                ]:
                                    # Resize the label to fit the image
                                    if label_repr.tensor.shape != repr.tensor.shape[-2:]:
                                        label_repr.tensor = F.resize(
                                            label_repr.tensor,
                                            repr.tensor.shape[-2:],
                                            interpolation=F.InterpolationMode.NEAREST,
                                        )
                                    label_repr.tensor = label_repr.tensor.long()
                                elif (
                                    args.for_type is CompressType.pixeltoken
                                    and label_repr.meta["label_type"] == LabelType.seg.name
                                ):
                                    # Sieve pixels to reduce the number of redundant pixels used for training
                                    from mmm.api.dense_compression import compute_pixel_features_sklearn_by_purpose

                                    pixel_features, pixel_groundtruths = compute_pixel_features_sklearn_by_purpose(
                                        repr.tensor.unsqueeze(0), label_repr.tensor.unsqueeze(0), for_inference=False
                                    )
                                    # Concatenate the pixel features and the pixel groundtruths for sklearn style X, y
                                    # Results in a tensor of shape (N, C+1)
                                    label_repr.tensor = torch.cat(
                                        [pixel_features, pixel_groundtruths.unsqueeze(1)], dim=1
                                    )
                                pipe.hset(
                                    name=f"gt:{repr.meta['msubject'].id}:{label_name}:{args.for_type.value}",
                                    key=f'{repr.meta["context"]}',
                                    value=label_repr.to_bytes(args.for_type, is_ground_truth=True),
                                )
                        if not (args.only_gt_for_pixeltoken and args.for_type is CompressType.pixeltoken):
                            pipe.hset(
                                name=f"repr:{repr.meta['msubject'].id}:{args.for_type.value}",
                                key=f'{repr.meta["context"]}',
                                value=repr.to_bytes(args.for_type, is_ground_truth=False),
                            )
                    num_fields_added += sum(pipe.execute())
                    num_instances_processed += len(batch)
        wandb.finish()
        return CacheInstances.Results(
            num_fields_added=num_fields_added, num_instances_processed=num_instances_processed
        )
