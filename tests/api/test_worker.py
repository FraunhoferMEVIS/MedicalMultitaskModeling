from mmm.api.Finetuner import FineTuner
from mmm.api.data import ReprDataset, KVReprCohort
from mmm.task_sampling import CyclicTaskSampler
from mmm.trainer.Loop import TrainLoopConfig, ValLoopConfig
import pytest
from typing import Generator
import torch
import torchvision.transforms.functional as F
import imageio.v3 as iio

from mmm.settings import mtl_settings
from mmm.api.models import MSubject, Repr
from mmm.api.mtl_adapter import LabelingConfig
from m3_sdk.types import CompressType
from mmm.api.api_worker import cache_instances, cache_subjects, finetune, CacheInstances, CacheSubjects, Finetune
from mmm.api.PetTestData import PetTestData, LabelType

from .test_api import kv_initializer
from ..test_data import image_url


@pytest.fixture
def pet_testdata() -> PetTestData:
    """
    The dataset needs to exist in a directory accessible for the testing environment.
    It should return a tuple of an image and a tuple of the class index and the segmentation mask.
    """
    return PetTestData(
        cfg=PetTestData.Config(
            for_label=[LabelType.seg, LabelType.clf],
            num_cases=10,
            num_classes=2,
        ),
        data_directory="/mmm",
    )


@pytest.fixture(params=CompressType)
def compress_type(request) -> Generator[CompressType, None, None]:
    yield request.param


def test_instance_compression(image_url: str, kv_initializer, compress_type: CompressType):
    """
    Tests compression without annotations
    """
    if compress_type is CompressType.ctxtoken:
        pytest.skip("Context token compression is not yet implemented")

    subj = MSubject(id=f"test_instance_compression_subject_{image_url.split('/')[-1]}", data={"image": image_url})
    cache_subj_result = cache_subjects(
        CacheSubjects.Args(subjects=[subj]), kv=(kv := kv_initializer("instance_compression"))
    )
    redis_subject_keys = cache_subj_result.subject_keys
    assert redis_subject_keys[0].decode().startswith(f"{mtl_settings.subj_prefix}:")
    cache_result = cache_instances(
        CacheInstances.Args(
            for_type=compress_type, subject_keys=redis_subject_keys, only_gt_for_pixeltoken=False, num_workers=0
        ),
        kv=kv,
    )
    num_fields_added, num_instances_processed = cache_result.num_fields_added, cache_result.num_instances_processed
    assert num_fields_added == 1 == num_instances_processed

    # Load the compressed instance from cache and check for similarity
    repr_hash_name = f'repr:{redis_subject_keys[0].decode().split(":")[1]}:{compress_type.value}'
    repr_bytes = kv.hget(repr_hash_name, "...")
    compressed_instance = Repr.from_bytes(repr_bytes)

    if compress_type is CompressType.rgbimage:
        original_image = F.resize(F.to_tensor(iio.imread(image_url)), size=compressed_instance.tensor.shape[-2:])
        # Ensure that most pixels are close
        assert torch.isclose(original_image, compressed_instance.tensor, atol=0.1).sum() > 0.99 * original_image.numel()


def test_annotated_instance_compression(pet_testdata: PetTestData, kv_initializer, compress_type: CompressType):
    if compress_type is CompressType.ctxtoken:
        pytest.skip("Context token compression is not yet implemented")

    subj = pet_testdata.create_subject_from_case(
        pet_testdata[0], subj_id=f"test_annotated_instance_compression_{compress_type.value}"
    )
    kv = kv_initializer("annotated_instance_compression")
    subj_keys = cache_subjects(CacheSubjects.Args(subjects=[subj]), kv=kv).subject_keys

    args = CacheInstances.Args(
        for_type=compress_type,
        subject_keys=subj_keys,
        with_labels=["segmentation_testlabel"],
        only_gt_for_pixeltoken=False,
        batch_size=8,
        num_workers=0,
    )
    cache_instances(args, kv=kv)

    args.skip_if_exists = True
    r = cache_instances(args, kv=kv)
    num_fields_added, num_instances_processed = r.num_fields_added, r.num_instances_processed
    # Ensure that in the second run, no fields were added AND no instances were processed
    assert num_fields_added == 0 and num_instances_processed == 0

    args.skip_if_exists = False
    r = cache_instances(args, kv=kv)
    num_fields_added, num_instances_processed = r.num_fields_added, r.num_instances_processed
    # Assert that no new fields were added, but instances were processed indicating that the fields were replaced
    assert num_fields_added == 0 and num_instances_processed > 0


def test_finetune(pet_testdata: PetTestData, kv_initializer):
    compress_type: CompressType = CompressType.rgbimage  # put at top to extend with fixture later
    if compress_type is CompressType.ctxtoken:
        pytest.skip("Context token compression is not yet implemented")

    # Compress the dataset
    labeling = LabelingConfig(xml=pet_testdata._build_labeling_config_xml())
    # all_labels = list(labeling.get_parsed().keys())
    kv = kv_initializer("test_finetune")
    subjs = [
        pet_testdata.create_subject_from_case(pet_testdata[i], subj_id=f"test_dataset_{i}_{compress_type.value}")
        for i in range(len(pet_testdata))
    ]
    subj_keys = cache_subjects(CacheSubjects.Args(subjects=subjs), kv=kv).subject_keys

    cache_instances(
        CacheInstances.Args(
            for_type=compress_type,
            subject_keys=subj_keys,
            with_labels=list(labeling.get_parsed().keys()),
            only_gt_for_pixeltoken=False,
            batch_size=8,
            num_workers=0,
        ),
        kv=kv,
    )

    # Datasets are collections of subject-ids (not keys), create a subset with 5 subjects for validation
    kv.sadd(ds_name := f"testdataset_{compress_type.value}", *[subj.id for subj in subjs[:5]])

    cohort_configs = [
        KVReprCohort.Config(
            batch_size=(2, 2),
            num_workers=0,
            compress_type=compress_type,
            labeling_config=labeling,
            train_dataset=ds_name,
            validation_dataset=ds_name,
            for_data=ReprDataset.Config(for_input=cfg["to_name"][0], for_label=label),
        )
        for label, cfg in labeling.get_parsed().items()
    ]

    finetune_settings = FineTuner.Config(
        mtl_train_loop=TrainLoopConfig(
            max_steps=10,
            task_sampler=CyclicTaskSampler.Config(mode="break_with_longest_loader"),
        ),
        mtl_val_loop=ValLoopConfig(max_steps=10),
        early_stopping=None,
    )
    finetune(
        Finetune.Args(cohorts=cohort_configs, finetuning_id="test_finetune", cfg=finetune_settings, num_loops=1),
        kv=kv,
    )
