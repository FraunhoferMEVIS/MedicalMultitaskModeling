from types import SimpleNamespace
import pytest
from mmm.api.functions.deeplearning import Predict
from mmm.api.models import Annotation, MSubject, BaseResult, Volume3DImage, Volume3DMask
from mmm.api.mtl_adapter import LabelingConfig
from mmm.api.WorkerState import WorkerState
from m3_sdk.DistributedPath import DistributedPath
from mmm.settings import mtl_settings
from tests.test_data import CLF_VOLUME_IDS, CLF_VOLUMES, IMAGES, VOLUME_IDS, VOLUMES


@pytest.fixture(
    ids=["2DClf", "2DSeg"],
    params=[
        (
            Predict.Args(
                subjects=[
                    MSubject(
                        id="pytest_baseballimage",
                        data={"image": IMAGES[0]},
                    )
                ],
                label_config=LabelingConfig(
                    xml='<View><Image name="image"/><Choices name="clf" toName="image" /></View>'
                ),
                for_labels={"clf": "imgnetclf"},
            ),  # imagenet 1k has to be used for the class name to match the expected result
            lambda r: r.predictions[0].result[0].get_multiclass_classification() == "baseball",
        ),
        (
            Predict.Args(
                subjects=[
                    MSubject(
                        id="pytest_baseballimage_seg",
                        data={"image": IMAGES[0]},
                    )
                ],
                label_config=LabelingConfig(
                    xml='<View><Image name="image"/><BrushLabels name="seg" toName="image" /></View>'
                ),
                for_labels={"seg": "cocoseg"},
            ),  # imagenet 1k has to be used for the class name to match the expected result
            lambda r: "sports ball" in [r["value"]["brushlabels"][0] for r in r.predictions[0].result],
        ),
    ],
)
def subject_task_expected(request):
    return request.param


def test_prediction(subject_task_expected):
    args, is_correct = subject_task_expected
    res = Predict.invoke(args, WorkerState(), mtl_settings.kv)
    assert is_correct(res), f"Prediction assumption failed for {res} and {args}"


@pytest.fixture(ids=VOLUME_IDS, params=VOLUMES)
def volume_segmentation(request):
    img_path, mask_path, num_classes, task_name, volume_inference = request.param

    return SimpleNamespace(
        subject=MSubject(
            id=f"pytest_volume_seg_{task_name}",
            data={"tomoimg": Volume3DImage(url=DistributedPath(uri=img_path))},
            annotations=[
                Annotation(
                    result=[
                        Volume3DMask(value=DistributedPath(uri=mask_path), to_name="tomoimg", from_name="mask")  # type: ignore
                    ]
                )
            ],
        ),
        labeling=LabelingConfig(
            xml='<View><Image name="tomoimg" /><Volume3DMask name="mask" toName="tomoimg" /></View>'
        ),
        num_classes=num_classes,
        task_name=task_name,
        volume_inference=volume_inference,
    )


def test_volume_segmentation(volume_segmentation):
    args = Predict.Args(
        subjects=[volume_segmentation.subject],
        for_labels={"mask": volume_segmentation.task_name},
        label_config=volume_segmentation.labeling,
    )
    res = Predict.invoke(args, WorkerState(), mtl_settings.kv)

    assert res.predictions[0].result[0].value.exists()  # type: ignore


@pytest.fixture(ids=CLF_VOLUME_IDS, params=CLF_VOLUMES)
def volume_classification(request):
    (img_path, class_name) = request.param
    return SimpleNamespace(
        subject=MSubject(
            id="volumeclf1",
            data={"tomoimg": Volume3DImage(url=DistributedPath(uri=img_path))},
            annotations=[
                Annotation(
                    result=[
                        BaseResult(type="choices", value={"choices": ["malignant"]}, to_name="tomoimg", from_name="clf")
                    ]
                )
            ],
        ),
        labeling=LabelingConfig(xml='<View><Image name="tomoimg" /><Choices name="clf" toName="tomoimg" /></View>'),
        task_name="luna25_malignancy",
    )


def test_volume_classification(volume_classification):
    args = Predict.Args(
        subjects=[volume_classification.subject],
        for_labels={"clf": volume_classification.task_name},
        label_config=volume_classification.labeling,
    )
    res: Predict.Results = Predict.invoke(args, WorkerState(), mtl_settings.kv)

    assert (
        res.predictions[0].result[0].get_multiclass_classification()
        == volume_classification.subject.annotations[0].result[0].get_multiclass_classification()
    )
    volume_classification.subject.predictions = [res.predictions[0]]

    from mmm.api.mtl_adapter import ClassificationAdapter as ClfAdapter
    from mmm.api.mtl_adapter import ClassificationAdapterConfig as ClfAdapterCfg

    assert (
        ClfAdapter.compute_metrics(
            ClfAdapterCfg(), [volume_classification.subject], {"labels": ["benign", "malignant"]}
        )["acc"]
        == 1.0
    )
