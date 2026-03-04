from copy import deepcopy
from mmm.mmm_types.LabelType import LabelType
import pytest
from mmm.api.mtl_adapter import ADAPTERS, process_for_metrics, LabelingConfig
from mmm.api.models import MSubject, Annotation, Prediction, Result, BaseResult


def clf_result(classname: str, labelname: str, **kwargs) -> BaseResult:
    return BaseResult(
        value={"choices": [classname]},
        from_name=labelname,
        to_name="image",
        type="choices",
        **kwargs,
    )


@pytest.fixture(
    ids=["clf_with_irrelevant", "clf_multiple"],  # , "seg", "surv", "vol_seg"],
    params=[
        (
            [
                MSubject(
                    id="S0",
                    annotations=[
                        Annotation(result=[clf_result("C1", "clf"), clf_result("C1", "clf_nopred")]),
                    ],
                    predictions=[
                        # This prediction should be ignored, because there is a newer prediction for "testmodel"
                        Prediction(score=1.0, result=[clf_result("C2", "clf", score=0.6)], model_version="testmodel"),
                        Prediction(
                            score=1.0,
                            result=[clf_result("C1", "clf", score=0.7), clf_result("C1", "clf_noanno", score=0.7)],
                            model_version="testmodel",
                        ),
                        Prediction(score=1.0, result=[clf_result("C1", "clf", score=0.7)], model_version="othermodel"),
                    ],
                )
            ],
            {
                "clf": {
                    "acc": 1.0,
                }
            },  # auc not defined for one sample! "auc": 1.0}},
            """
<View>
  <Image name="image" value="$image"/>
  <Choices name="clf" toName="image">
    <Choice value="C1"/>
    <Choice value="C2" />
  </Choices>
</View>
""",
        ),
        (
            [
                MSubject(
                    id="S0",
                    annotations=[
                        Annotation(
                            result=[
                                clf_result("C2", "clf1"),
                            ]
                        ),
                    ],
                    predictions=[
                        Prediction(
                            score=1.0,
                            result=[
                                clf_result("C2", "clf1", score=0.7, all_class_scores=[0.3, 0.7]),
                            ],
                            model_version="testmodel",
                        ),
                    ],
                ),
                MSubject(
                    id="S0",
                    annotations=[
                        Annotation(
                            result=[
                                clf_result("C1", "clf2"),
                            ]
                        ),
                    ],
                    predictions=[
                        Prediction(
                            score=1.0,
                            result=[
                                clf_result("C2", "clf2", score=0.7),
                            ],
                            model_version="testmodel",
                        ),
                    ],
                ),
                MSubject(
                    id="S0",
                    annotations=[
                        Annotation(result=[clf_result("C1", "clf1"), clf_result("C1", "clf2")]),
                    ],
                    predictions=[
                        Prediction(
                            score=1.0,
                            result=[
                                clf_result("C1", "clf1", score=0.6, all_class_scores=[0.6, 0.4]),
                                clf_result("C1", "clf2", score=0.7),
                            ],
                            model_version="testmodel",
                        ),
                    ],
                ),
            ],
            {"clf1": {"acc": 1.0, "auc": 1.0}, "clf2": {"acc": 0.5}},
            """
<View>
  <Image name="image" value="$image"/>
  <Choices name="clf1" toName="image">
    <Choice value="C1"/>
    <Choice value="C2" />
  </Choices>
  <Choices name="clf2" toName="image">
    <Choice value="C1"/>
    <Choice value="C2" />
  </Choices>
</View>
""",
        ),
    ],
)
def test_subjects_outcomes(request):
    return request.param


def test_split_preds_by_label(test_subjects_outcomes):
    subjects, expected, labeling_xml = test_subjects_outcomes
    original_subjects = deepcopy(subjects)
    results = process_for_metrics(subjects, "testmodel", LabelingConfig(xml=labeling_xml))
    assert set(results.keys()) == set(expected.keys())

    for label, label_subjects in results.items():
        for subject in label_subjects:
            # all annotations and predictions should contain only one annotation and one prediction
            assert len(subject.annotations) == 1
            assert len(subject.predictions) == 1
            # all annotations and predictions should contain the same label
            assert False not in [r["from_name"] == label for r in subject.annotations[0].result]
            assert False not in [r["from_name"] == label for r in subject.predictions[0].result]

    assert subjects[0].model_dump_json() == original_subjects[0].model_dump_json()


def test_compute_metric_adapter(test_subjects_outcomes):
    subjects, expected, labeling_xml = test_subjects_outcomes
    for_model = "testmodel"
    results = process_for_metrics(subjects, for_model, labeling := LabelingConfig(xml=labeling_xml))

    for label, label_subjects in results.items():
        # There may be more metrics reported, but check that at least the expected ones are present
        metrics = (adapter := ADAPTERS[LabelType.from_string(labeling.get_parsed()[label]["type"])]).compute_metrics(
            adapter.Config(), label_subjects, labeling.get_parsed()[label]
        )
        for metric_name, metric_value in expected[label].items():
            assert metric_name in metrics, f"Metric {metric_name} not found in {metrics}"
            assert metrics[metric_name] == metric_value, f"{metric_name=}, {metrics=}, {metric_value=}"
