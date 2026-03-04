import json
from mmm.api.models import MSubject
from mmm.api.mtl_adapter import LabelingConfig
from tests.test_data import LABELED_SUBJECT_DIR


def test_extract_repr_label_for_image():
    labeling = LabelingConfig(xml=LABELED_SUBJECT_DIR.joinpath("labeling.xml").read_text())
    assert "classification" in labeling.get_parsed()
    assert "segmentation" in labeling.get_parsed()
    subject = MSubject(**json.loads(LABELED_SUBJECT_DIR.joinpath("subject.json").read_text()))

    clf_gt = subject.get_last_updated_annotation().extract_gt("classification", subject, labeling, None)
    assert clf_gt.meta["class_name"] == "Teddy"

    seg_gt = subject.get_last_updated_annotation().extract_gt("segmentation", subject, labeling, None)
    assert seg_gt.tensor.unique().numel() == len(labeling["segmentation"]["labels"]) + 1  # class_names + unlabeled

    instances = list(MSubject.extract_instances(subject, labeling))
    assert len(instances) == 1

    clf_gt = subject.get_last_updated_annotation().extract_gt("classification", subject, labeling, instances[0])
    seg_gt = subject.get_last_updated_annotation().extract_gt("segmentation", subject, labeling, instances[0])
