"""
The dataset is designed for development. It does not enable for meaningful metrics.
"""

import random
from pathlib import Path
from typing import Literal

import numpy as np
import torchvision.datasets as datasets
from m3_sdk.utils import rgbnumpy_to_base64
from PIL.Image import Image

from mmm.api.models import MSubject
from mmm.api.mtl_adapter import LabelingConfig
from mmm.api.utils import binary_mask_to_result
from mmm.BaseModel import BaseModel
from mmm.data_loading.utils import TransformedSubset
from mmm.mmm_types.LabelType import LabelType

CLASS_TO_SPECIES = {
    "Abyssinian": "cat",
    "American Bulldog": "dog",
    "American Pit Bull Terrier": "dog",
    "Basset Hound": "dog",
    "Beagle": "dog",
    "Bengal": "cat",
    "Birman": "cat",
    "Bombay": "cat",
    "Boxer": "dog",
    "British Shorthair": "cat",
    "Chihuahua": "dog",
    "Egyptian Mau": "cat",
    "English Cocker Spaniel": "dog",
    "English Setter": "dog",
    "German Shorthaired": "dog",
    "Great Pyrenees": "dog",
    "Havanese": "dog",
    "Japanese Chin": "dog",
    "Keeshond": "dog",
    "Leonberger": "dog",
    "Maine Coon": "cat",
    "Miniature Pinscher": "dog",
    "Newfoundland": "dog",
    "Persian": "cat",
    "Pomeranian": "dog",
    "Pug": "dog",
    "Ragdoll": "cat",
    "Russian Blue": "cat",
    "Saint Bernard": "dog",
    "Samoyed": "dog",
    "Scottish Terrier": "dog",
    "Shiba Inu": "dog",
    "Siamese": "cat",
    "Sphynx": "cat",
    "Staffordshire Bull Terrier": "dog",
    "Wheaten Terrier": "dog",
    "Yorkshire Terrier": "dog",
}


class PetTestData:
    """
    Creating a test dataset with classification and segmentation labels.
    """

    class Config(BaseModel):
        for_label: list[LabelType] = [LabelType.clf, LabelType.seg, LabelType.surv]
        max_bag_size: int = 3

        num_classes: int = 3
        num_cases: int = 20

        image_encoding: Literal["base64", "abs_filepath"] = "base64"

    @staticmethod
    def data_exists(data_directory: str = "/mmm") -> bool:
        return (Path(data_directory) / "oxford-iiit-pet").exists()

    def __init__(
        self,
        cfg: Config,
        data_directory: str = "/mmm",
        download_pet_data_from_torchvision: bool = True,
    ) -> None:
        self.cfg = cfg
        ds_kwargs = dict(
            root=data_directory,
            target_types=["category", "segmentation"],
            download=download_pet_data_from_torchvision,
        )
        trainval = datasets.OxfordIIITPet(split="trainval", **ds_kwargs)  # type: ignore
        test = datasets.OxfordIIITPet(split="test", **ds_kwargs)  # type: ignore

        self._class_indices = random.sample(list(set(trainval._labels)), self.cfg.num_classes)
        assert len(self._class_indices) == len(set(self._class_indices)), "Duplicate class indices"
        self.classes = [trainval.classes[i] for i in self._class_indices]
        assert len(self.classes) == len(set(self.classes)), "Duplicate class names"
        self.trainval = TransformedSubset(
            trainval, indices=[i for i, c in enumerate(trainval._labels) if c in self._class_indices]
        )
        self.test = TransformedSubset(test, indices=[i for i, c in enumerate(test._labels) if c in self._class_indices])

        # Build target bags
        all_indices = list(range(len(self.trainval) + len(self.test)))
        self.cases = [
            [random.choice(all_indices) for _ in range(random.randint(1, self.cfg.max_bag_size))]
            for case_id in range(self.cfg.num_cases)
        ]

    def encode_image(self, img: Image, idx: int) -> str:
        if self.cfg.image_encoding == "base64":
            return rgbnumpy_to_base64(np.array(img))
        elif self.cfg.image_encoding == "abs_filepath":
            split, split_i = self.get_ds_for_index(idx)
            original_indices = split.indices
            image_path = split._images[original_indices[split_i]]
            return str(image_path.absolute())
        else:
            raise ValueError(f"Unknown image encoding: {self.cfg.image_encoding}")

    def build_frame_annotation(self, frame: tuple[Image, tuple[int, Image]], item_index: int | None):
        img, (class_idx, mask) = frame
        if LabelType.seg in self.cfg.for_label:
            res = [
                # 1 is pet, 3 is boundary
                binary_mask_to_result(
                    np.array(mask) == 2, class_name="background", brush_name="segmentation_testlabel"
                ),
                binary_mask_to_result(np.array(mask) != 2, class_name="pet", brush_name="segmentation_testlabel"),
            ]
            for r in res:
                r["item_index"] = item_index
            return res
        return []

    def create_subject_from_case(self, case: list[int], subj_id: str) -> MSubject:
        """
        Converts a torchvision dataset case to a LabelStudioTask for direct import.
        """
        # img, (class_idx, mask) = ds_case
        frames = [self.get_frame(idx) for idx in case]
        images = [self.encode_image(frame[0], idx) for idx, frame in zip(case, frames)]
        frame_results = [self.build_frame_annotation(frame, i_in_group) for i_in_group, frame in enumerate(frames)]
        anno_results = [r for res in frame_results for r in res]

        if LabelType.clf in self.cfg.for_label:
            classes_in_case = set([frame[1][0] for frame in frames])
            classes_not_in_case = set(self._class_indices) - classes_in_case
            class_names_in = [
                (f"Has {self.classes[self._class_indices.index(i)]}", self.classes[self._class_indices.index(i)])
                for i in classes_in_case
            ]
            class_names_out = [
                (f"No {self.classes[self._class_indices.index(i)]}", self.classes[self._class_indices.index(i)])
                for i in classes_not_in_case
            ]

            for class_name, for_label in class_names_in + class_names_out:
                anno_results.append(
                    {
                        "type": "choices",
                        "value": {"choices": [class_name]},
                        "to_name": "image",
                        "from_name": for_label,
                    }
                )
        assert LabelType.surv not in self.cfg.for_label, "Survival labels not supported"
        # if LabelType.surv in self.cfg.for_label:
        #     # Simulate a survival label based on the class index. The higher the class index, the higher the risk score.
        #     anno_results.append(
        #         {
        #             "type": "textarea",
        #             "value": {"text": [f"{class_idx}.0|EVENT"]},
        #             "to_name": "image",
        #             "from_name": "survival_testlabel",
        #         }
        #     )
        return MSubject(
            id=subj_id,
            data={
                # directly embed the image into the task by converting it to base64
                "image": images,
            },
            annotations=[{"result": anno_results}],
            meta={},
        )

    def get_ds_for_index(self, idx: int) -> tuple[datasets.OxfordIIITPet, int]:
        if idx < len(self.trainval):
            return self.trainval, idx
        else:
            return self.test, idx - len(self.trainval)

    def get_frame(self, frame_id: int) -> tuple[Image, tuple[int, Image]]:
        ds, ds_i = self.get_ds_for_index(frame_id)
        return ds[ds_i]

    def __getitem__(self, idx: int) -> tuple[Image, tuple[int, Image]]:
        """
        Returns an image and a tuple of the class index and the segmentation mask.
        """
        return self.cases[idx]

    def __len__(self) -> int:
        return len(self.cases)

    def get_labeling(self) -> LabelingConfig:
        return LabelingConfig(xml=self._build_labeling_config_xml())

    def _build_labeling_config_xml(self) -> str:
        """
        Labeling config for classification and segmentation labels.
        """
        if LabelType.seg in self.cfg.for_label:
            seg_config = """
    <BrushLabels name="segmentation_testlabel" toName="image">
        <Label value="pet" background="rgba(255, 0, 0, 0.4)"/>
        <Label value="background" background="rgba(255, 255, 0, 0.4)"/>
    </BrushLabels>"""
        else:
            seg_config = ""

        if LabelType.clf in self.cfg.for_label:
            # choices = "\n\t\t".join([f'<Choice value="{c}"/>' ])
            clf_config = "\n".join(
                [
                    f"""
    <Choices name="{c}" toName="image" choice="single-radio">
        <Choice value="Has {c}"/>
        <Choice value="No {c}"/>
    </Choices>"""
                    for c in self.classes
                ]
            )
        else:
            clf_config = ""

        if LabelType.surv in self.cfg.for_label:
            surv_config = """
            <TextArea name="survival_testlabel" toName="image" editable="true" />
            """
        else:
            surv_config = ""

        label_config = f"""
<View>
    <Image name="image" valueList="$image" zoom="true"/>
    {seg_config}
    {clf_config}
    {surv_config}
</View>"""
        return label_config
