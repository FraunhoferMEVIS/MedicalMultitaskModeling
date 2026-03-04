from __future__ import annotations

import json
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from mmm.transforms import KeepOnlyKeysInDict

from .MTLDataset import MTLDataset, SrcCaseType
from .SemSegDataset import SemSegDataset


class MultilabelClassificationDataset(MTLDataset):
    """
    Expects cases with the keywords:

    - image: (typical image assumptions of mmm)
    - class_labels: torch.FloatTensor with a float between 0. and 1. for the confidence of that class being True

    Optional:

    - loss_weights: Allows to ignore certain class during loss computation. Use this to deal e.g. with NaNs targets
    """

    @staticmethod
    def from_semseg(ds: SemSegDataset, ignore_classes: Optional[List[int]] = None) -> MultilabelClassificationDataset:
        """
        Uses ds in-place to construct a multilabel classification dataset by appending a batch transform.
        """
        if ignore_classes is None:
            ignore_classes = []
        assert len(ignore_classes) == len(set(ignore_classes)), f"Don't ignore a class twice {ignore_classes=}"

        def semsegbatch_to_mclf_batch(num_classes: int, c, semsegbatch: Dict[str, Any]) -> Dict[str, Any]:
            masks = semsegbatch["label"]
            class_labels = []
            for i in range(masks.shape[0]):
                x = torch.zeros(num_classes)
                original_classes = torch.unique(masks[i, :]).tolist()
                x[[c[x] for x in original_classes if x in c]] = 1
                class_labels.append(x)

            semsegbatch["class_labels"] = torch.stack(class_labels)
            return semsegbatch

        class_names = [v for i, v in enumerate(ds.class_names) if i not in ignore_classes]
        conv = {}
        for i, _ in enumerate(ds.class_names):
            if i not in ignore_classes:
                conv[i] = len(conv)

        mclf_ds = MultilabelClassificationDataset(
            ds.src_ds,
            class_names=class_names,
            src_transform=ds.src_transform,
            batch_transform=ds.batch_transform,
            collate_fn=transforms.Compose(
                [
                    ds.collate_fn,
                    partial(semsegbatch_to_mclf_batch, len(class_names), conv),
                ]
            ),
        )
        # For this approach to work the data stripper needs to chill and allow semseg and multilabel keys
        mclf_ds.data_stripper = KeepOnlyKeysInDict(
            keys=set(list(ds.data_stripper.keys) + list(mclf_ds.data_stripper.keys)),
        )

        return mclf_ds

    def __init__(self, src_ds: Dataset[SrcCaseType], class_names: List[str], **kwargs) -> None:
        self.class_names: List[str] = class_names
        super().__init__(src_ds, **kwargs)

    @staticmethod
    def get_mandatory_keys() -> list[str]:
        return super(MultilabelClassificationDataset, MultilabelClassificationDataset).get_mandatory_keys() + [
            "image",
            "class_labels",
        ]

    @staticmethod
    def get_optional_keys() -> list[str]:
        return super(MultilabelClassificationDataset, MultilabelClassificationDataset).get_optional_keys() + [
            "loss_weights"
        ]

    def verify_case(self, case):
        if "meta" in case and "imagetype" in case["meta"] and case["meta"]["imagetype"] == "compressed":
            len(case["image"].shape) == 1
        else:
            self.assert_image_data_assumptions(case["image"])
        assert isinstance(case["class_labels"], torch.FloatTensor), "Labels should be confidences between 0. and 1."
        assert len(case["class_labels"]) == len(self.class_names)
        assert torch.min(case["class_labels"]) >= 0.0 and torch.max(case["class_labels"]) <= 1.0

        if "loss_weights" in case:
            assert isinstance(case["loss_weights"], torch.FloatTensor), "Loss weights should be float between 0. and 1."
            assert len(case["loss_weights"]) == len(self.class_names)
            assert torch.min(case["loss_weights"]) >= 0.0 and torch.max(case["loss_weights"]) <= 1.0

    def get_input_output_tuple(self, batch: Dict[str, Any]) -> Tuple[Any, ...]:
        return batch["image"], batch["class_labels"]

    def st_case_viewer(self, ls: list[dict[str, Any]], i: int = -1) -> None:
        from mmm.logging.st_ext import Image2D, M3Image, m3_image, st

        m3_image(
            data=M3Image.Data(
                images=[
                    Image2D.from_tensor(
                        img=d["image"],
                        class_names=self.class_names,
                        desc=json.dumps(d["meta"], indent=2, default=str) if "meta" in d else None,
                        caption=self._label_to_html(
                            d["class_labels"],
                            d["loss_weights"] if "loss_weights" in d else None,
                            context=d.get("meta", {}).get("context", ()),
                        ),
                    )
                    for d in ls
                ],
            ),
            key=f"img{i}_original",
        )

    def _compute_batchsize_from_batch(self, batch: Dict[str, Any]) -> int:
        return batch["image"].shape[0]

    def _visualize_batch_case(self, batch: Dict[str, Any], i: int) -> None:
        from mmm.logging.st_ext import Image2D, M3Image, m3_image, st

        patch, class_labels = batch["image"][i], batch["class_labels"][i]

        # self._print_relevant_classes(class_labels, batch["loss_weights"][i] if "loss_weights" in batch else None)

        try:
            assert batch["meta"][i]["imagetype"] == "compressed"
            st.json(batch["meta"][i])

            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots()
            ax.set_title(f"Compressed image: {patch.shape}")
            sns.histplot(patch, ax=ax)
            st.plotly_chart(fig)

        except (KeyError, AssertionError):
            m3_image(
                data=M3Image.Data(
                    images=[
                        Image2D.from_tensor(
                            img=batch["image"][i],
                            class_names=self.class_names,
                            desc=json.dumps(batch["meta"][i], indent=2, default=str) if "meta" in batch else None,
                            caption=self._label_to_html(
                                batch["class_labels"][i],
                                batch["loss_weights"][i] if "loss_weights" in batch else None,
                                context=batch.get("meta", {}).get(i, {}).get("context", ()),
                            ),
                        )
                    ],
                    group_meta={"nogroup": 14},
                ),
                key=f"img{i}_original",
            )

    def _label_to_html(
        self, class_labels: torch.Tensor, loss_weights: None | torch.Tensor = None, context: tuple = ()
    ) -> str:
        classes_repr = ", ".join(
            [
                f'<span style="color:green">{v}</span>' if v > 0.0 else f'<span style="color:red">{v}</span>'
                for i, v in enumerate(class_labels)
            ]
        )
        html = f"{classes_repr} (Classes)<br>"
        if loss_weights is not None:
            weight_repr = ", ".join(
                [
                    f'<span style="color:green">{v}</span>' if v > 0.0 else f'<span style="color:red">{v}</span>'
                    for i, v in enumerate(loss_weights)
                ]
            )
            html += f"{weight_repr} (Weights)"
        if context:
            html += f"<br>Context: <span style='color:orange'>{context}</span>"
        return html
