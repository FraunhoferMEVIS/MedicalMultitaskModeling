from __future__ import annotations

import logging
import random
import re
import uuid
from copy import copy
from typing import Any, Generic, Literal, TypeVar

import logfire
import numpy as np
import torch
from m3_sdk.Repr import Repr
from m3_sdk.utils import convert_results_to_seglabel
from pydantic import Field
from torch.utils.data import Dataset
from typing_extensions import Annotated

from mmm.api.models import BaseResult, MSubject, Result, Volume3DBox, Volume3DImage, Volume3DMask
from mmm.BaseModel import BaseModel
from mmm.data_loading.ClassificationDataset import ClassificationDataset
from mmm.data_loading.MTLDataset import MTLDataset
from mmm.data_loading.MultilabelClassificationDataset import MultilabelClassificationDataset
from mmm.data_loading.SemSegDataset import SemSegDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.mmm_types.GroupUsage import GroupUsage, MaskingStrategy
from mmm.mmm_types.LabelType import LabelType
from mmm.mtl_modules.shared_blocks.Grouper import Grouper
from mmm.mtl_modules.tasks.ClassificationTask import CLF_METRICS, ClassificationTask
from mmm.mtl_modules.tasks.MTLTask import MTLTask, TokenContext
from mmm.mtl_modules.tasks.MultilabelClassificationTask import BCESurvivalTask
from mmm.mtl_modules.tasks.SemSegTask import SemSegTask, get_single_metrics, smp_metrics
from mmm.settings import mtl_settings
from mmm.volume3d import SegMetric3D


class ConsecutiveReprSelector(BaseModel):
    """
    >>> ctxs = [(f"R_{i}", i).__str__().encode() for i in range(5)]  # Generate test contexts
    >>> selection = list(map(Repr.resolve_context, ConsecutiveReprSelector(use_index=1, num_repr=(1, 2)).apply(ctxs)))
    >>> 1 <= len(selection) <= 2
    True

    If the number of contexts is shorter than the number requested, all contexts are returned
    >>> list(map(Repr.resolve_context, ConsecutiveReprSelector(use_index=1, num_repr=(6, 6)).apply(ctxs)))
    [('R_0', 0), ('R_1', 1), ('R_2', 2), ('R_3', 3), ('R_4', 4)]

    Outer contexts are only selected as context window of inner contexts.
    For this reason, this sampling strategy oversamples contexts in the middle in a normal distribution.

    Check the sampling bias like:

    ```python
    contexts = [(f"R_{i}", i).__str__().encode() for i in range(50)]
    ls = []
    for _ in range(1000):
        ls.extend(list(map(
            lambda byt: Repr.resolve_context(byt)[1],
            ConsecutiveReprSelector(use_index=1, num_repr=(1, 50)).apply(contexts)
        )))
    import seaborn as sns
    sns.histplot(ls, bins=50)
    ```

    """

    type: Literal["consecutive"] = "consecutive"
    num_repr: tuple[int, int] = Field(
        (4, 8),
        description="Min and max number of consecutive representations to select",
    )
    use_index: int = Field(
        -1, description="Contexts are tuples. This selects the index of each tuple to use for the sorting. -1 is last."
    )

    def apply(self, contexts: list[bytes]) -> list[bytes]:
        contexts.sort(key=lambda x: Repr.resolve_context(x)[self.use_index])
        # Take a chunk of num_repr consecutive representations by selecting a middle and then growing from there

        if (optimal_size := random.randint(*self.num_repr)) >= len(contexts):
            # Just returns the sorted contexts
            return contexts

        # Do not take middles at the borders to evenly to return uniformly distributed lengths of context lists
        middle_index = random.randint(optimal_size // 2, len(contexts) - 1 - optimal_size // 2)
        # middle_index = random.randint(0, len(contexts) - 1)

        return contexts[max(0, middle_index - optimal_size // 2) : middle_index + optimal_size // 2 + optimal_size % 2]


class RandomReprSelector(BaseModel):
    """
    >>> ctxs = [(f"R_{i}", i).__str__().encode() for i in range(5)]

    RandomReprSelector will never repeat contexts

    >>> indices = [Repr.resolve_context(byt)[1] for byt in RandomReprSelector(num_repr=(2, 8)).apply(ctxs)]
    >>> len(set(indices)) == len(indices)
    True
    """

    type: Literal["random"] = "random"
    num_repr: tuple[int, int] = Field(
        (50, 100),
        description="Min and max number of representations to select",
    )

    def apply(self, contexts: list[bytes]) -> list[bytes]:
        k = random.randint(*self.num_repr)
        return random.sample(contexts, k=min(k, len(contexts)))


ReprSelector = Annotated[ConsecutiveReprSelector | RandomReprSelector, Field(discriminator="type")]


def _build_inputs(reprs: list[Repr], device, collater) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    inputs = [Repr.to_mmm_image(repr) for repr in reprs]
    if MTLDataset.case_is_compressed(inputs[0]):  # assume all are either compressed or not
        valid_inputs = torch.stack([r.tensor for r in reprs]).to(device)
    else:
        valid_inputs = collater(inputs)["image"].to(device)
    supercase_indices = Grouper.extract_ids_from_batch([x.meta.get("group_id") for x in reprs]).to(device)
    contexts = [x.meta.get("context") for x in reprs]
    return valid_inputs, supercase_indices, contexts


def process_for_metrics(
    subjects: list[MSubject], for_model_version: str, label_config: LabelingConfig
) -> dict[str, list[MSubject]]:
    """
    Returns a list of subjects for each label where the subject only contains one annotation and one prediction
    for that label.
    """
    res = {}

    for subject in subjects:
        # First, find the labels that metrics can be computed for: those that have both annotation and prediction
        anno, pred = subject.get_last_updated_annotation(), subject.get_last_prediction(for_model_version)
        if not anno or not pred:
            logging.info(f"Skipping subject {subject.id} because {anno=}, {pred=}")
            continue
        labels_in_common = set([a["from_name"] for a in anno.result]) & set([p["from_name"] for p in pred.result])
        if not labels_in_common:
            logging.warning(f"No common labels in {anno=} and {pred=}")
        for label_name in labels_in_common:
            label_subject = copy(subject)
            label_anno = copy(anno)
            # Should only contain results for the current label
            label_anno.result = [copy(a) for a in anno.result if a["from_name"] == label_name]
            label_pred = copy(pred)
            label_pred.result = [copy(p) for p in pred.result if p["from_name"] == label_name]
            label_subject.annotations = [label_anno]
            label_subject.predictions = [label_pred]
            res.setdefault(label_name, []).append(label_subject)

    return res


D = TypeVar("D", bound=MTLDataset)
T = TypeVar("T", bound=MTLTask)


class MTLAdapterConfig(BaseModel):
    pass


C = TypeVar("C", bound=MTLAdapterConfig)


class MTLLabelExtra(BaseModel):
    """
    Models the settings that control how one label is treated by M3 trainings.
    """

    repr_selector: ReprSelector | None = None
    grouping: GroupUsage | None = None
    token_contexts: list[TokenContext] | None = None
    positions: tuple[int, bool] | None = None

    # used e.g. for augmentations which does not happen for compressed data
    domain: Literal["gigapixel", "tomographic", "xray"] | None = None
    replay_augmentations_for_groups: bool | None = None


E = TypeVar("E", bound=MTLLabelExtra)


class MTLAdapter(Generic[D, T, C, E]):
    Config = MTLAdapterConfig
    Extra = MTLLabelExtra

    @staticmethod
    def build_gt_repr(
        cfg: C, results: list[Result], label_name: str, subject: MSubject, for_instance: Repr, labeling: LabelingConfig
    ) -> Repr:
        raise NotImplementedError

    @staticmethod
    def extract_label(
        cfg: C, mmm_dict: dict[str, Any], gt_repr: Repr, label: dict, train: bool, extra: E
    ) -> dict[str, Any]:
        mmm_dict["meta"] = gt_repr.meta
        return mmm_dict

    @staticmethod
    def build_dataset(cfg: C, src_ds: Dataset, label: dict, train: bool, *args, **kwargs) -> D:
        raise NotImplementedError

    @staticmethod
    def build_task(cfg: C, for_cohort: TrainValCohort[D], for_model, module_name: str, extra: E) -> T:
        raise NotImplementedError

    @staticmethod
    def predict(
        cfg: C,
        fm,
        shared_blocks,
        task: T,
        subject: MSubject,
        t: float,
        label_key: str,
        data_key: str,
        reprs: list[Repr],
        labeling: LabelingConfig,
    ) -> list[Result]:
        raise NotImplementedError

    @staticmethod
    def compute_metrics(cfg: C, subjects: list[MSubject], label: dict) -> dict[str, Any]:
        """
        The annotation and prediction should have only results relevant for this adapter, see process_for_metrics!
        """
        raise NotImplementedError


class ClassificationAdapterConfig(MTLAdapter.Config):
    metrics: list[CLF_METRICS] = ["accuracy", "auc", "confusion matrix", "f1"]


class ClassificationLabelExtra(MTLLabelExtra):
    type: Literal[LabelType.clf] = LabelType.clf


class ClassificationAdapter(
    MTLAdapter[ClassificationDataset, ClassificationTask, ClassificationAdapterConfig, ClassificationLabelExtra]
):
    Config = ClassificationAdapterConfig  # does not use the nested class pattern to enable the use of Generic
    Extra = ClassificationLabelExtra

    @staticmethod
    def build_gt_repr(cfg, results, label_name, subject, for_instance, labeling):
        assert len(results) == 1 and isinstance(
            results[0], BaseResult
        ), f"The classification label is unclear for multiple results within one annotation."
        class_name = results[0].get_multiclass_classification()
        res = Repr(
            tensor=torch.empty(0),  # Placeholder for the class index
            meta={"class_name": class_name},
        )

        if labeling is not None:
            class_names = labeling[label_name]["labels"]
            res.meta["num_classes"] = len(class_names)
            res.tensor = torch.tensor(class_names.index(class_name)).long().unsqueeze(0)
        return res

    @staticmethod
    def build_task(cfg, for_cohort, for_model, module_name, extra):
        return ClassificationTask(
            hidden_dim=for_model[for_model.cfg.squeezer_key].get_hidden_dim(),
            args=ClassificationTask.Config(
                module_name=module_name,
                encoder_key=for_model.cfg.encoder_key,
                squeezer_key=for_model.cfg.squeezer_key,
                grouper_key=for_model.cfg.grouper_key if extra.grouping is None else extra.grouping,
                token_contexts=[] if extra.token_contexts is None else extra.token_contexts,
                positions=extra.positions,
            ),
            cohort=for_cohort,
        )

    @staticmethod
    def extract_label(cfg, mmm_dict, gt_repr, label, train, extra):
        mmm_dict["class"] = label["labels"].index(gt_repr.meta["class_name"])
        return mmm_dict

    @staticmethod
    def build_dataset(cfg, src_ds, label, train, *args, **kwargs):
        return ClassificationDataset(src_ds=src_ds, class_names=label["labels"], *args, **kwargs)

    @staticmethod
    def predict(cfg, fm, shared_blocks, task, subject, t, label_key, data_key, reprs, labeling):
        valid_inputs, supercase_indices, contexts = _build_inputs(reprs, task.torch_device, fm.collate_instances)
        logits = task.forward((valid_inputs, supercase_indices, contexts), shared_blocks)

        if task.args.grouper_key.grouper_key:
            group_meta = task._grouper_meta
            task._grouper_meta = None

            # seqlen, heads
            by_instance_heads = shared_blocks[
                task.args.grouper_key.grouper_key
            ].reducer.reshape_weights_into_instance_heads(group_meta["attn_weights"])

            # seqlen
            by_instance = [
                (ctx, instance_heads.mean().item()) for ctx, instance_heads in zip(contexts, by_instance_heads)
            ]
        else:
            by_instance = None

        if True in ["valueList" in inp.keys() for inp in labeling.get_parsed()[label_key]["inputs"]]:
            logfire.warning("multiple item indices not implemented for classification {label_key}", label_key=label_key)
        scores = torch.softmax(logits, dim=1).cpu()
        result = [
            BaseResult(
                **{
                    "value": {
                        "choices": [task.class_names[torch.argmax(scores[0]).item()]],
                    },
                    "score": torch.max(scores[0]).item(),
                    #   "item_index": i,
                    "from_name": label_key,
                    "to_name": data_key,
                    "type": "choices",
                    "all_class_scores": scores[0].cpu().tolist(),
                    "attn_weights": by_instance,
                }
            )
            # for i in valuelist! each needs a prediction if each item gets a potentially different label
        ]
        return result

    @staticmethod
    def compute_metrics(cfg, subjects, label: dict):
        class_names = label["labels"]
        # First, build three arrays: y_true, y_pred, and y_score
        y_true, y_pred, confidences, scores = [], [], [], []
        for subject in subjects:
            # When computing metrics the subject has exactly the relevant annotation and prediction
            y_true.append(subject.get_last_updated_annotation().result[0]["value"]["choices"][0])
            y_pred.append(subject.get_last_prediction(None).result[0]["value"]["choices"][0])
            confidences.append(subject.get_last_prediction(None).result[0]["score"])
            if scores is not None:
                if hasattr(subject.get_last_prediction(None).result[0], "all_class_scores"):
                    scores.append(subject.get_last_prediction(None).result[0]["all_class_scores"])
                else:
                    scores = None
        y_true_idx = np.array([class_names.index(y) for y in y_true])
        y_pred_idx = np.array([class_names.index(y) for y in y_pred])
        y_score = np.array(scores) if scores is not None else None

        metrics, print_str = ClassificationTask.compute_metrics(
            y_true_idx,
            y_pred_idx,
            y_score,
            cfg.metrics,
            plot_info={
                "classnames": class_names,
            },
        )
        logging.info(f"Computed metrics: {print_str}")

        return metrics


class SegmentationAdapterConfig(MTLAdapter.Config):
    pass


class SegmentationLabelExtra(MTLLabelExtra):
    type: Literal[LabelType.seg] = LabelType.seg
    use_mixer_from_fm: Literal["if_grouper_is_used"] = "if_grouper_is_used"


class SegmentationAdapter(MTLAdapter[SemSegDataset, SemSegTask, SegmentationAdapterConfig, SegmentationLabelExtra]):
    Config = SegmentationAdapterConfig
    Extra = SegmentationLabelExtra

    @staticmethod
    def build_gt_repr(cfg, results, label_name, subject, for_instance, labeling):
        import torchvision.transforms.functional as F

        inp_key = results[0].to_name
        assert False not in [r.to_name == inp_key for r in results], f"Irrelevant results for {label_name=}"
        if results and results[0].item_index is not None:
            if False in [r.item_index == results[0].item_index for r in results]:
                logfire.warning(
                    "Different item indices within one segmentation annotation for {label}: {indices}!",
                    label=label_name,
                    indices=[r.item_index for r in results],
                    subject=subject.model_dump(),
                )

        # Extract the mask if the task has annotations
        full_mask, mask_classes = convert_results_to_seglabel(results)
        gt_meta = {"class_names": mask_classes}
        if for_instance is not None:
            gt_meta["mask_size"] = for_instance.tensor.shape[-2:]
            full_mask = F.resize(
                torch.from_numpy(full_mask).unsqueeze(0),
                gt_meta["mask_size"],
                interpolation=F.InterpolationMode.NEAREST,
            )
        else:
            gt_meta["mask_size"] = full_mask.shape[-2:]

        return Repr(tensor=full_mask, meta=gt_meta)

    @staticmethod
    def build_task(cfg, for_cohort, for_model, module_name, extra):
        task_config = SemSegTask.Config(
            module_name=module_name,
            encoder_key=for_model.cfg.encoder_key,
            squeezer_key=for_model.cfg.squeezer_key,
            decoder_key=for_model.cfg.decoder_key,
            token_contexts=[] if extra.token_contexts is None else extra.token_contexts,
            positions=extra.positions,
        )
        if extra.grouping is not None:
            task_config.grouper_key = extra.grouping
        if task_config.grouper_key.grouper_key and extra.use_mixer_from_fm == "if_grouper_is_used":
            task_config.mixer_key = for_model.cfg.mixer_key
        assert for_cohort.datasets[0].class_names is not None, f"Set {for_cohort.datasets[0].class_names=}"
        return SemSegTask(
            for_cohort.datasets[0].class_names,
            for_model[for_model.cfg.decoder_key],
            for_model[for_model.cfg.squeezer_key],
            task_config,
            for_cohort,
        )

    @staticmethod
    def _build_label_for_class_names(
        common_class_names: list[str], arr: torch.Tensor, class_names_in_tensor: list[str]
    ):
        swap_numbers = {i: common_class_names.index(v) for i, v in enumerate(class_names_in_tensor)}
        assert mtl_settings.ignore_class_value not in swap_numbers.values()
        swap_numbers[mtl_settings.ignore_class_value] = mtl_settings.ignore_class_value
        arr = arr.apply_(lambda x: swap_numbers[x])
        return arr

    @staticmethod
    def extract_label(cfg, mmm_dict, gt_repr, label, train, extra):
        # Add tensor with the class_names from the labeling config
        mmm_dict["label"] = SegmentationAdapter._build_label_for_class_names(
            label["labels"], gt_repr.tensor.long().squeeze(0), gt_repr.meta["class_names"]
        )

        return mmm_dict

    @staticmethod
    def build_dataset(cfg, src_ds, label, train, *args, **kwargs):
        return SemSegDataset(src_ds=src_ds, class_names=label["labels"], *args, **kwargs)

    @staticmethod
    def predict_for_tensor(
        fm,
        sharedblocks,
        mtl_task,
        valid_inputs: torch.Tensor,
        pred_threshold: float,
        original_size: tuple[int, int] | list[tuple[int, int]],
        group_indices=None,
        representation_contexts=None,
    ):
        import torchvision.transforms.functional as F

        logits = mtl_task.forward(
            (
                valid_inputs,
                group_indices,
                representation_contexts,
            ),
            sharedblocks,
        )
        probas = mtl_task.logits_to_probas(logits=logits)
        preds = mtl_task.probas_to_preds(probas, pixel_threshold=pred_threshold)

        # Interpolate back to original size
        probas_orig, preds_orig = [], []
        if isinstance(original_size, list):
            original_sizes: list[tuple[int, int]] = original_size
        else:
            original_sizes: list[tuple[int, int]] = [original_size for _ in range(valid_inputs.shape[0])]
        for i in range(valid_inputs.shape[0]):
            original_sizes[i]
            probas_orig.append(F.resize(probas[i], original_sizes[i], interpolation=F.InterpolationMode.BILINEAR))
            preds_orig.append(
                F.resize(
                    preds[i].unsqueeze(0), original_sizes[i], interpolation=F.InterpolationMode.NEAREST_EXACT
                ).squeeze(0)
            )

        return probas_orig, preds_orig

    @staticmethod
    def predict(cfg, fm, shared_blocks, task, subject, t, label_key, data_key, reprs, labeling):
        from mmm.api.utils import binary_mask_to_result

        valid_inputs, supercase_indices, contexts = _build_inputs(reprs, task.torch_device, fm.collate_instances)
        original_sizes = [r.meta["original_image_size"] for r in reprs]
        probas, preds = SegmentationAdapter.predict_for_tensor(
            fm,
            shared_blocks,
            task,
            valid_inputs,
            original_size=original_sizes,
            pred_threshold=t,
            group_indices=supercase_indices,
            representation_contexts=contexts,
        )
        all_results = []
        for i in range(len(task.class_names)):
            for probas_repr, pred_repr, repr in zip(probas, preds, reprs):
                item_index = repr.meta.get("item_index", None)
                where = pred_repr == i
                if not where.any():
                    continue
                score = probas_repr[i][where].mean().item()
                res = binary_mask_to_result(
                    (pred_repr == i).cpu().numpy(),
                    task.class_names[i],
                    label_key,
                    score=score,
                    image_tag=data_key,
                )
                res["item_index"] = item_index
                all_results.append(BaseResult(**res))

        return all_results

    @staticmethod
    def compute_metrics(cfg, subjects, label: dict):
        class_names = label["labels"]
        tp_fp_fn_tn = []
        for subject in subjects:
            for unique_item_index in set([r.item_index for r in subject.predictions[0].result]):
                anno_results = [r for r in subject.annotations[0].result if r.item_index == unique_item_index]
                pred_results = [r for r in subject.predictions[0].result if r.item_index == unique_item_index]
                if anno_results and pred_results:
                    gt_arr, cls_names_gt = convert_results_to_seglabel(anno_results)
                    pred_arr, cls_names_pred = convert_results_to_seglabel(pred_results)

                    tp_fp_fn_tn.append(
                        get_single_metrics(
                            SegmentationAdapter._build_label_for_class_names(  # Fix order of values in array
                                class_names, torch.from_numpy(gt_arr), cls_names_gt
                            ),
                            SegmentationAdapter._build_label_for_class_names(  # Fix order of values in array
                                class_names, torch.from_numpy(pred_arr), cls_names_pred
                            ),
                            gt_arr.shape[-2:],
                            class_names,
                            ignore_index=mtl_settings.ignore_class_value,
                        )
                    )
        # Compute the metrics
        tp = torch.cat([t[0] for t in tp_fp_fn_tn])
        fp = torch.cat([t[1] for t in tp_fp_fn_tn])
        fn = torch.cat([t[2] for t in tp_fp_fn_tn])
        tn = torch.cat([t[3] for t in tp_fp_fn_tn])
        by_class_ious = smp_metrics.iou_score(tp, fp, fn, tn).mean(dim=0)
        res = {"meanclassiou": by_class_ious.mean().item()}
        res.update({f"{c}iou": by_class_ious[i].item() for i, c in enumerate(class_names)})
        return res


class GeoMaskAdapterConfig(SegmentationAdapterConfig):
    pass


class GeoLabelExtra(SegmentationLabelExtra):
    type: Literal[LabelType.geomask] = LabelType.geomask  # type: ignore


class GeoMaskAdapter(SegmentationAdapter):
    Config = GeoMaskAdapterConfig
    Extra: type[GeoLabelExtra] = GeoLabelExtra

    @staticmethod
    def build_gt_repr(cfg, results, label_name, subject, for_instance, labeling):
        from rasterio.features import rasterize
        from shapely import GeometryCollection, clip_by_rect
        from shapely.affinity import scale, translate

        y, x = for_instance.meta["row_col"]
        assert (
            for_instance.tensor.shape[1] == for_instance.tensor.shape[2]
        ), "Check edge order to implement rectangular masks"
        downsample_fac = for_instance.meta["downsample_fac"]
        height, width = for_instance.tensor.shape[1] * downsample_fac, for_instance.tensor.shape[2] * downsample_fac
        patch_mask = np.zeros(for_instance.tensor.shape[1:], dtype=np.int64)
        patch_mask.fill(mtl_settings.ignore_class_value)
        class_names = []
        for shap in for_instance._buffer["geomask_shapes"]:
            shape_patch: GeometryCollection = clip_by_rect(
                shap[0], xmin=x, ymin=y, xmax=x + 448 * downsample_fac, ymax=y + 448 * downsample_fac
            )
            # Move shape to patch coordinates and downsample
            shape_patch = translate(shape_patch, xoff=-x, yoff=-y)
            shape_patch = scale(shape_patch, xfact=1 / downsample_fac, yfact=1 / downsample_fac, origin=(0, 0))
            if not shape_patch.is_empty:
                if shap[1] not in class_names:
                    class_names.append(shap[1])
                rasterize(
                    shapes=[(shape_patch, class_names.index(shap[1]))],
                    out=patch_mask,
                    fill=mtl_settings.ignore_class_value,
                )

        return Repr(tensor=torch.from_numpy(patch_mask).unsqueeze(0), meta={"class_names": class_names})


class VolumeMaskAdapterConfig(SegmentationAdapterConfig):
    overlap_fraction: float = Field(
        0.2, description="Overlap in predictions. If 0, no overlap, if 0.5 and batchsize 32, 16 instances will overlap"
    )
    batchsize: int = Field(
        16,
        description="Number of instances that are processed at the same time. 1 for 2D processing, -1 for all at once.",
    )
    use_affine_from_image: bool = Field(
        True, description="If True, the affine from the image is used for the output mask."
    )
    metric: SegMetric3D.Config = SegMetric3D.Config()
    # with_positions: None | tuple[int, bool] = (0, True)


class VolumeMaskLabelExtra(SegmentationLabelExtra):
    type: Literal[LabelType.volume_seg] = LabelType.volume_seg  # type: ignore

    return_attentions: bool = Field(default=False)
    exclude_class_from_confidence: list[int] = [0]
    predict_boxes: bool = Field(
        default=False,
        description="Computes bounding boxes around connected components in the volume segmentation.",
    )
    predict_instancemask: bool = Field(
        default=False,
        description="Computes instance masks for connected components in the volume segmentation.",
    )

    repr_selector: ReprSelector | None = ConsecutiveReprSelector(num_repr=(1, 3), use_index=-1)
    replay_augmentations_for_groups: bool | None = True


class VolumeMaskAdapter(SegmentationAdapter):
    Config = VolumeMaskAdapterConfig
    Extra: type[VolumeMaskLabelExtra] = VolumeMaskLabelExtra

    @staticmethod
    def build_gt_repr(cfg, results, label_name, subject, for_instance, labeling):
        # During extract_instances of the image the mask was written into for_instance._buffer
        mask_tensor = for_instance._buffer[label_name][..., for_slice := for_instance.meta["context"][-1]].unsqueeze(0)
        # The mask_tensor is now a view. If no care is taken during serialization, the serialized data will be huge.
        # If Repr._for_serialization is used the tensor will be copied before serialization to avoid this.
        return Repr(
            tensor=mask_tensor,
            meta={"volume_shape": mask_tensor.shape, "for_slice": for_slice, "context": for_instance.meta["context"]},
        )

    @staticmethod
    def build_task(cfg, for_cohort, for_model, module_name, extra):
        return SemSegTask(
            for_cohort.datasets[0].class_names,
            for_model[for_model.cfg.decoder_key],
            for_model[for_model.cfg.squeezer_key],
            SemSegTask.Config(
                module_name=module_name,
                encoder_key=for_model.cfg.encoder_key,
                squeezer_key=for_model.cfg.squeezer_key,
                decoder_key=for_model.cfg.decoder_key,
                grouper_key=for_model.cfg.grouper_key if extra.grouping is None else extra.grouping,
                token_contexts=[] if extra.token_contexts is None else extra.token_contexts,
                mixer_key=for_model.cfg.mixer_key
                if extra.use_mixer_from_fm == "if_grouper_is_used" and for_model.cfg.grouper_key.grouper_key
                else "",
                positions=(0, True) if extra.positions is None else extra.positions,
            ),
            for_cohort,
        )

    @staticmethod
    def extract_label(cfg, mmm_dict, gt_repr, label, train, extra):
        mmm_dict["label"] = gt_repr.tensor.long().squeeze(0)

        # if torch.max(mmm_dict["label"]) >= len(label["labels"]):
        #     logging.warning(f"Label value exceeds number of classes. for {gt_repr=}, {label=}")

        return mmm_dict

    @staticmethod
    def predict(cfg, fm, shared_blocks, task, subject, t, label_key, data_key, reprs, labeling):
        from mmm.volume3d import Volume3DInference

        extra = labeling.get_extra(label_key)
        assert isinstance(subject.data[data_key], Volume3DImage), f"Lists of images not implemented for volumemask."
        H, W = reprs[0].meta["original_image_size"]
        image_affine = torch.tensor(reprs[0].meta["affine"])
        original_size = reprs[0].meta["original_image_size"]

        inference = Volume3DInference()

        DEPTH = len(reprs)

        mask_volume_out: torch.Tensor = torch.zeros((H, W, DEPTH), dtype=torch.int16).fill_(
            mtl_settings.ignore_class_value
        )
        batches = [reprs[i : i + cfg.batchsize] for i in range(0, len(reprs), cfg.batchsize)]
        logfire.info(
            "Predicting volume mask for {num_repr} slices in {num_batches} batches with batchsize",
            num_repr=len(reprs),
            num_batches=len(batches),
            batchsize=cfg.batchsize,
        )
        for batch_reprs in batches:
            valid_inputs, supercase_indices, contexts = _build_inputs(
                batch_reprs, task.torch_device, fm.collate_instances
            )
            probas, preds = SegmentationAdapter.predict_for_tensor(
                fm,
                shared_blocks,
                task,
                valid_inputs,
                pred_threshold=t,
                original_size=tuple(original_size),
                group_indices=supercase_indices,
                representation_contexts=contexts,
            )

            for i, (proba, pred, pos) in enumerate(zip(probas, preds, [r.meta["context"][0] for r in batch_reprs])):
                mask_volume_out[..., pos] = pred
        if mtl_settings.default_log_folder is not None:
            OUT_PATH = mtl_settings.default_log_folder / "volumemasks" / f"{uuid.uuid4()}_volumemask.nii.gz"
        else:
            import os

            from m3_sdk.DistributedPath import DistributedPath

            OUT_PATH = (
                DistributedPath(uri=os.getenv("ML_DATA_OUTPUT", default="/tmp"))
                / "volumemasks"
                / f"{uuid.uuid4()}_volumemask.nii.gz"
            )
        OUT_PATH.parent.upath().mkdir(parents=True, exist_ok=True)
        import nibabel as nib

        nib.save(
            nib.Nifti1Image(mask_volume_out.numpy(), image_affine if cfg.use_affine_from_image else np.eye(4)),
            OUT_PATH.upath(),
        )
        logfire.info("Saved file for volumemask at {out_path}", out_path=OUT_PATH)

        all_best_scores, for_index = torch.stack(probas).max(dim=1)
        # use_channels = [i for i in range(num_classes) if i not in cfg.exclude_class_from_confidence]
        ignore_voxels = torch.zeros_like(all_best_scores)
        for i in extra.exclude_class_from_confidence:
            ignore_voxels[for_index == i] = 1

        res = [Volume3DMask(value=OUT_PATH, to_name=data_key, from_name=label_key)]

        if extra.return_attentions:
            res[0].meta = {
                "attentions": {
                    k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in task._grouper_meta.items()
                }
            }

        if not torch.isnan(score := all_best_scores[~ignore_voxels.bool()].mean()):
            res[0].score = score.item()

        if extra.predict_boxes or extra.predict_instancemask:
            raise NotImplementedError()
            regions = find_regions(probas_volume_out)

            if extra.predict_boxes:
                for region in regions:
                    res.append(
                        Volume3DBox(
                            value={
                                "coordinates": region["region"].bbox,
                                "class": task.class_names[
                                    max(region["region_scores"], key=region["region_scores"].get)
                                ],
                            },
                            to_name=data_key,
                            from_name=label_key,
                            score=max(region["region_scores"].values()),
                        )
                    )

            if extra.predict_instancemask:
                region_vis = render_regions_to_volume(regions, size=(H, W, DEPTH))
                nib.save(
                    nib.Nifti1Image(region_vis.numpy(), image_affine if cfg.use_affine_from_image else np.eye(4)),
                    OUT_PATH.upath().parent / f"{OUT_PATH.upath().stem}_regions.nii.gz",
                )

        return res

    @staticmethod
    def compute_metrics(cfg, subjects, label):
        from monai.transforms import LoadImage

        metric = SegMetric3D(cfg.metric)
        ls = []
        for subject in subjects:
            assert len(subject.annotations) == 1 and len(subject.predictions) == 1
            anno, pred = subject.annotations[0], subject.predictions[0]
            assert len(anno.result) == 1 and pred.result[0].type == "volume3dmask"
            mask_loader = LoadImage(dtype=np.int64, image_only=False, simple_keys=True)
            anno_mask, anno_header = mask_loader(anno.result[0].url.upath())
            pred_mask, pred_header = mask_loader(pred.result[0].url.upath())
            metrics = metric(pred_mask.unsqueeze(0), anno_mask.unsqueeze(0), num_classes=len(label["labels"]))
            ls.append(metrics)
        res = {}
        for metric_name in cfg.metric.metrics:
            metrics = torch.cat([l[metric_name] for l in ls], dim=0)
            res.update({f"{metric_name}_{c}": metrics[:, i].mean() for i, c in enumerate(label["labels"])})
            res[f"mean{metric_name}"] = torch.tensor(list(res.values())).mean()
        return res


def _augment_target(mmm_instance, bins, augment_event_for_cens_prob, target_aug_strength):
    # augment the regression target
    mmm_instance["meta"]["original_regression_target"] = mmm_instance["target"]
    mmm_instance["target"] = min(
        max(min(bins), mmm_instance["target"] + np.random.normal(0, target_aug_strength)),
        max(bins),
    )

    # augment an event in case there is censoring with a small probability
    if np.random.rand() < augment_event_for_cens_prob:
        has_event = mmm_instance["meta"]["event"] == 1
        mmm_instance["meta"]["original_event"] = mmm_instance["meta"]["event"]
        mmm_instance["meta"]["event"] = 1 if not has_event else 0
        if has_event:
            # The regression target should be moved to a target between min(bins) and previous, possibly augmented, target
            mmm_instance["target"] = np.random.uniform(min(bins), mmm_instance["target"])
        else:
            # The regression target should be moved to a target between previous, possibly augmented, target and max(bins)
            mmm_instance["target"] = np.random.uniform(mmm_instance["target"], max(bins))
    return mmm_instance


class SurvivalAdapterConfig(MTLAdapter.Config):
    pass


class SurvivalLabelExtra(MTLLabelExtra):
    type: Literal[LabelType.surv] = LabelType.surv

    target_aug_strength: float = Field(
        default=2.0, description="Variance of the normal distribution used for target augmentation."
    )
    augment_event_for_cens_prob: float = 0.1
    target_factor: float = Field(default=1.0, description="This can be used to scale the target time to the bins.")


class SurvivalAdapter(
    MTLAdapter[MultilabelClassificationDataset, BCESurvivalTask, SurvivalAdapterConfig, SurvivalLabelExtra]
):
    Config = SurvivalAdapterConfig
    Extra = SurvivalLabelExtra

    @staticmethod
    def build_gt_repr(cfg, results, label_name, subject, for_instance, labeling):
        assert len(results) == 1, f"The survival label is unclear for multiple results within one annotation."
        text_representations = results[0]["value"]["text"]
        assert len(text_representations) == 1, f"For survival, {text_representations=} should have exactly one element"
        time, event = float((parts := text_representations[0].split("|"))[0]), parts[1] == "EVENT"
        return Repr(tensor=torch.tensor([event, time]).float(), meta={})

    @staticmethod
    def build_task(cfg, for_cohort, for_model, module_name, extra):
        return BCESurvivalTask(
            hidden_dim=for_model[for_model.cfg.squeezer_key].get_hidden_dim(),
            args=BCESurvivalTask.Config(
                module_name=module_name,
                encoder_key=for_model.cfg.encoder_key,
                squeezer_key=for_model.cfg.squeezer_key,
                grouper_key=for_model.cfg.grouper_key if extra.grouping is None else extra.grouping,
                token_contexts=[] if extra.token_contexts is None else extra.token_contexts,
                positions=extra.positions,
            ),
            cohort=for_cohort,
        )

    @staticmethod
    def extract_label(cfg, mmm_dict, gt_repr, label, train, extra):
        event, targettime = bool(gt_repr.tensor[0].item()), gt_repr.tensor[1].item()

        if extra.target_factor != 1.0:
            targettime = targettime * extra.target_factor

        mmm_dict["meta"]["event"] = event
        mmm_dict["target"] = targettime
        bins = torch.Tensor(label.get("bins", [float(b) for b in range(0, 25)]))
        if train:
            mmm_dict = _augment_target(
                mmm_dict,
                bins=torch.Tensor(label.get("bins", [float(b) for b in range(0, 25)])),
                augment_event_for_cens_prob=extra.augment_event_for_cens_prob,
                target_aug_strength=extra.target_aug_strength,
            )

        return BCESurvivalTask.clfdataset_from_regdataset(mmm_dict, bins=bins)

    @staticmethod
    def build_dataset(cfg, src_ds, label, train, *args, **kwargs):
        bins: torch.Tensor = torch.Tensor(label.get("bins", [float(b) for b in range(0, 25)]))
        # ts = [src_transform] if src_transform is not None else []

        return MultilabelClassificationDataset(
            src_ds,
            # src_transform=transforms.Compose(
            #     ts + [ApplyToList(partial(BCESurvivalTask.clfdataset_from_regdataset, bins=bins))]
            # ),
            class_names=[f"t{b:.2f}" for b in bins],
            *args,
            **kwargs,
        )

    @staticmethod
    def predict(cfg, fm, shared_blocks, task, subject, t, label_key, data_key, reprs, labeling):
        logits = task.forward(_build_inputs(reprs, task.torch_device, fm.collate_instances), shared_blocks)
        if True in ["valueList" in inp.keys() for inp in labeling.get_parsed()[label_key]["inputs"]]:
            logging.warning(f"multiple item indices not implemented for time-to-event {label_key}")
        logging.error(f"TTE prediction not implemented")
        return []


ADAPTERS: dict[LabelType, type[MTLAdapter]] = {
    LabelType.clf: ClassificationAdapter,
    LabelType.seg: SegmentationAdapter,
    LabelType.volume_seg: VolumeMaskAdapter,
    LabelType.surv: SurvivalAdapter,
    LabelType.geomask: GeoMaskAdapter,
}
LabelExtra = Annotated[
    ClassificationLabelExtra | SegmentationLabelExtra | VolumeMaskLabelExtra | SurvivalLabelExtra,
    Field(discriminator="type"),
]


class LabelingConfig(BaseModel):
    xml: str
    m3_extra: dict[str, LabelExtra] | None = None
    _parsed: dict | None = None

    def __getitem__(self, key):
        return self.get_parsed()[key]

    def get_all_labeltypes_for_input_key(self, input_key: str):
        """
        Often, computing multiple types of features at once makes sense.
        This function can be used to find all label types that are associated with a given input key.
        """
        output_keys_with_input_key = [
            output_key
            for output_key, output_cfg in self.get_parsed().items()
            if input_key in [d["value"] for d in output_cfg["inputs"]]
        ]
        output_types = [
            LabelingConfig.determine_type_of_label(self.get_parsed()[output_cfg])
            for output_cfg in output_keys_with_input_key
        ]
        return output_keys_with_input_key, output_types

    def get_extra(self, label_name: str) -> LabelExtra:
        with logfire.span(
            "Discovering extra for {label_name} from {patterns}",
            label_name=label_name,
            patterns=list(self.m3_extra.keys()) if self.m3_extra else [],
        ):
            if self.m3_extra is not None:
                matches = {pattern: extra for pattern, extra in self.m3_extra.items() if re.match(pattern, label_name)}
                assert len(matches) <= 1, f"Found multiple matching extras: {matches.keys()}"
                if len(matching_patterns := list(matches.keys())) == 1:
                    res = self.m3_extra[matching_patterns[0]]
                    logging.info(f"Matched {matching_patterns=}, {res.model_dump_json(exclude_none=True)}")
                    return res

            # if self.m3_extra and label_name in self.m3_extra:
            #     return self.m3_extra[label_name]
            label_type: LabelType = LabelType.from_string(self.determine_type_of_label(self.get_parsed()[label_name]))
            logging.info(f"No matching extra found, using defaults for {label_name} with {label_type=}")
            return ADAPTERS[label_type].Extra()  # type: ignore

    @staticmethod
    def determine_type_of_label(label: dict) -> str:
        """
        Expects a dictionary with a 'type': str, 'name': str, ...
        """
        if label["type"].lower() == "textarea":
            # This needs to be extended once there are more TextArea types
            return "survival"
        else:
            return label["type"].lower()

    def get_all_labeltypes(self):
        return list(set([LabelingConfig.determine_type_of_label(x) for x in self.get_parsed().values()]))

    def get_parsed(self):
        if self._parsed is None:
            from label_studio_sdk.converter.utils import parse_config

            self._parsed = parse_config(self.xml)

        return self._parsed
