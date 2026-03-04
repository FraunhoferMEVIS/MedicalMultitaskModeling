import logging
import random
import uuid
from copy import deepcopy
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import logfire
import monai.transforms as monai_transforms
import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.metrics import compute_dice
from monai.transforms import LoadImage, Resize
from pydantic import Field

from mmm.data_loading.MTLDataset import mtl_collate
from mmm.utils import make_divisable_by

try:
    import gzip

    import nibabel as nib
    from nibabel import FileHolder, Nifti1Image
except ImportError:
    if not TYPE_CHECKING:
        nib, FileHolder, Nifti1Image, gzip = None, None, None, None
    else:
        raise  # Avoids errors in type checking tools

from m3_sdk.DistributedPath import DistributedPath

from mmm.BaseModel import BaseModel
from mmm.torch_ext import CachingSubCaseDSSampler
from mmm.transforms import ApplyToKey, RandomApply, UnifySizes

# from monai.data.meta_obj import set_track_meta
# set_track_meta(False)  # Improves efficiency when the metadata tracked by MONAI is not used


def find_regions(
    scores_chwd: torch.FloatTensor, background: int = 0, background_threshold=0.5, spacing=None
) -> list[dict[str, Any]]:
    """
    Uses connected components analysis from segmentation scores.
    First, the holes in the background class are used for candidate regions.
    Then, for each region its properties and class scores are returned.

    Args:
        prediction_chwd: (Classes, H, W, D) tensor with class scores between 0 and 1, usually obtained from softmax
        background: class id of the background class

    Returns:
        list of dictionaries with region properties and class scores
    """
    from skimage.measure import label, regionprops

    background_mask = scores_chwd[background] < background_threshold
    labeled_array, num_features = label(background_mask, connectivity=1, return_num=True)
    res = []
    for region in regionprops(labeled_array, spacing=spacing):
        foreground = torch.from_numpy(region.image).to(scores_chwd.device)
        region_scores = scores_chwd[(...,) + region.slice]

        _, region_pred = torch.max(region_scores, dim=0)
        confidences = {
            class_index.item(): region_scores[class_index][foreground].mean().item()
            for class_index in region_pred.unique()
            if class_index != background
        }
        res.append(
            {
                "region": region,
                "region_scores": confidences,
            }
        )
    return res


def render_regions_to_volume(
    regions: list[dict[str, Any]], size: tuple[int, int, int], mode: Literal["masks", "boxes"] = "masks"
) -> torch.FloatTensor:
    """
    Stamp each region onto the output volume with an intensity according to its highest class score
    """
    out = torch.zeros(size, dtype=torch.float32)
    for region in regions:
        highest_confidence = max(region["region_scores"].values())
        foreground = torch.from_numpy(region["region"].image).to(out.device)
        intensity = highest_confidence * 255.0
        if mode == "masks":
            out[region["region"].slice][foreground] = intensity
        elif mode == "boxes":
            out[region["region"].slice] = intensity
        else:
            raise ValueError(f"Unknown mode {mode}")
    return out


def volume_scale_to_0_1(img: torch.Tensor, windowing: None | tuple[float, float] = None) -> torch.Tensor:
    """
    Scales the input array to the range [0, 1].
    """
    # If the image consists only of zeros, return it:
    if torch.count_nonzero(img) == 0:
        return img
    # Make sure extreme outliers do not influence the scaling
    if windowing is None:
        return monai_transforms.ScaleIntensityRangePercentiles(lower=2.5, upper=97.5, b_min=0.0, b_max=1.0, clip=True)(
            img
        )
    else:
        return monai_transforms.ScaleIntensityRange(
            a_min=windowing[0], a_max=windowing[1], b_min=0.0, b_max=1.0, clip=True
        )(img)


def load_nifti_from_distpath(dp: DistributedPath) -> nib.nifti1.Nifti1Image:
    """
    Does not require to save the file to disk.
    .nii.gz is compressed nifti file, so we need to decompress it first.
    """
    with dp.file().open() as f:
        fh = FileHolder(fileobj=BytesIO(gzip.GzipFile(fileobj=f).read()))
    img = Nifti1Image.from_file_map({"header": fh, "image": fh})
    return img


class ManyClassesSampler(CachingSubCaseDSSampler):
    """
    Oversamples samples with multiple classes to balance the dataset.
    """

    class Config(BaseModel):
        max_context: int = Field(
            default=5,
            description="Number of context instances. If 0, no context is used.",
        )

    def __init__(self, cfg: Config = None, inform_about_weirdness: bool = False):
        super().__init__()
        self.cfg = cfg if cfg is not None else self.Config()
        self.context_indices = []
        self.meta_suffix = ""  # appended to context windows to avoid duplicate group ids

        # Statistics
        self.diverse_yielded = 0
        self.not_diverse_yielded = 0
        self.inform_about_weirdness = inform_about_weirdness

    def hook_new_subcases(self, subcases: list):
        # Count the pure background cases
        only_background = [not self.is_diverse(subcase) for subcase in subcases]
        sample_metas = [{"repeat": 1} for _ in subcases]

        num_background = sum(only_background)
        if num_background == 0:
            if self.inform_about_weirdness:
                logging.debug(f"No background cases in a list of cases such as {subcases[0]} in {self=}")
            return list(zip(subcases, sample_metas))
        if num_background == len(subcases):
            if self.inform_about_weirdness:
                logging.info(f"Only background cases such as {subcases[0]} in {self=}")
            return list(zip(subcases, sample_metas))

        target_num = num_background * 3
        # Compute how many diverse samples should be repeated to balance this case
        repeat_num = max(0, target_num - len(subcases))
        diverse_indices = [i for i, is_only_bg in enumerate(only_background) if not is_only_bg]
        up_to_average = min(1, (target_num - len(subcases)) // len(subcases) + 1)
        # print(f"{up_to_average=}, {repeat_num=}, {len(subcases)=}, {num_background=}, {target_num=}")
        while repeat_num > 0:
            # Find a diverse sample from sample_metas
            diverse_index = random.choice(diverse_indices)
            repeat_times = random.randint(1, up_to_average)
            sample_metas[diverse_index]["repeat"] += repeat_times
            repeat_num -= repeat_times

        return list(zip(subcases, sample_metas))

    def decide_removal(self, popped_case: dict, draining_phase: bool, index: int) -> bool:
        if draining_phase:
            res = True
        else:
            _, sample_meta = popped_case
            res = sample_meta["repeat"] <= 0

        if res:
            self.context_indices = [
                ctx_index - 1 if ctx_index > index else ctx_index for ctx_index in self.context_indices
            ]

        return res

    def set_context(self, main_subcase_index: int) -> list[int]:
        """
        If the context should be used, the respective indices are computed here.
        """
        if self.cfg.max_context == 0:
            raise NotImplementedError

        # Find the index of the first subcase in the group (assuming the cache is sorted like [g1, g1, g2, ->g2, g2])
        index_where_group_starts = main_subcase_index
        main_group_id = self.cacheds.subcases[main_subcase_index][0]["meta"]["group_id"]
        for distance in range(1, min(self.cfg.max_context + 1, index_where_group_starts)):
            if self.cacheds.subcases[main_subcase_index - distance][0]["meta"]["group_id"] != main_group_id:
                break
            index_where_group_starts = main_subcase_index - distance

        index_where_group_ends = main_subcase_index
        for distance in range(1, min(self.cfg.max_context + 1, len(self.cacheds.subcases) - index_where_group_ends)):
            if self.cacheds.subcases[main_subcase_index + distance][0]["meta"]["group_id"] != main_group_id:
                break
            index_where_group_ends = main_subcase_index + distance

        return list(range(index_where_group_starts, index_where_group_ends + 1))

    def sample_from_cache(self, draining_phase: bool) -> int:
        if not self.context_indices:
            sampling_weights = [meta["repeat"] for _, meta in self.cacheds.subcases]
            subcase_index = random.choices(range(len(self.cacheds.subcases)), weights=sampling_weights)[0]
            context_indices = self.set_context(subcase_index)
            self.context_indices, self.meta_suffix = context_indices, f"_p{len(context_indices)}_{uuid.uuid4().hex[:6]}"

        res = self.context_indices.pop(0)
        subcase, subcase_meta = self.cacheds.subcases[res]
        if random.random() < 0.05:  # should be enough to gather some useful statistics
            if self.is_diverse(subcase):
                self.diverse_yielded += 1
            else:
                self.not_diverse_yielded += 1
            if (self.diverse_yielded + self.not_diverse_yielded) % 500 == 0:
                logging.debug(f"Diverse: {self.diverse_yielded}, Not diverse: {self.not_diverse_yielded}, {self=}")
        subcase_meta["repeat"] -= 1
        return res

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def postprocess_subcase(self, subcase: tuple):
        # Internally, the sampler represents the subcase as a tuple (subcase, sampling_information)
        res = subcase[0]
        res["meta"] = deepcopy(res.get("meta", {}))
        res["meta"]["group_id"] += self.meta_suffix
        return res

    @staticmethod
    def is_diverse(case: dict):
        return torch.unique(case["label"]).numel() > 1


class MemFriendlyCrop(monai_transforms.RandCropByLabelClassesd):
    def __call__(self, data, lazy=None):
        d = dict(data)
        self.randomize(d.get(self.label_key), d.pop(self.indices_key, None), d.get(self.image_key))  # type: ignore

        # initialize returned list with shallow copy to preserve key ordering
        ret: list = [{k: v for k, v in d.items() if k not in self.keys} for _ in range(self.cropper.num_samples)]

        # deep copy all the unmodified data
        # for i in range(self.cropper.num_samples):
        #     for key in set(d.keys()).difference(set(self.keys)):
        #         ret[i][key] = deepcopy(d[key])

        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(d):
            for i, im in enumerate(self.cropper(d[key], randomize=False, lazy=lazy_)):
                ret[i][key] = im.clone()
        for i in range(self.cropper.num_samples):
            ret[i]["meta"] = deepcopy(d["meta"])
            # the label key was only used for patching, it is not needed anymore
            if self.label_key not in self.keys:
                ret[i].pop(self.label_key, None)

        return ret


class Tomo3DProcessor:
    """
    For 3D data such as tomographic CT and MRI.

    Images have the shape (C, spatials..., extra-dimensions) where extra dimension is expected to exist on the mask.
    If an extra dimension does not exist on the mask the image is expected to be list[MetaTensor] instead of MetaTensor.
    In that case, the mask will be assumed to be the same for all entries in the list.

    Populates the meta information with a key "tomo_info" which contains statistics about the 3D image.

    For example, for segmentation with some basic augmentations

    >>> from mmm.volume3d import Tomo3DProcessor
    >>> Tomo3DProcessor(
    ...     Tomo3DProcessor.Config(rotate_3d="always"),
    ...     augs_constructor=Tomo3DProcessor.base_volume_augs,
    ...     with_segmask=True)
    Tomo3DProcessor(normalize=True fix_orientation='never' rotate_3d='always' divisable_by=None spacing=None) with augs: base_volume_augs
    """

    class Config(BaseModel):
        normalize: bool = True
        fix_orientation: Literal["never", "smallest_axis_into_depth"] | str = Field(default="never")
        rotate_3d: Literal["never", "always", "ifcubical"] = Field(
            default="ifcubical",
            description="ifcubical -> rotates if the shortest axis is longer than half the longest axis",
        )
        divisable_by: None | tuple[int, int, int] = Field(
            default=None,
            description="Each edge will be padded to be divisable. fix_orientation will be applied before padding.",
        )
        spacing: None | tuple[float, float, float] = Field(
            default=None,
            description="Spacing between voxels after applying fix-orientation",
        )

    def __repr__(self) -> str:
        repr = f"Tomo3DProcessor({self.args})"
        try:
            repr += f" with augs: {self.augs.__name__}"
        except Exception:  # __name__ is not available for lambdas
            repr += f" with {self.augs} augmentations"
        return repr

    def extract_slice(self, d: dict, i: int, slice_axis: int, image_index: int | None = None, clone: bool = True):
        image: torch.Tensor = d["image"][image_index] if isinstance(d["image"], list) else d["image"]
        num_slices = image.shape[-1]
        subcase = {"image": torch.index_select(image, slice_axis, torch.tensor(i)).squeeze(slice_axis)}

        # If the slice axis is the last axis (-1), the following will do subcase[key] = d[key][..., i]
        if self.with_segmask and "label" in d:
            subcase["label"] = torch.index_select(d["label"], slice_axis, torch.tensor(i)).squeeze(slice_axis)

        if clone:  # Avoid issues with tensor views
            subcase["image"] = subcase["image"].clone()
            if "label" in subcase:
                subcase["label"] = subcase["label"].clone()
        subcase["meta"] = {}
        if "meta" in d:
            subcase["meta"]["supermeta"] = d["meta"]
            subcase["meta"]["axis"] = slice_axis
            subcase["meta"]["subcaseslice"] = f"{i}/{num_slices-1}"
            if "group_id" in d["meta"]:
                # The group id is used for multiple instance learning and is expected to exist in the top level
                subcase["meta"]["group_id"] = d["meta"]["group_id"]

            subcase["meta"]["context"] = d["meta"].get("context", ()) + (i,)
            if image_index is not None:
                subcase["meta"]["context"] = subcase["meta"]["context"] + (image_index,)
                subcase["meta"]["image_index"] = image_index

        return subcase

    def extract_slices(self, d: dict, slice_axis: int = -1):
        res = []
        image_shape = (d["image"][0] if isinstance(d["image"], list) else d["image"]).shape
        for i in range(image_shape[slice_axis]):
            for image_index in range(len(d["image"])) if isinstance(d["image"], list) else [None]:
                res.append(self.extract_slice(d, i, slice_axis, image_index))

        return res

    @staticmethod
    def repeat_channels(img: torch.Tensor, num_channels=3):
        if img.shape[0] != num_channels:
            return torch.cat([img] * (num_channels // img.shape[0]), dim=0)
        return img

    @staticmethod
    def get_3d_rotations(keys):
        """
        If the shortest axis is longer than half the longest axis 3d rotations are a good idea.
        """
        return [
            monai_transforms.RandRotate90D(keys=keys, prob=0.2, spatial_axes=(0, 1)),
            monai_transforms.RandRotate90D(keys=keys, prob=0.2, spatial_axes=(0, 2)),
            monai_transforms.RandRotate90D(keys=keys, prob=0.2, spatial_axes=(1, 2)),
        ]

    @staticmethod
    def base_volume_augs(keys) -> list:
        only_image_keys = [k for k in keys if k != "label"]
        return [
            monai_transforms.RandAdjustContrastD(keys=only_image_keys, prob=0.1),
            RandomApply(monai_transforms.RandScaleCropD(keys=keys, roi_scale=[0.8, 0.8, 0.8], random_size=True), p=0.3),
        ]

    def __init__(self, args: Config, augs_constructor: Callable | None, with_segmask: bool) -> None:
        self.args = args
        self.with_segmask = with_segmask
        # self.aug_keys = ["image", "label"] if with_segmask else ["image"]

        # If path is a directory it is assumed to be DICOM, otherwise Nifti
        self.image_loader = LoadImage(dtype=np.float32, image_only=False, simple_keys=True)
        if with_segmask:
            self.mask_loader = LoadImage(dtype=np.int64, image_only=False, simple_keys=True)
        self.augs: Callable | None = augs_constructor

    def extract_statistics(self, d, key) -> dict:
        """
        Extracts statistics about the 3D image.
        """
        img = d[key]
        res = {
            "min": img.min().item(),
            "max": img.max().item(),
            "shape": img.shape,
        }
        if torch.is_floating_point(img):
            res["mean"] = img.mean().item()
            res["std"] = img.std().item()

        return res

    def m3_to_monai(self, d: dict[str, Any]):
        if isinstance(d["image"], list):
            aug_keys = []
            num_images = len(d["image"])
            for i, image in enumerate(d.pop("image")):
                img_key = f"image{i}"
                d[img_key] = image
                aug_keys.append(img_key)
        else:
            num_images = -1
            aug_keys = ["image"]

        for key in aug_keys:
            assert key in d and len(d[key].shape) >= 4, f"C, spatials... dimensions expected for {d['meta']}"

        if self.with_segmask:
            for key in aug_keys:
                assert len(d[key].shape) == len(d["label"].shape) + 1
            aug_keys = aug_keys + ["label"]
            # Monai expects a channel dimension which is always empty for multiclass problems
            d["label"] = torch.unsqueeze(d["label"], 0)
        return d, num_images, aug_keys

    def fix_orientation(self, d, aug_keys):
        if self.args.fix_orientation == "never":
            orientation_meta = "never"
        elif self.args.fix_orientation == "smallest_axis_into_depth":
            smallest_axis = int(torch.argmin(torch.tensor(d[aug_keys[0]].shape[1:])).item())
            if smallest_axis != 2:
                rotator = monai_transforms.Rotate90D(keys=aug_keys, spatial_axes=(smallest_axis, 2))
                d = rotator(d)
            orientation_meta = smallest_axis
        else:
            d = monai_transforms.OrientationD(keys=aug_keys, axcodes=self.args.fix_orientation)(d)
            orientation_meta = self.args.fix_orientation

        d.setdefault("meta", {})["orientation"] = orientation_meta

        if self.args.spacing is not None:
            d = monai_transforms.SpacingD(
                # scale_extent=True matches sitk behaviour in edge cases
                keys=aug_keys,
                pixdim=self.args.spacing,
                scale_extent=True,
                align_corners=True,
            )(d)
        return d

    def normalize(self, d, aug_keys):
        for key in aug_keys:
            if key != "label":
                d[key] = volume_scale_to_0_1(d[key])
        return d

    def monai_to_m3(self, d: dict[str, Any], num_images: int, aug_keys: list[str]):
        # Before returning, get rid of the MONAI meta tensor
        if "meta" not in d:
            d["meta"] = {}
        for key in aug_keys:
            if isinstance(d[key], MetaTensor):
                d["meta"].setdefault("monai_meta", {})[key] = d[key].meta
                d["meta"].setdefault("tomo_info", {})[key] = self.extract_statistics(d, key)
                d[key] = d[key].as_tensor()

        if num_images != -1:
            d["image"] = [d.pop(f"image{i}") for i in range(num_images)]
        if self.with_segmask:
            d["label"] = d["label"][0]
        # if not isinstance(d["image"], list):
        #     d["meta"]["tomo_info"] = self.extract_statistics(d)
        return d

    def rotate_if_applicable(self, d, aug_keys):
        use_3d_rot = self.args.rotate_3d == "always"
        if self.args.rotate_3d == "ifcubical":
            edge_lengths = torch.tensor(d[aug_keys[0]].shape[1:])  # if one image is cubical, all are
            min_length = torch.max(edge_lengths) / 2
            use_3d_rot = torch.min(edge_lengths) >= min_length
            logging.debug(f"Using 3d rotations: {use_3d_rot=} because: {min_length=}, {edge_lengths=}")
        if use_3d_rot:
            d = monai_transforms.Compose(self.get_3d_rotations(aug_keys))(d)
        return d

    def apply_padding(self, d, aug_keys):
        if self.args.divisable_by is not None:
            for key in aug_keys:
                # Each dimension needs to be divisable by the respective factor
                padsize = []
                for i, factor in enumerate(self.args.divisable_by):
                    # At this point, all images and masks are channels first
                    if d[key].shape[i + 1] % factor != 0:
                        padsize.append(make_divisable_by(d[key].shape[i + 1], factor))
                    else:
                        padsize.append(-1)  # Applies no padding

                d[key] = monai_transforms.SpatialPad(spatial_size=tuple(padsize))(d[key])
        return d

    def __call__(self, d: dict[str, Any]) -> dict[str, Any]:
        d, num_images, aug_keys = self.m3_to_monai(d)

        d = self.fix_orientation(d, aug_keys)

        if self.args.normalize:
            d = self.normalize(d, aug_keys)

        d = self.apply_padding(d, aug_keys)

        if self.augs:
            d = monai_transforms.Compose(self.augs(aug_keys))(d)

        d = self.rotate_if_applicable(d, aug_keys)

        return self.monai_to_m3(d, num_images, aug_keys)

    def extract_patches(
        self,
        d: dict[str, Any],
        min_max_patchedge: tuple[int, int],
        min_max_patchdepth: tuple[int, int],
        ratios: list[float],
        num_classes: int | None = None,
        num_patches: int | None = None,
    ) -> dict[str, Any]:
        """
        If no num_classes is given, ratios=[background, foreground] should be provided
        because all non-zero values in the mask are then treated equally as foreground.
        """
        d, num_images, aug_keys = self.m3_to_monai(d)

        d = self.fix_orientation(d, aug_keys)

        if self.args.normalize:
            d = self.normalize(d, aug_keys)

        d = self.apply_padding(d, aug_keys)

        if self.augs:
            d = monai_transforms.Compose(self.augs(aug_keys))(d)

        d = self.rotate_if_applicable(d, aug_keys)

        # Extract patches
        edgesize = random.randint(*min_max_patchedge)
        fullimage_shape = d[aug_keys[0]].shape  # C=1, H, W, D
        patchsize = (
            min(fullimage_shape[1], edgesize),
            min(fullimage_shape[2], edgesize),
            min(fullimage_shape[3], random.randint(*min_max_patchdepth)),
        )
        if num_classes is None:
            d["patchsample_mask"] = d["label"] > 0
        patches = MemFriendlyCrop(
            keys=aug_keys,
            label_key="label" if num_classes is not None else "patchsample_mask",
            spatial_size=patchsize,
            num_classes=num_classes if num_classes is not None else 2,
            ratios=ratios,
            num_samples=max(1, fullimage_shape[3] // (2 * patchsize[-1])) if num_patches is None else num_patches,
            allow_smaller=True,
        )(d)
        for i, patch in enumerate(patches):
            # if "patchsample_mask" in patch:
            #     del patch["patchsample_mask"]
            patch["meta"]["group_id"] = f"{patch['meta']['group_id']}_p{i}_{str(uuid.uuid4())[:4]}"
            patch["meta"]["patchsize"] = patchsize
            patch["meta"]["num_patches"] = len(patches)
        assert len(patches) < 2 or patches[0]["meta"]["group_id"] != patches[1]["meta"]["group_id"]
        return list(map(partial(self.monai_to_m3, num_images=num_images, aug_keys=aug_keys), patches))


class SometimesPatcher(BaseModel):
    patchify: float = Field(default=0.8, description="probability for patchification")
    min_max_patchedge: tuple[int, int] = (128, 384)
    min_max_patchdepth: tuple[int, int] = (1, 8)
    ratios: list[float] = [0.05, 0.95]

    def process_supercase(self, supercase, processor: Tomo3DProcessor, as_patch_list=False):
        if random.random() < self.patchify:
            patchlist = processor.extract_patches(
                supercase,
                min_max_patchedge=self.min_max_patchedge,
                min_max_patchdepth=self.min_max_patchdepth,
                ratios=self.ratios,
                num_classes=len(self.ratios) if len(self.ratios) > 2 else None,
            )
            res = [processor.extract_slices(patch) for patch in patchlist]
        else:
            res = [processor.extract_slices(processor(supercase))]

        if as_patch_list:
            return res
        else:
            return [x for patch in res for x in patch]  # flatten the list of lists


class SegMetric3D:
    class Config(BaseModel):
        metrics: list[Literal["dice", "nsd"]] = Field(default=["dice"], description="Metrics to compute")
        include_background: bool = Field(default=True, description="Whether to include first class in metric.")
        resize_to: tuple[int, int, int] | None = Field(
            default=None, description="If not None, resize y_true and y_pred to this shape before computing metric."
        )

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if self.cfg.resize_to is not None:
            self.resizer = Resize(spatial_size=self.cfg.resize_to, mode="nearest-exact")

    def _to_monai(self, y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
        # convert (B, H, W, D) to binarized (B, C, H, W, D)
        y_pred = torch.nn.functional.one_hot(y_pred, num_classes=num_classes).permute(0, 4, 1, 2, 3)
        y_true = torch.nn.functional.one_hot(y_true, num_classes=num_classes).permute(0, 4, 1, 2, 3)
        return y_pred, y_true

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int, *args, **kwargs):
        """
        For multi-class segmentation:

        y_pred shape: (B, H, W, D)
        y_true shape: (B, H, W, D)

        type should be LongTensor
        """
        assert y_pred.shape == y_true.shape, "y_pred and y_true should have same shape (B, H, W, D)"
        assert y_pred.dtype == torch.long, "y_pred should be LongTensor"
        assert y_true.dtype == torch.long, "y_true should be LongTensor"

        y_pred, y_true = self._to_monai(y_pred, y_true, num_classes)

        if self.cfg.resize_to is not None:
            # Each item of the batch needs to be resized independently
            y_pred = torch.stack([self.resizer(y) for y in y_pred])
            y_true = torch.stack([self.resizer(y) for y in y_true])

        res = {}
        if "dice" in self.cfg.metrics:
            # When setting num_classes, MONAI assumes single-channel class-indices
            # However, it still requires an empty channel to be there
            dices = compute_dice(
                y_pred, y_true, include_background=self.cfg.include_background, ignore_empty=False, *args, **kwargs
            )
            res["dice"] = dices.as_tensor() if isinstance(dices, MetaTensor) else dices
        if "nsd" in self.cfg.metrics:
            raise NotImplementedError("NSD metric is not implemented yet")

        return res


class Volume3DInference(BaseModel):
    """
    A volume prediction always has one mask, but can have multiple images.
    For example, an MRI with multiple sequences.
    """

    instances: Literal["none", "boxes", "instancemasks"] = "none"
    context: Literal["none", "aggregation", "positionalencoding"] = "positionalencoding"
    context_length: int = Field(default=24, description="How many slices to use for a single prediction.")
    tomo_processor: Tomo3DProcessor.Config = Tomo3DProcessor.Config(fix_orientation="never", rotate_3d="never")
    for_image: int = Field(default=0, description="Index of the image to use for the mask.")
    max_edge_length: int = Field(default=512, description="Maximum edge length of the spatial dimensions of the image.")
    num_workers: int = Field(default=0, description="Number of workers to use for data loading.")

    @staticmethod
    def load_volume(tomo_processor: Tomo3DProcessor.Config, image_paths: list[DistributedPath | Path], for_image=0):
        """
        Loads voxel data for the images in the format expected by the method invocation.
        """
        image_paths = [p.upath() if isinstance(p, DistributedPath) else p for p in image_paths]
        processor = Tomo3DProcessor(
            tomo_processor,
            None,
            with_segmask=False,
        )
        patient = {
            "image": [processor.image_loader(filename=p)[0].unsqueeze(0) for p in image_paths],
            "meta": {"group_id": "inference"},
        }
        processed_patient = processor(patient)
        logfire.info(
            "Loaded volume with shape {shape} for inference",
            shape=processed_patient["image"][for_image].shape,
        )
        slice_generator = map(
            ApplyToKey(processor.repeat_channels, key="image"), processor.extract_slices(processed_patient)
        )
        main_image_affine = processed_patient["meta"]["monai_meta"][f"image{for_image}"]["affine"]
        return slice_generator, processed_patient["image"][for_image].shape, main_image_affine  # C, H, W, DEPTH

    @torch.inference_mode()
    def __call__(self, model, task, slices, chwd: tuple, main_image_affine):
        import torchvision.transforms as T
        import torchvision.transforms.functional as F
        from torch.utils.data import DataLoader

        from mmm.mtl_modules.shared_blocks.Grouper import Grouper

        _, H, W, DEPTH = chwd
        unify_sizes = UnifySizes(max_edge_len=self.max_edge_length)
        mask_volume_out: torch.Tensor = torch.zeros((H, W, DEPTH), dtype=torch.int16).fill_(-1)
        if self.instances != "none":
            probas_volume_out: torch.Tensor = torch.zeros(
                (len(task.class_names), H, W, DEPTH), dtype=torch.float32
            ).fill_(torch.nan)
        num_images = max([x["meta"]["context"][1] for x in slices]) + 1

        with logfire.span(
            "Computing volume mask for {num_images} images and {num_slices} slices",
            num_images=num_images,
            num_slices=len(slices),
        ):
            for batch_reprs in DataLoader(
                slices,
                batch_size=num_images * self.context_length,
                shuffle=False,
                collate_fn=T.Compose(
                    [
                        unify_sizes,
                        mtl_collate,
                    ]
                ),
                num_workers=self.num_workers,
            ):
                # print("input-batch shape: ", batch_reprs["image"].shape)
                masks_logits = task.forward(
                    (
                        batch_reprs["image"].to(model.device),
                        Grouper.extract_ids_from_batch(["main_image" for x in range(len(batch_reprs["meta"]))]).to(
                            model.device
                        ),
                        ctx := [m["context"] for m in batch_reprs["meta"]],
                    ),
                    model,
                )
                # print(f"Mask logits shape: {masks_logits.shape}, Contexts length: {len(ctx)}")
                assert len(ctx) == masks_logits.shape[0], "Mismatch between contexts and logits batch size"
                for i, (ctx, logits) in enumerate(zip(ctx, masks_logits)):
                    if ctx[1] == self.for_image:
                        # assert mask_volume_out[:, :, ctx[0]].unique().tolist() == [-1]
                        # print(ctx[0], logits.shape)
                        if logits.shape[1:] != (H, W):
                            logits_resized = F.resize(
                                logits.unsqueeze(0),
                                size=(H, W),
                                interpolation=F.InterpolationMode.BILINEAR,
                            ).squeeze(0)
                        else:
                            logits_resized = logits
                        mask_volume_out[:, :, ctx[0]] = logits_resized.argmax(dim=0).cpu()
                        if self.instances != "none":
                            probas_volume_out[..., ctx[0]] = logits_resized.softmax(dim=0).cpu()

        res = {"mask": mask_volume_out, "image_affine": main_image_affine}

        if self.instances != "none":
            regions = find_regions(probas_volume_out)

        if self.instances == "boxes":
            res["boxes"] = []
            for region in regions:
                res["boxes"].append(
                    {
                        "box": region["region"].bbox,
                        "area": region["region"].area,
                        "class": task.class_names[max(region["region_scores"], key=region["region_scores"].get)],
                        "scores": region["region_scores"][1],
                    }
                )
        elif self.instances == "instancemasks":
            raise NotImplementedError("Instance masks are not implemented yet")

        return res
