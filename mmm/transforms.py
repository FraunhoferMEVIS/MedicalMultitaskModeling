from enum import Enum
from pydantic import Field
import random
from functools import reduce
import numpy as np
import torch
import logging
from typing import Any, Callable, List, Tuple, Dict, Optional, Set, Iterable

import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.models.detection.transform import resize_boxes
from monai.data.utils import get_random_patch
from monai.transforms.utils import generate_spatial_bounding_box
import albumentations as A

from mmm.BaseModel import BaseModel
from mmm.utils import make_divisable_by
from mmm.data_loading.geojson import GeojsonRegionWindows, GeoAnno
from mmm.logging.type_ext import TransformsSeqType


class TupleToDict:
    def __init__(self, key_names: Optional[Tuple[str, ...]] = None) -> None:
        self.key_names = key_names

    def __call__(self, x: Tuple) -> Dict:
        if self.key_names is not None:
            assert len(x) == len(self.key_names)
            return {k: v for k, v in zip(self.key_names, x)}
        return {str(i): v for i, v in enumerate(x)}


class SaveResultAsKey:
    def __init__(self, key_name: str, f: Callable[[Dict], Any]) -> None:
        self.key_name = key_name
        self.f = f

    def __call__(self, d: Dict) -> Any:
        assert self.key_name not in d, f"{self.key_name} already exists in {d}"
        d[self.key_name] = self.f(d)
        return d


class ApplyToKey:
    def __init__(self, f: Callable, key: str):
        self.f, self.key = f, key

    def __call__(self, d: Dict) -> Dict:
        d[self.key] = self.f(d[self.key])
        return d


class RandomApply:
    def __init__(self, f: Callable, p: float):
        self.f, self.p = f, p

    def __call__(self, d: Dict) -> Dict:
        if random.random() < self.p:
            d = self.f(d)
        return d


class ApplyToKeys:
    """
    >>> dictionary = {"key1": 1, "key2": 2}
    >>> def swap(a,b): return b, a
    >>> from mmm.transforms import ApplyToKeys
    >>> transform = ApplyToKeys(swap, ["key1", "key2"])
    >>> result = transform(dictionary)
    >>> result
    {'key1': 2, 'key2': 1}
    """

    def __init__(self, f: Callable, keys: List[str]):
        self.f, self.keys = f, keys

    def __call__(self, d: Dict) -> Dict:
        # Provide function with arguments corresponding to the dictionary keys
        tuple_result = self.f(*[d[key] for key in self.keys])

        # Play results of function back into dictionary
        for i, key in enumerate(self.keys):
            d[key] = tuple_result[i]

        return d


class ApplyToList:
    """
    Applies a function to a list
    """

    def __init__(self, f: Callable):
        self.f = f

    def __call__(self, ls):
        return [self.f(x) for x in ls]


class ExtractKeysAsTuple:
    def __init__(self, keys: Tuple):
        self.keys = keys

    def __call__(self, d: Dict) -> Tuple:
        return tuple([d[k] for k in self.keys])


class CopyKeysInDict:
    def __init__(self, keys: Dict[str, str]) -> None:
        self.keys = keys

    def __call__(self, d: Dict) -> Dict:
        for key, copy_key in self.keys.items():
            assert copy_key not in d, f"{copy_key} already exists on case {d}"
            d[copy_key] = d[key]
        return d


class KeepOnlyKeysInDict:
    """
    Takes a dictionary and returns that same dictionary but deletes all non-desired elements.

    >>> import numpy as np
    >>> from mmm.transforms import KeepOnlyKeysInDict
    >>> somedictionary = {"keepkey": np.array([1, 1]), "deletethiskey": np.array([2, 2])}
    >>> somedictionary
    {'keepkey': array([1, 1]), 'deletethiskey': array([2, 2])}
    >>> KeepOnlyKeysInDict(keys={"keepkey"})(somedictionary)
    {'keepkey': array([1, 1])}

    If you also want to rename some of the remaining keys, use rename_keys.
    >>> KeepOnlyKeysInDict(keys={"keepkey"}, rename_keys={"keepkey": "newkey"})(somedictionary)
    {'newkey': array([1, 1])}
    """

    def __init__(self, keys: Set[str], rename_keys: Optional[Dict[str, str]] = None) -> None:
        self.keys: Set[str] = keys
        self.rename_dict = rename_keys

    def __call__(self, d: Dict) -> Dict:
        # d changes size during iteration, so copy the iterator with the keys to a list first
        for key in list(d.keys()):
            if key not in self.keys:
                d.pop(key)

        if self.rename_dict is not None:
            for k, v in self.rename_dict.items():
                d[v] = d.pop(k)

        return d


def batchify(ls: List[np.ndarray], warn_when_empty=True):
    """
    Adds an empty batch-dimension to every item in the list and stacks them.
    """
    if ls:
        return np.concatenate([np.expand_dims(x, axis=0) for x in ls], axis=0)
    else:
        if warn_when_empty:
            logging.warning(f"encountered empty object in `batchify`: {ls}")
        return np.array([], np.float32)


class ChannelsSwapFirstLast:
    def __init__(self, keys: Tuple[str]) -> None:
        self.keys = keys

    def __call__(self, d: Dict[str, Any]) -> Any:
        for key in self.keys:
            d[key] = torch.moveaxis(d[key], -1, 0)

        return d


class ResizeWithMask:
    def __init__(self, new_size: Tuple[int, int]) -> None:
        self.img_resizer = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.mask_resizer = transforms.Compose(
            [
                lambda x: torch.unsqueeze(x, 0),
                transforms.Resize(new_size, interpolation=transforms.InterpolationMode.NEAREST),
                lambda x: x[0],
            ]
        )

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Any:
        assert len(image.shape) - 1 == len(mask.shape), "Images should have one more dimension than masks"
        return self.img_resizer(image), self.mask_resizer(mask)


class ResizeImage:
    def __init__(self, new_size: List[int], original_size_storage_key: str = "original_size") -> None:
        self.img_resizer = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.original_size_storage_key = original_size_storage_key

    def __call__(self, d: Dict[str, Any]) -> Any:
        if self.original_size_storage_key not in d:
            d[self.original_size_storage_key] = list(d["image"].shape[1:])
        else:
            assert (
                len(d["image"].shape) == len(d[self.original_size_storage_key].shape) - 1
            ), "Original shape should only consist of spatial dimensions"

        d["image"] = self.img_resizer(d["image"])
        return d


class ResizeBoxes:
    def __init__(self, overwrite_original_size=None, overwrite_new_size=None) -> None:
        self.overwrite_original_size = overwrite_original_size
        self.overwrite_new_size = overwrite_new_size

    def __call__(self, d: Dict[str, Any]) -> Any:
        if self.overwrite_original_size is None:
            assert "original_size" in d, "Original size of image needs to be known for resizing boxes"
            original_size = d["original_size"]
        else:
            original_size = self.overwrite_original_size

        new_size = d["image"].shape[1:] if self.overwrite_new_size is None else self.overwrite_new_size

        d["boxes"] = resize_boxes(torch.Tensor(d["boxes"]), original_size, new_size)

        return d


class ExtractMaskedPatches:
    """
    Takes in a large image with a mask and outputs a list of subcases.

    Forms a rectangular bounding box from foreground and extracts patches within this bounding box.
    If you have very large masks where a bounding box would introduce a large false positive area,
    consider using the GeoJSON utilities for region proposal and extract masks from that.

    For color images channels last needs to be used.

    Masks should not have a channel dimension:
    - 2D: [width, height]
    - 3D: [width, height, slices]

    It exists because MONAI cannot deal well with colorimages.
    """

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (224, 224),
        image_key: str = "image",
        label_key: str = "label",
        out_image_key: str = "image",
        out_label_key: str = "label",
        max_coverage_foreground: float = 1.0,
        channels_first: bool = True,
        max_patches: Optional[int] = None,
        patch_transform: Optional[Callable] = None,
    ) -> None:
        self.patch_size = patch_size
        self.image_key, self.label_key = image_key, label_key
        self.out_image_key, self.out_label_key = out_image_key, out_label_key
        self.channels_first = channels_first
        self.max_coverage_foreground = max_coverage_foreground
        assert self.channels_first, "Currently, does only support channels_first images and masks without channel-dim."
        self.max_patches = max_patches
        self.patch_transform = patch_transform
        logging.warning("Deprecated, use `MaskedPatchExtractor`")

    def __call__(self, d: Dict) -> Any:
        img: torch.Tensor = d[self.image_key]
        mask: torch.Tensor = d[self.label_key]

        patches = []
        patch_masks = []
        patch_coords = []

        def gen_one_patch():
            pos_with_roi = tuple([roi[1][i] - roi[0][i] for i in range(len(roi[0]))])
            slices = get_random_patch(pos_with_roi, patch_size=self.patch_size)
            slices_with_roi = [slice(s.start + roi[0][i], s.stop + roi[0][i]) for i, s in enumerate(slices)]
            img_slices = [slice(img.shape[0])] + slices_with_roi if self.channels_first else slices_with_roi
            return img[img_slices], mask[slices_with_roi], slices_with_roi

        roi = generate_spatial_bounding_box(mask.unsqueeze(0), margin=[x // 2 for x in self.patch_size])
        roi_area: int = reduce(lambda x, y: x * y, [roi[1][i] - roi[0][i] for i in range(len(roi[0]))])
        patch_area = reduce(lambda x, y: x * y, self.patch_size)
        for _ in range(roi_area // patch_area):
            p_img, p_mask, p_coords = gen_one_patch()
            if self.patch_transform is not None:
                p_img, p_mask = self.patch_transform(p_img, p_mask)

            patches.append(p_img)
            patch_masks.append(p_mask)
            patch_coords.append(p_coords)

            if (self.max_patches is not None) and (len(patches) >= self.max_patches):
                break

        if not patch_coords:
            d[self.out_image_key] = torch.Tensor()
            d["meta"] = []
            d[self.out_label_key] = torch.Tensor()
        else:
            d[self.out_image_key] = torch.stack(patches)
            if "meta" not in d:
                d["meta"] = [{} for _ in patch_coords]
            else:
                d["meta"] = [d["meta"].copy() for _ in patch_coords]
            for i, patch_coord in enumerate(patch_coords):
                d["meta"][i][f"{self.out_image_key}_subindex"] = i
                d["meta"][i][f"{self.out_image_key}_subcoord"] = patch_coord
            d[self.out_label_key] = torch.stack(patch_masks)

        return d


class MaskedPatchExtractor:
    """
    Takes in a large image with a mask and outputs a list of patches suitable for `CachingSubCaseDS`.

    Masks should not have a channel dimension: [width, height]

    It exists because MONAI cannot deal well with colorimages.

    To suppress patches with only the background class, it uses a region of interest around the foreground.
    Currently, foreground is defined as any non-zero value.

    Expects channels first images.
    """

    class Config(BaseModel):
        patch_sizes: List[int] = [224]
        sizeaugmentation: float = Field(
            default=0.1,
            description="Jiggle the width, height and coordinates of the patch by a maximum of this factor.",
        )
        max_patches: Optional[int] = None

    def __init__(
        self,
        args: Config,
        patchfilter: Optional[Callable] = None,
        mask_key: str = "label",
    ):
        self.args = args
        self.mask_key = mask_key
        self.patchfilter = patchfilter

        self.min_patch_size = min(args.patch_sizes)
        self.regionextractor = GeojsonRegionWindows(
            coordinate_augmentation="random",
            windowsize_augmentation="relative",
            augmentation_strength=args.sizeaugmentation,
            stepsize=2,
            patch_size=(self.min_patch_size, self.min_patch_size),
        )

    def apply(self, d: Dict) -> Iterable[Dict[str, Any]]:
        img: torch.Tensor = d["image"]
        mask: torch.Tensor = d[self.mask_key]

        rect = GeoAnno.rectangle_builder((0, 0), (mask.shape[-2], mask.shape[-1]))

        pseudo_levels = {i: patchsize / self.min_patch_size for i, patchsize in enumerate(self.args.patch_sizes)}
        g = self.regionextractor.iter_valid_windows(rect, pseudo_levels)
        for i, (level, xy, wh) in enumerate(g):
            wh_scaled = wh[0] * pseudo_levels[level], wh[1] * pseudo_levels[level]
            patchmeta = {"level": level, "xy": xy, "wh": wh_scaled, "i": i}

            x1 = max(0, int(xy[0]))
            x2 = min(mask.shape[-2], int(xy[0] + wh_scaled[0]))
            y1 = max(0, int(xy[1]))
            y2 = min(mask.shape[-1], int(xy[1] + wh_scaled[1]))
            res = {
                "image": img[:, x1:x2, y1:y2],
                self.mask_key: mask[..., x1:x2, y1:y2],
                "meta": {"patchmeta": patchmeta},
            }
            if "meta" in d:
                res["meta"]["supermeta"] = d["meta"]
            if self.patchfilter is None or self.patchfilter(res):
                yield res


def flatten_list(lst: List[List[Any]]) -> List[Any]:
    return [item for sublist in lst for item in sublist]


def _case_to_aformat(
    d: Dict[str, Any]
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[float]], Optional[List[int]]]:
    aimg = (d["image"] * 255.0).permute(1, 2, 0).numpy().astype(np.uint8)
    amask = d["label"].numpy() if "label" in d else None
    amasks = [d["masks"].permute(1, 2, 0).numpy()] if "masks" in d else None
    aboxes = d["boxes"].type(torch.LongTensor).tolist() if "boxes" in d else None
    alabels = d["labels"].tolist() if "labels" in d else None

    if aboxes is not None and alabels is not None:
        invalid_box = [b[0] >= b[2] or b[1] >= b[3] for b in aboxes]
        if True in invalid_box:
            logging.warning(f"Invalid box found in {d['meta']}")
            aboxes = [b for i, b in enumerate(aboxes) if not invalid_box[i]]
            alabels = [l for i, l in enumerate(alabels) if not invalid_box[i]]
    return aimg, amask, amasks, aboxes, alabels


def _atransform_into_case(transformed: Dict[str, Any]) -> Dict[str, Any]:
    res = {"image": F.to_tensor(transformed["image"])}  # torch.from_numpy(transformed['image']).permute(2, 0, 1),
    if "bboxes" in transformed:
        res["boxes"] = torch.Tensor(transformed["bboxes"])
        res["labels"] = torch.Tensor(transformed["class_labels"]).long()
    if "mask" in transformed:
        res["label"] = torch.from_numpy(transformed["mask"]).long()
    if "masks" in transformed:
        res["masks"] = torch.from_numpy(transformed["masks"][0]).permute(2, 0, 1).long()
    return res


class Alb:
    def __init__(self, transforms: TransformsSeqType, support_boxes=False) -> None:
        box_kwargs = (
            {"bbox_params": A.BboxParams(format="pascal_voc", label_fields=["class_labels"])} if support_boxes else {}
        )
        self.transform = A.Compose(
            transforms,
            **box_kwargs,
        )

    def __call__(self, d: Dict[str, Any]) -> Dict[str, Any]:
        aimg, amask, amasks, aboxes, alabels = _case_to_aformat(d)

        transform_kwargs: Dict[str, Any] = dict(image=aimg)
        for kwarg_name, kwarg_value in [
            ("mask", amask),
            ("masks", amasks),
            ("bboxes", aboxes),
            ("class_labels", alabels),
        ]:
            if kwarg_value is not None:
                transform_kwargs[kwarg_name] = kwarg_value
        transformed = self.transform(**transform_kwargs)

        transformed_case = _atransform_into_case(transformed)
        d.update(transformed_case)
        return d


class AlbWithBoxes(Alb):
    def __init__(self, transforms: TransformsSeqType) -> None:
        super().__init__(transforms, support_boxes=True)


class UnifySizes:
    """
    Can be used before a collate_fn in dataloaders to make sure that all images have the same size.

    The size will be divisable by divisable_by to accomodate network constraints.

    The dynamic batch size can be tuned by the max_pixels_in_batch parameter.
    For example, if you think that you can fit a batch size of 96 224x224 images into your GPU memory,
    you can set max_pixels_in_batch to 96*224*224.

    THIS WILL CHANGE THE ORDER INSIDE THE BATCH!
    """

    def __init__(
        self,
        divisable_by=32,
        max_pixels_in_batch=None,
        max_edge_len=None,
        support_boxes=False,
        enforce_order=False,
    ) -> None:
        self.divisable_by = divisable_by
        self.max_pixels_in_batch, self.max_edge_len = max_pixels_in_batch, max_edge_len
        self.support_boxes = support_boxes
        self.enforce_order = enforce_order

    @staticmethod
    def resize_case(d: Dict[str, Any], new_width: int, new_height: int, support_boxes: bool = False) -> Dict[str, Any]:
        return Alb(transforms=[A.Resize(new_height, new_width)], support_boxes=support_boxes)(d)

    @staticmethod
    def get_divisable_dims(img: torch.Tensor, divisable_by: int) -> Tuple[int, int]:
        height = img.shape[1]
        width = img.shape[2]

        width_divisable = make_divisable_by(width, by=divisable_by)
        height_divisable = make_divisable_by(height, by=divisable_by)

        return height_divisable, width_divisable

    def __call__(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Pick one random element from batch
        sizegiver = random.choice(batch)

        # If the max edge length is not none, first resize the sizegiver keeping its aspect ratio
        if self.max_edge_len is not None:
            if sizegiver["image"].shape[1] > self.max_edge_len or sizegiver["image"].shape[2] > self.max_edge_len:
                # Resizing only needs to happen if the image is larger than the max_edge_len
                sizegiver = Alb(
                    transforms=[A.LongestMaxSize(max_size=self.max_edge_len, always_apply=True)],
                    support_boxes=self.support_boxes,
                )(sizegiver)

        height_divisable, width_divisable = self.get_divisable_dims(sizegiver["image"], self.divisable_by)

        # The batch size might need to be adapted if the image size is changed
        if self.max_pixels_in_batch is not None:
            new_batch_size = self.max_pixels_in_batch // (width_divisable * height_divisable)
            if new_batch_size < len(batch):
                logging.warning(f"UnifySizes had to reduce batch size from {len(batch)} to {new_batch_size}!")
        else:
            new_batch_size = len(batch)

        # Resize all images in the batch to the same size
        filtered_batch = (
            [
                self.resize_case(
                    sizegiver,
                    width_divisable,
                    height_divisable,
                    support_boxes=self.support_boxes,
                )
            ]
            if not self.enforce_order
            else []
        )
        for d in batch:
            # sizegiver is already in the batch
            if (d is sizegiver) and (not self.enforce_order):
                continue

            if len(filtered_batch) < new_batch_size:
                filtered_batch.append(
                    self.resize_case(
                        d,
                        width_divisable,
                        height_divisable,
                        support_boxes=self.support_boxes,
                    )
                )
            else:
                break

        return filtered_batch
