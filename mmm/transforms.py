import itertools
import logging
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from pydantic import Field
from torchvision.models.detection.transform import resize_boxes

from mmm.BaseModel import BaseModel
from mmm.data_loading.geojson import GeoAnno, GeojsonRegionWindows
from mmm.data_loading.geojson.NoUsefulWindowException import NoUsefulWindowException
from mmm.logging.type_ext import TransformsSeqType
from mmm.utils import make_divisable_by


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


class MaskedPatchExtractor:
    """
    Takes in a large image with a mask and outputs a list of patches suitable for `CachingSubCaseDS`.

    Masks should not have a channel dimension: [H, W]

    It exists because MONAI cannot deal well with colorimages.

    To suppress patches with only the background class, it uses a region of interest around the foreground.
    Currently, foreground is defined as any non-zero value.

    Expects channels first images.
    """

    class Config(BaseModel):
        patch_sizes: List[int] = [224]
        sizeaugmentation: float = Field(
            default=0.1,
            description="Jiggle the H, W and coordinates of the patch by a maximum of this factor.",
        )
        max_patches: int | None = None

    def __init__(
        self,
        args: Config,
        patchfilter: Optional[Callable] = None,
        mask_key: str | None = "label",
        with_boxes: bool = False,
    ):
        self.args = args
        self.mask_key = mask_key
        self.with_boxes = with_boxes
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
        mask: torch.Tensor | None = d[self.mask_key] if self.mask_key is not None else None

        rect = GeoAnno.rectangle_builder((0, 0), (img.shape[-2], img.shape[-1]))

        pseudo_levels = {i: patchsize / self.min_patch_size for i, patchsize in enumerate(self.args.patch_sizes)}
        g = self.regionextractor.iter_valid_windows(rect, pseudo_levels)
        try:
            for i, (level, xy, hw) in enumerate(itertools.islice(g, self.args.max_patches)):
                hw_scaled = hw[0] * pseudo_levels[level], hw[1] * pseudo_levels[level]
                patchmeta = {"level": level, "xy": xy, "hw": hw_scaled, "i": i}

                x1 = max(0, int(xy[0]))
                x2 = min(img.shape[-2], int(xy[0] + hw_scaled[0]))
                y1 = max(0, int(xy[1]))
                y2 = min(img.shape[-1], int(xy[1] + hw_scaled[1]))
                res = {
                    "image": img[:, x1:x2, y1:y2],
                    "meta": {"patchmeta": patchmeta},
                }
                if mask is not None:
                    res[self.mask_key] = mask[..., x1:x2, y1:y2]
                if self.with_boxes:
                    if d["boxes"].shape[0] == 0:
                        res["boxes"] = d["boxes"]
                        res["labels"] = d["labels"]
                    else:
                        boxes_within_patch = []
                        labels_within_patch = []
                        for box_i in range(d["boxes"].shape[0]):
                            coords = d["boxes"][box_i]
                            box_xsize = coords[2] - coords[0]
                            box_ysize = coords[3] - coords[1]

                            # If at least half of the box is within the patch, keep it
                            if in_patch := (
                                False
                                not in [
                                    coords[0] >= y1 - (box_xsize * 0.5),
                                    coords[1] >= x1 - (box_ysize * 0.5),
                                    coords[2] <= y2 + (box_xsize * 0.5),
                                    coords[3] <= x2 + (box_ysize * 0.5),
                                ]
                            ):
                                boxes_within_patch.append(
                                    torch.tensor(
                                        [
                                            max(coords[0] - y1, 0),
                                            max(coords[1] - x1, 0),
                                            min(coords[2] - y1, y2 - y1),
                                            min(coords[3] - x1, x2 - x1),
                                        ],
                                        dtype=torch.float32,
                                    )
                                )
                                labels_within_patch.append(d["labels"][box_i])
                        res["boxes"] = (
                            torch.stack(boxes_within_patch)
                            if len(boxes_within_patch) > 0
                            else torch.empty((0, 4), dtype=torch.float32)
                        )
                        res["labels"] = (
                            torch.tensor(labels_within_patch)
                            if len(labels_within_patch) > 0
                            else torch.empty((0,), dtype=torch.int64)
                        )
                if "meta" in d:
                    res["meta"]["supermeta"] = d["meta"]
                    if "group_id" in d["meta"]:
                        res["meta"]["group_id"] = d["meta"]["group_id"]
                if self.patchfilter is None or self.patchfilter(res):
                    yield res
        except NoUsefulWindowException as e:
            # If the image is smaller than the smallest patch size, just yield the image
            # if img.shape[-2] < self.min_patch_size or img.shape[-1] < self.min_patch_size:
            res = {
                "image": img,
                "meta": {
                    "patchmeta": {"level": 0, "xy": (0, 0), "hw": (img.shape[-2], img.shape[-1]), "i": 0},
                },
            }
            if self.mask_key is not None:
                res[self.mask_key] = mask
            if self.with_boxes:
                res["boxes"] = d["boxes"]
                res["labels"] = d["labels"]
            if "meta" in d:
                res["meta"]["supermeta"] = d["meta"]
                if "group_id" in d["meta"]:
                    res["meta"]["group_id"] = d["meta"]["group_id"]
            if self.patchfilter is None or self.patchfilter(res):
                yield res


def flatten_list(lst: List[List[Any]]) -> List[Any]:
    return [item for sublist in lst for item in sublist]


def _case_to_aformat(
    d: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[float]], Optional[List[int]]]:
    aimg = (d["image"] * 255.0).permute(1, 2, 0).numpy().astype(np.uint8)
    amask = d["label"].numpy() if "label" in d else None
    amasks = [d["masks"].permute(1, 2, 0).numpy()] if "masks" in d else None
    aboxes = d["boxes"].type(torch.LongTensor).tolist() if "boxes" in d else None
    alabels = d["labels"].tolist() if "labels" in d else None

    if aboxes is not None and alabels is not None:
        invalid_box = [b[0] >= b[2] or b[1] >= b[3] for b in aboxes]
        if True in invalid_box:
            logging.debug(f"Invalid box found in {d['meta']}")
            aboxes = [b for i, b in enumerate(aboxes) if not invalid_box[i]]
            alabels = [l for i, l in enumerate(alabels) if not invalid_box[i]]
    return aimg, amask, amasks, aboxes, alabels


def _atransform_into_case(transformed: dict[str, Any]) -> dict[str, Any]:
    res = {"image": F.to_tensor(transformed["image"])}  # torch.from_numpy(transformed['image']).permute(2, 0, 1),
    if "bboxes" in transformed:
        res["boxes"] = torch.Tensor(transformed["bboxes"])
        res["labels"] = torch.Tensor(transformed["class_labels"]).long()
    if "mask" in transformed:
        res["label"] = torch.from_numpy(transformed["mask"]).long()
    if "masks" in transformed:
        res["masks"] = torch.from_numpy(transformed["masks"][0].copy()).permute(2, 0, 1).long()
    return res


class Alb:
    """
    We use pascal_voc format for boxes. Augmentation might change the number of boxes and labels.
    """

    def __init__(self, transforms: TransformsSeqType, support_boxes=False, replay_for_groups=False) -> None:
        self.replay_for_groups = replay_for_groups
        box_kwargs = (
            {"bbox_params": A.BboxParams(format="pascal_voc", label_fields=["class_labels"])} if support_boxes else {}
        )
        if self.replay_for_groups:
            self.transform = A.ReplayCompose(
                transforms,
                **box_kwargs,
            )
            self._group_id = None
            self._replaydata = None
        else:
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

        if self.replay_for_groups:
            group_id = d["meta"]["group_id"] if "meta" in d and "group_id" in d["meta"] else None
            if self._replaydata is None or group_id is None or group_id != self._group_id:
                transformed = self.transform(**transform_kwargs)

                # See if some replay data should be stored
                if group_id is not None:
                    if self._group_id is None or self._group_id != d["meta"]["group_id"]:
                        self._group_id = d["meta"]["group_id"]
                        self._replaydata = transformed["replay"]
            else:
                try:
                    transformed = A.ReplayCompose.replay(self._replaydata, **transform_kwargs)
                except Exception as e:
                    logging.warning(f"Replay failed for {d['meta']} with {e}")
                    transformed = self.transform(**transform_kwargs)
        else:
            transformed = self.transform(**transform_kwargs)

        transformed_case = _atransform_into_case(transformed)
        d.update(transformed_case)

        return d


class AlbWithBoxes(Alb):
    def __init__(self, transforms: TransformsSeqType, **kwargs) -> None:
        super().__init__(transforms, support_boxes=True, **kwargs)


class UnifySizes:
    """
    Can be used before a collate_fn in dataloaders to make sure that all images have the same size.

    The size will be divisable by divisable_by to accomodate network constraints.
    """

    def __init__(
        self,
        divisable_by=32,
        max_edge_len=None,
        support_boxes=False,
    ) -> None:
        self.divisable_by = divisable_by
        self.max_edge_len = max_edge_len
        self.support_boxes = support_boxes
        if max_edge_len is not None:
            assert (
                max_edge_len % divisable_by == 0
            ), f"max_edge_len {max_edge_len} should be divisable by {divisable_by}"

    @staticmethod
    def add_size_to_meta(d: Dict[str, Any]) -> Dict[str, Any]:
        if "meta" not in d:
            d["meta"] = {}

        if "original_image_size" not in d["meta"]:  # Store the size before the first resize
            # If the image instance is already in a format suitable for Albumentations, the channel dim is first
            d["meta"]["original_image_size"] = d["image"].shape[1:]
        d["meta"]["image_size_before_resize_case"] = d["image"].shape[1:]  # Store the most recent image size

        return d

    @staticmethod
    def resize_case(d: Dict[str, Any], resizer) -> Dict[str, Any]:
        d = UnifySizes.add_size_to_meta(d)

        return resizer(d)

    @staticmethod
    def get_divisable_dims(img: torch.Tensor, divisable_by: int, max_edge: int | None) -> Tuple[int, int]:
        height = img.shape[1]
        width = img.shape[2]

        if max_edge is not None and max(height, width) > max_edge:
            scale = max_edge / max(height, width)
            height = int(height * scale)
            width = int(width * scale)

        width_divisable = make_divisable_by(width, by=divisable_by)
        height_divisable = make_divisable_by(height, by=divisable_by)

        return height_divisable, width_divisable

    def __call__(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Pick one random element from batch to determine the size to which all images will be resized.
        sizegiver = random.choice(batch)
        height_divisable, width_divisable = self.get_divisable_dims(
            sizegiver["image"], self.divisable_by, self.max_edge_len
        )
        resizer = Alb(transforms=[A.Resize(height_divisable, width_divisable)], support_boxes=self.support_boxes)
        # Resize all images in the batch to the same size
        return [self.resize_case(d, resizer) for d in batch]
