import logging
import cv2
import numpy as np
from typing import Dict, Any, List, TypeVar, Tuple, Optional, Union, Callable
from PIL.Image import Image
import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
from monai.data.meta_tensor import MetaTensor
import monai.transforms as monai_transforms
import albumentations as A
import torchvision.transforms.functional as F
from mmm.BaseModel import BaseModel
from mmm.transforms import RandomApply
from mmm.logging.type_ext import TransformsSeqType
from mmm.settings import mtl_settings

# Typevariable for either a PIL image or a Tensor, indicates that the augmentation does not change the type
ImageType = TypeVar("ImageType", Image, torch.Tensor)

# Taken directly from albumentations.composition (but redefined since it is not exported there)


def get_histo_augs(
    img_fill_value=(255, 255, 255), mask_fill_value=mtl_settings.ignore_class_value
) -> TransformsSeqType:
    return [
        A.OneOf(  # Color variation (OneOf always executes exactly one if it is chosen)
            [
                # Included for positive transfer to grayscale images via multi-task learning
                A.ToGray(p=0.2),
                A.ChannelShuffle(p=0.2),
                # Should approximate typical color variation in stains
                A.HueSaturationValue(
                    hue_shift_limit=30, sat_shift_limit=20, val_shift_limit=20, always_apply=False, p=0.7
                ),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.RandomGamma(p=1),
                A.GaussNoise(),
            ],
            p=0.3,
        ),
        A.OneOf(
            [
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ],
            p=0.3,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=img_fill_value,
            mask_value=mask_fill_value,
        ),
    ]


def get_artefact_histo_augs():
    return [
        RandomBlobAugmentation(),
        RandomLineAugmentation(),
        RandomBlobAugmentation(),
    ]


def get_weak_default_augs(img_fill_value=0, mask_fill_value=mtl_settings.ignore_class_value) -> TransformsSeqType:
    """
    Designed to be compatible with as many as possible intensity-based images.
    Does not flip.
    """
    return [
        A.OneOf(
            [
                A.RandomGamma(p=1),
                A.GaussNoise(),
            ],
            p=0.1,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.8,
            border_mode=cv2.BORDER_CONSTANT,
            value=img_fill_value,
            mask_value=mask_fill_value,
        ),
        A.OneOf(
            [
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                # This has a bug, it swaps all non-zero labels in the mask to one :O
                # A.PiecewiseAffine(p=0.3),
            ],
            p=0.1,
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ],
            p=0.2,
        ),
        A.HueSaturationValue(p=0.2),
    ]


def get_realworld_augs() -> TransformsSeqType:
    return [
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.ToGray(p=0.2),
        A.OneOf(
            [
                # A.IAAAdditiveGaussianNoise(),
                A.RandomGamma(p=1),
                A.GaussNoise(),
            ],
            p=0.2,
        ),
        A.OneOf(
            [
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ],
            p=0.2,
        ),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf(
            [
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                # This has a bug, it swaps all non-zero labels in the mask to one :O
                # A.PiecewiseAffine(p=0.3),
            ],
            p=0.2,
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ],
            p=0.3,
        ),
        A.HueSaturationValue(p=0.3),
    ]


def get_xray_augs(img_fill_value=0, mask_fill_value=mtl_settings.ignore_class_value) -> TransformsSeqType:
    return [
        A.InvertImg(p=0.3),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ],
            p=0.3,
        ),
        A.OneOf(
            [
                # A.IAAAdditiveGaussianNoise(),
                A.RandomGamma(p=1),
                A.GaussNoise(),
            ],
            p=0.2,
        ),
        A.OneOf(
            [
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ],
            p=0.2,
        ),
        # A.OneOf([
        #     A.OpticalDistortion(p=0.3),
        #     A.GridDistortion(p=.1),
        #     A.PiecewiseAffine(p=0.3),
        # ], p=0.2),
        A.ShiftScaleRotate(
            p=1.0,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=img_fill_value,
            mask_value=mask_fill_value,
        ),
    ]


def get_mri2d_augs() -> TransformsSeqType:
    return [
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf(
            [
                A.RandomGamma(p=1),
                A.GaussNoise(),
            ],
            p=0.2,
        ),
        A.OneOf(
            [
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ],
            p=0.2,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.2,
            rotate_limit=45,
            p=0.2,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.OneOf(
            [
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                # This has a bug, it swaps all non-zero labels in the mask to one :O
                # A.PiecewiseAffine(p=0.3),
            ],
            p=0.2,
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ],
            p=0.3,
        ),
        A.HueSaturationValue(p=0.3),
    ]


def get_contrastive_2D_augs():
    """
    Contrastive augmentations based on https://arxiv.org/pdf/2105.04906.pdf
    """
    return [
        transforms.RandomResizedCrop(224, scale=(0.08, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomSolarize(p=0.1, threshold=), # left our bcs I don't know about the threshold yes
        transforms.RandomApply([transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]),
    ]


def volume_scale_to_0_1(img: torch.Tensor):
    """
    Scales the input array to the range [0, 1].
    """
    # Make sure extreme outliers do not influence the scaling
    return monai_transforms.ScaleIntensityRangePercentiles(lower=2.5, upper=97.5, b_min=0.0, b_max=1.0, clip=True)(img)


class MRI3DProcessor:
    class Config(BaseModel):
        normalize: bool = True
        rotate_3d: bool = False
        augment_intensity: bool = False

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
    def base_volume_augs(keys):
        return [
            monai_transforms.RandAdjustContrastD(keys=["image"], prob=0.1),
            RandomApply(
                monai_transforms.RandScaleCropD(keys=keys, roi_scale=[0.8, 0.8, 0.8], random_size=True),
                p=0.3,
            ),
        ]

    def __init__(self, args, augs_constructor: Optional[Callable], with_segmask: bool) -> None:
        self.args = args
        if augs_constructor:
            self.augs = monai_transforms.Compose(
                augs_constructor(["image", "label"] if with_segmask else ["image"])
                # self.get_recommended_mri_augs(keys=["image", "label"] if with_segmask else ["image"])
            )
        else:
            self.augs = None

    def __call__(self, d: Dict[str, Any]) -> Dict[str, Any]:
        assert "image" in d and len(d["image"].shape) >= 4, "C, spatials... dimensions expected"
        if "label" in d:
            assert len(d["image"].shape) == len(d["label"].shape) + 1

            # Monai expects a channel dimension which is always empty for multiclass problems
            d["label"] = torch.unsqueeze(d["label"], 0)

        if self.args.normalize:
            d["image"] = volume_scale_to_0_1(d["image"])
        if self.augs is not None:
            d = self.augs(d)  # type: ignore

        # Get rid of the MONAI meta tensor
        if isinstance(d["image"], MetaTensor):
            d["image"] = d["image"].as_tensor()

        if "label" in d:
            if isinstance(d["label"], MetaTensor):
                d["label"] = d["label"].as_tensor()
            d["label"] = torch.squeeze(d["label"])
        return d


def _case_to_aformat(
    d: Dict[str, Any]
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[float]], Optional[List[int]]]:
    aimg = (d["image"] * 255.0).permute(1, 2, 0).numpy().astype(np.uint8)
    amask = d["label"].numpy() if "label" in d else None
    aboxes = d["boxes"].type(torch.LongTensor).tolist() if "boxes" in d else None
    alabels = d["labels"].tolist() if "labels" in d else None
    return aimg, amask, aboxes, alabels


def _atransform_into_case(transformed: Dict[str, Any]) -> Dict[str, Any]:
    res = {"image": F.to_tensor(transformed["image"])}  # torch.from_numpy(transformed['image']).permute(2, 0, 1),
    if "bboxes" in transformed:
        res["boxes"] = torch.Tensor(transformed["bboxes"])
        res["labels"] = torch.Tensor(transformed["class_labels"]).long()
    if "mask" in transformed:
        res["label"] = torch.from_numpy(transformed["mask"]).long()
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
        aimg, amask, aboxes, alabels = _case_to_aformat(d)

        transform_kwargs: Dict[str, Any] = dict(image=aimg)
        for kwarg_name, kwarg_value in [
            ("mask", amask),
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


class PILHistoPatchAug:
    def __init__(self):
        self.f = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.75, saturation=0.25, hue=0.04),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.01, 0.01),
                    scale=(0.9, 1.3),
                    shear=(-0.1, 0.1),
                    fill=255,
                ),
            ]
        )

    def __call__(self, im: ImageType) -> ImageType:
        return self.f(im)


class SimCLRPatchAug:
    def __init__(self):
        self.cl_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 0.2))],
                    p=0.5,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(contrast=0.8, brightness=0.8, saturation=0.8, hue=0.2)],
                    p=0.8,
                ),
                transforms.RandomAffine(
                    degrees=90,
                    translate=(0.01, 0.01),
                    scale=(0.9, 1.3),
                    shear=(-0.1, 0.1),
                    fill=255,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, img: ImageType) -> ImageType:
        return self.cl_transform(img)


class RandomLineAugmentation:
    def __init__(self, p=0.7) -> None:
        self.p = p

    def __call__(self, image: torch.Tensor):
        """
        Expect a normlized RGB Torch Tensor
        """
        local_p = torch.rand(1).item()
        if local_p > self.p:
            return image
        else:
            data_c, data_x, data_y = image.squeeze().shape
            # generate random points (without replacement)
            rnd_x = torch.randperm(data_x)[:2]
            rnd_y = torch.randperm(data_y)[:2]

            # transfrom into cv2 readable landmarks
            pt1 = (rnd_x[0].item(), rnd_y[0].item())
            pt2 = (rnd_x[1].item(), rnd_y[1].item())

            # create artifact mask
            artifacts = torch.ones((data_x, data_y, 1), dtype=torch.uint8).numpy() * 255
            line_mask = cv.line(artifacts, pt1, pt2, color=(0, 0, 0), thickness=data_x // 32) / 255

            rnd_blurr_divider = torch.randint(15, 20, size=(1,)).item()
            # (data_x//rnd_blurr_divider, data_y//rnd_blurr_divider))
            blurred_mask = cv.blur(line_mask, (data_x // rnd_blurr_divider, data_y // rnd_blurr_divider))

            # multiply input img with mask to create artifact
            return image * blurred_mask  # , blurred_mask


class RandomBlobAugmentation:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, image: torch.Tensor) -> ImageType:
        local_p = torch.rand(1).item()
        if local_p > self.p:
            return image
        else:
            data_c, data_x, data_y = image.squeeze().shape
            # generate random points
            min_radius = data_x // 30
            rnd_x = torch.randint(min_radius, data_x - min_radius, size=(1,))
            rnd_y = torch.randint(min_radius, data_y - min_radius, size=(1,))
            center = (rnd_x.item(), rnd_y.item())
            rnd_size_divider = torch.randint(27, 50, size=(1,)).item()

            artifacts = torch.ones((data_x, data_y, 1), dtype=torch.uint8).numpy() * 255
            circle_mask = (
                cv.circle(
                    artifacts,
                    center,
                    radius=data_x // rnd_size_divider,
                    color=(0, 0, 0),
                    thickness=-1,
                )
                / 255
            )
            rnd_blurr_divider = torch.randint(15, 20, size=(1,)).item()
            blurred_mask = cv.blur(circle_mask, (data_x // rnd_blurr_divider, data_y // rnd_blurr_divider))

            return image * blurred_mask  # , blurred_mask


class GenerativeTransform:
    """
    Designed to be used as target transform in the GenerativeDataset.
    Receives two callable transforms, which modify the image input image.
    """

    def __init__(self, image_transform: Callable | None, target_transform: Callable | None) -> None:
        self.it = image_transform
        self.tt = target_transform

    def __call__(self, case: dict[str, torch.Tensor]) -> Any:
        res = {
            "image": self.it(case["image"]) if self.it else case["image"],
            "target": self.tt(case["image"]) if self.tt else case["image"],
            "meta": case["meta"] if "meta" in list(case.keys()) else {},
        }
        return res


def get_color_prediction_transform():
    """
    Reconstructing a normalized version of the input image
    Standard ImageNet-based normalization used for RGB images
    if working with 1 channel images, other normalzation should be used
    """
    return GenerativeTransform(
        image_transform=None,
        target_transform=transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    )


def get_surrounding_prediction_transform(center_crop: tuple[int]):
    """
    Reconstruction of the image itself and the surroundings.
    To be used with the SEMSEG decoder or attention head.
    """
    return GenerativeTransform(
        image_transform=transforms.RandomResizedCrop(size=center_crop),
        target_transform=transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    )
