from pathlib import Path
from typing import Any, Callable, TypeVar

import albumentations as A
import cv2
import cv2 as cv
import torch
from monai.data.meta_tensor import MetaTensor
from PIL.Image import Image
from torchvision import transforms

from mmm.logging.type_ext import TransformsSeqType
from mmm.settings import mtl_settings
from mmm.transforms import RandomApply

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
                A.ToGray(p=0.1),
                A.ChannelShuffle(p=0.2),
                # Should approximate typical color variation in stains
                A.HueSaturationValue(
                    hue_shift_limit=30,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    always_apply=False,
                    p=0.7,
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
                A.GaussNoise(p=1),
            ],
            p=0.5,
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
            fill=img_fill_value,
            fill_mask=mask_fill_value,
        ),
    ]


def get_artefact_histo_augs():
    return [
        RandomBlobAugmentation(),
        RandomLineAugmentation(),
        RandomBlobAugmentation(),
    ]


def get_weak_default_augs(
    img_fill_value=0, mask_fill_value=mtl_settings.ignore_class_value, with_boxes=False
) -> TransformsSeqType:
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
            fill=img_fill_value,
            fill_mask=mask_fill_value,
        ),
        A.OneOf(
            (
                [A.OpticalDistortion(p=0.3)]
                if with_boxes
                else [
                    A.OpticalDistortion(p=0.3),
                    # This has a bug with boxes: "IndexError: index 3 is out of bounds for axis 1 with size 3"
                    A.GridDistortion(p=0.1),
                ]
            ),
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


def get_realworld_augs(with_boxes=True) -> TransformsSeqType:
    return [
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
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
            (
                [A.OpticalDistortion(p=0.3)]
                if with_boxes
                else [
                    A.OpticalDistortion(p=0.3),
                    # A.GridDistortion has a bug with boxes: "IndexError: index 3 is out of bounds for axis 1 with size 3"
                    A.GridDistortion(p=0.1),
                    # This has a bug, it swaps all non-zero labels in the mask to one :O
                    # A.PiecewiseAffine(p=0.3),
                ]
            ),
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
        A.ShiftScaleRotate(
            p=1.0,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            fill=img_fill_value,
            fill_mask=mask_fill_value,
        ),
    ]


def get_mri2d_augs(use_boxes: bool = False) -> TransformsSeqType:
    return [
        A.RandomRotate90(p=1.0),
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
            (
                [A.OpticalDistortion(p=0.3)]
                if use_boxes
                else [
                    A.OpticalDistortion(p=0.3),
                    # This has a bug with boxes: "IndexError: index 3 is out of bounds for axis 1 with size 3"
                    A.GridDistortion(p=0.1),
                    # This has a bug, it swaps all non-zero labels in the mask to one :O
                    # A.PiecewiseAffine(p=0.3),
                ]
            ),
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


class PoissonNoise(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(p, always_apply)

    def apply(self, img, **params):
        noise = np.random.poisson(img, img.shape)
        img = img + noise
        return img.astype(np.uint8)


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
            line_mask = cv2.line(artifacts, pt1, pt2, color=(0, 0, 0), thickness=data_x // 32) / 255

            rnd_blurr_divider = torch.randint(15, 20, size=(1,)).item()
            # (data_x//rnd_blurr_divider, data_y//rnd_blurr_divider))
            blurred_mask = cv2.blur(line_mask, (data_x // rnd_blurr_divider, data_y // rnd_blurr_divider))

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
                cv2.circle(
                    artifacts,
                    center,
                    radius=data_x // rnd_size_divider,
                    color=(0, 0, 0),
                    thickness=-1,
                )
                / 255
            )
            rnd_blurr_divider = torch.randint(15, 20, size=(1,)).item()
            blurred_mask = cv2.blur(circle_mask, (data_x // rnd_blurr_divider, data_y // rnd_blurr_divider))

            return image * blurred_mask  # , blurred_mask


class UniTransform:
    """
    Warps Uni version 1 into a callabl class
    Weights obtained from https://huggingface.co/MahmoodLab/UNI
    """

    def __init__(self, path_to_weights: str | Path) -> None:
        import timm
        import torch
        from torchvision import transforms

        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
            global_pool="token",
            embed_dim=1024,
        )
        state_dict = torch.load(path_to_weights)
        model.load_state_dict(state_dict, strict=False)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model = model.eval()
        self.hidden_dim = 1024
        self.name = "uni"

    def _move_to_device(self, device):
        print(f"Moving UniTransform to device {device}")
        self.device = device
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image = self.transform(image)
            image = self.model(image.to(self.device)).squeeze()
            return image


class ProvGigaPathPatchTransform:
    """
    Warps ProvGigaPath Patch extractor into a callabl class
    Weights obtained from https://huggingface.co/prov-gigapath/prov-gigapath
    """

    def __init__(self, path_to_weights: str | Path) -> None:
        import timm
        import torch
        from torchvision import transforms

        tile_encoder = timm.create_model(
            "vit_giant_patch14_dinov2",
            img_size=224,
            in_chans=3,
            patch_size=16,
            embed_dim=1536,
            depth=40,
            num_heads=24,
            init_values=1e-05,
            mlp_ratio=5.33334,
            num_classes=0,
            global_pool="token",
        )
        state_dict = torch.load(path_to_weights)
        tile_encoder.load_state_dict(state_dict)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model = tile_encoder.eval()
        self.hidden_dim = 1536
        self.name = "prov-gigapath"

    def _move_to_device(self, device):
        print(f"Moving ProvGigaPath-patch to device {device}")
        self.device = device
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image = self.transform(image)
            image = self.model(image.to(self.device)).squeeze()
            return image


class HOptimus0Transform:
    """
    Warps H-Optimus-0 into a callabl class
    Weights obtained from https://huggingface.co/bioptimus/H-optimus-0

    TODO:
    Needs updated timm library (and also segmentation-models-pytorch).
    """

    def __init__(self, path_to_weights: str | Path) -> None:
        import timm
        import torch
        from torchvision import transforms

        assert timm.__version__[0] == "1"
        f"timm needs to be version 1 or higher. yours: {timm.__version__=}"
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", pretrained=False, init_values=1e-5, dynamic_img_size=False
        )
        model.load_state_dict(torch.load(path_to_weights))
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
            ]
        )
        self.model = model.eval()
        self.hidden_dim = 1536
        self.name = "h-optimus0"

    def _move_to_device(self, device):
        print(f"Moving H-Optimus0 to device {device}")
        self.device = device
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image = self.transform(image)
            image = self.model(image.to(self.device)).squeeze()
            return image


class GenerativeTransform:
    """
    Designed to be used as target transform in the GenerativeDataset.
    Receives two callable transforms, which modify the image input image.
    """

    def __init__(self, image_transform: Callable | None, target_transform: Callable | None) -> None:
        self.it = image_transform
        self.tt = target_transform
        print(f"Got augmentations {self.it=} and {self.tt=}")

    def __call__(self, case: dict[str, torch.Tensor]) -> Any:
        case["image"] = self.it(case["image"]) if self.it else case["image"]
        case["target"] = self.tt(case["image"]) if self.tt else case["image"]
        case["meta"] = case["meta"] if "meta" in list(case.keys()) else {}
        return case


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


def get_place_holder_transform():
    return GenerativeTransform(nn.Identity(), nn.Identity())
