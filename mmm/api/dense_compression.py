from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from einops import rearrange

from mmm.settings import mtl_settings


def prediction_to_spatial(mask_size: tuple, orig_size: tuple, probas: np.ndarray):
    """
    Takes scores in the shape (N, C) and converts them to a spatial format.

    First, it reshapes to the spatial size of the mask, then resizes to the original image size.
    """
    probas_spatial = probas.view(*mask_size, probas.shape[1])
    probas_spatial = F.resize(
        probas_spatial.permute(2, 0, 1),
        orig_size,
        interpolation=F.InterpolationMode.NEAREST_EXACT,
    ).permute(1, 2, 0)
    return probas_spatial


def generate_multi_class_contour_mask(for_mask: torch.Tensor, thickness: int = 3):
    for batch_dim in range(for_mask.shape[0]):
        mask_uint8 = for_mask[batch_dim].squeeze().numpy().astype(np.uint8)
        contour_mask = np.zeros(mask_uint8.shape, dtype=np.uint8)

        for class_value in np.unique(mask_uint8):
            class_mask = np.where(mask_uint8 == class_value, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_mask, contours, -1, 255, thickness=thickness)

        return contour_mask


def compute_pixel_features_sklearn(feature_maps, masks: torch.Tensor, only_for_pixels: torch.Tensor = None):
    """Computation of the dense features

    Take for example these feature maps from the encoder and decoder for one image:

    ```python
    feature_maps = torch.Tensor([
        [[0, 0], [2, 0]],
        [[1, 0], [1, 1]],
        [[2, 4], [2, 2]],
    ])
    feature_maps = torch.stack([feature_maps.clone(), (feature_maps + 3).clone()]) # (B=2, C=3, H=2, W=2)
    # or:
    feature_maps = torch.stack([
        torch.stack([torch.arange(16).view(4, 4) for i in range(3)])
        for _ in range(2)])
    masks = torch.stack([torch.arange(16).view(4, 4) for _ in range(2)])
    ```
    The first feature map is [[0, 0], [0, 0]] and the last is [[2, 2], [2, 2]].
    The first pixel should have the feature vector [0, 1, 2].
    The pixel features should have shape (8, 3) or (B=2, 4, C=3)
    """
    assert masks.shape == only_for_pixels.shape
    # If the spatial shapes do not match, resize the mask to fit the spatial features
    if feature_maps.shape[-2:] != masks.shape[-2:]:
        target_dtype = masks.dtype
        masks = F.resize(masks, feature_maps.shape[-2:], interpolation=F.InterpolationMode.NEAREST_EXACT).type(
            target_dtype
        )

    if only_for_pixels is not None and only_for_pixels.shape[-2:] != masks.shape[-2:]:
        only_for_pixels = F.resize(
            only_for_pixels, feature_maps.shape[-2:], interpolation=F.InterpolationMode.NEAREST_EXACT
        ).bool()

    # Transpose the channel dimension to the end
    # transposed = feature_maps.permute(0, 2, 3, 1)

    # Reshape into (B*H*W, C)
    # flattened = transposed.contiguous().view(-1, transposed.size(3))

    flattened = rearrange(feature_maps, "B C H W -> (B H W) C")
    flat_mask = rearrange(masks, "B 1 H W -> (B H W)")
    # flat_mask = masks.contiguous().view(-1)

    if only_for_pixels is not None:
        # Reshape the spatial only_for_pixels mask
        # flat_only_for_pixels = only_for_pixels.contiguous().view(-1)
        flat_only_for_pixels = rearrange(only_for_pixels, "B 1 H W -> (B H W)")
        # Mask out the pixels that are not in the only_for_pixels mask
        flat_mask = flat_mask[flat_only_for_pixels > 0]
        flattened = flattened[flat_only_for_pixels > 0]

    return flattened, flat_mask


def compute_pixel_features_sklearn_by_purpose(
    spatial_features: torch.Tensor,
    for_mask: torch.Tensor | None,
    for_inference: bool,
    ignore_index: int = mtl_settings.ignore_class_value,
    pixel_droprate: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes sklearn-compatible features from a feature map and, optionally, a mask.

    Args:
        spatial_features: Spatial features of the form (B, C, H, W) and type FloatTensor
        for_mask: (B, H, W) LongTensor with class indices (optional)
        for_inference: if False, only a subset of the pixels may be used to make the training faster
        ignore_index (int, optional): Input pixels at these positions in the mask are not used.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: X, y
    """
    if for_mask is None:
        only_for_pixels = None
        for_mask = torch.zeros(spatial_features.shape[0], *spatial_features.shape[-2:]).fill_(ignore_index)
    else:
        if not for_inference:
            # Only use a subset of the pixels for training
            rand_dropped_out_mask = torch.rand_like(for_mask, dtype=torch.float32) > pixel_droprate
            # Add the contours of the original mask
            contour_mask = torch.Tensor(generate_multi_class_contour_mask(for_mask)).unsqueeze(0).bool()
            only_for_pixels = rand_dropped_out_mask | contour_mask
        else:
            only_for_pixels = None
    X, y = compute_pixel_features_sklearn(spatial_features, for_mask, only_for_pixels=only_for_pixels)
    if not for_inference:
        X = X[y != ignore_index]
        y = y[y != ignore_index]
    return X, y
