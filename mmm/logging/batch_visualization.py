import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from mmm.logging.ZipLog import ZipLog
from mmm.settings import mtl_settings


def _json_serializer(o: Any):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, Path):
        return o.absolute().__str__()
    return str(o)


@torch.no_grad()
def visualize_batch(
    input_batch: torch.Tensor,
    meta: list[dict],
    overlays: dict[str, tuple[torch.Tensor, list[str]]] | None = None,
    metrics: list[dict] | None = None,
    captions: list[str] | None = None,
    batch_info: dict | None = None,
    **kwargs,
):
    """
    Creates a directory layout that can be read by the M3 Streamlit tools.

    Args:
        input_batch: If the input images are not Bx3xHxW, they are treated as neural representations.
        meta: list of length B with metadata dicts for each image (full details)
        overlays: dict mapping segmentation category to tuple of (B x Classes x H x W tensor, list of class names)
        metrics: list of length B with metric dicts for each image (summary stats like dice)
        captions: list of length B with captions for each image

    >>> import tempfile, zipfile, skimage.data as skdata, m3_sdk.DistributedPath as dist_path
    >>> Path(d := tempfile.TemporaryDirectory().name).mkdir()
    >>> cloud_path = dist_path.DistributedPath.from_string(d).joinpath("ziplog_doctest.zip")
    >>> input_batch = torch.stack([F.to_tensor(skdata.astronaut()), F.to_tensor(skdata.astronaut())])

    >>> log = visualize_batch(input_batch, [{}, {}], upload_path=cloud_path)

    >>> [
    ...     info.filename.split("/")[-1]
    ...     for info in zipfile.ZipFile(log.upload_path.to_local_path(), 'r').infolist()
    ...     if ".jpg" in info.filename
    ... ]
    ['input_image.jpg', 'input_image.jpg']

    If the input data does not correspond to RGB images, we assume that the data contains neural representations:

    >>> repr_batch = torch.rand(B := 2, Z := 7)  # Two representations with dimensionality 7
    >>> log = visualize_batch(repr_batch, [{}, {}], upload_path=cloud_path)
    >>> [
    ...     info.filename.split("/")[-1]
    ...     for info in zipfile.ZipFile(log.upload_path.to_local_path(), 'r').infolist()
    ...     if ".pt" in info.filename
    ... ]
    ['input.pt', 'input.pt']
    """
    assert len(meta) == input_batch.shape[0]
    if overlays is None:
        overlays = {}

    for seg_category, (seg_data, class_names) in overlays.items():
        assert seg_data.shape[1] == len(class_names), f"{seg_category}: {seg_data.shape[1]=}, {len(class_names)=}"
        assert input_batch.shape[0] == seg_data.shape[0], f"{input_batch.shape[0]=}, {seg_data.shape[0]=}"

    if metrics is not None:
        assert len(metrics) == len(meta)

    group_indices = {}
    for i, m in enumerate(meta):
        group_indices.setdefault(m.get("group_id", "nogroup"), []).append(i)

    with (log := ZipLog(**kwargs)).add_files() as logdir:
        if batch_info is not None:
            logdir.joinpath("batch_info.json").write_text(json.dumps(batch_info, default=_json_serializer, indent=4))
        for group_id, group_indices in group_indices.items():
            (group_dir := logdir.joinpath(str(group_id))).mkdir(parents=True, exist_ok=True)
            for idx in group_indices:
                (item_dir := group_dir.joinpath(f"batch_idx_{idx:010d}")).mkdir(parents=True, exist_ok=True)
                overlays_meta = {"overlay_categories": {}}
                item_dir.joinpath(f"meta.json").write_text(json.dumps(meta[idx], default=_json_serializer, indent=4))
                if captions is not None:
                    item_dir.joinpath(f"caption.txt").write_text(captions[idx])
                if metrics is not None:
                    item_dir.joinpath(f"metrics.json").write_text(json.dumps(metrics[idx], indent=4))

                if len(input_batch[idx].shape) >= 2:
                    input_img = F.to_pil_image(input_batch[idx])
                    input_img.save(item_dir.joinpath(f"input_image.jpg"), format="JPEG", quality=95)
                else:
                    torch.save(input_batch[idx], item_dir.joinpath("input.pt"))

                for seg_category, (seg_data, class_names) in overlays.items():
                    (out_dir := item_dir.joinpath(seg_category)).mkdir(parents=True, exist_ok=True)

                    # For predictions, log the highest mtl_settings.max_classes_detailed_logging classes
                    # For ground truth, log all classes with at least one positive value
                    if seg_data.dtype != torch.long:  # ground truth masks are long tensors
                        selected_indices = torch.argsort(seg_data[idx].sum(dim=(-1, -2)), descending=True).tolist()[
                            : mtl_settings.max_classes_detailed_logging
                        ]
                    else:
                        selected_indices = [i for i in range(seg_data.shape[1]) if seg_data[idx, i].sum() > 0]

                    for class_i in selected_indices:
                        class_pred = (seg_data[idx, class_i].float().numpy() * 255).astype(np.uint8)
                        rgba_pred = np.stack(
                            [
                                np.ones_like(class_pred) * 255,
                                zeroes := np.zeros_like(class_pred),
                                zeroes,
                                class_pred,
                            ],
                            axis=-1,
                        )
                        overlay_path = out_dir.joinpath(f"{class_names[class_i]}.png")
                        Image.fromarray(rgba_pred, mode="RGBA").save(overlay_path)
                        overlays_meta["overlay_categories"].setdefault(seg_category, []).append(
                            dict(class_name=class_names[class_i], file_path=str(overlay_path.relative_to(item_dir)))
                        )
                item_dir.joinpath("overlays_meta.json").write_text(json.dumps(overlays_meta, indent=4))

    return log
