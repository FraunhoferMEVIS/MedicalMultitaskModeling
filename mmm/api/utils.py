import base64
import datetime
import io
import json
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np
import requests
from m3_sdk.DistributedPath import DistributedPath
from PIL import Image, ImageOps

try:
    from label_studio_sdk import Client, Project
    from label_studio_sdk.converter.brush import encode_rle
except ImportError:
    if not TYPE_CHECKING:
        Client, Project = Any, Any
    else:
        raise  # Avoids errors in type checking tools

from mmm.utils import remove_folder_blocking_if_exists


def sync_to_disk(
    project: Project,
    data_folder: Path,
    annotations,
    case_writer,
    dateformat: str = r"%Y-%m-%dT%H:%M:%S.%fZ",
):
    """
    LabelStudio has this built in, just add a target storage
    """
    data_folder.mkdir(parents=True, exist_ok=True)
    tasks_updated_cachepath: Path = data_folder / "tasks_updated_at.json"
    if tasks_updated_cachepath.exists():
        tasks_updated_at = {
            int(annoid): datetime.datetime.strptime(taskupdated, dateformat)
            for annoid, taskupdated in json.loads(tasks_updated_cachepath.read_text()).items()
        }
    else:
        tasks_updated_at = {}
    # Contains a map from the unique labelstudio id to a updated_at date like: 2023-09-24T07:18:12.618719Z
    tasks_updated_at_new = {a["id"]: datetime.datetime.strptime(a["updated_at"], dateformat) for a in annotations}
    updated_tasks = {
        new_anno_id: newdate
        for new_anno_id, newdate in tasks_updated_at_new.items()
        if new_anno_id not in tasks_updated_at or tasks_updated_at[new_anno_id] < newdate
    }
    logging.info(f"Found tasks which need to be updated: {updated_tasks.keys()}")
    tasks_updated_at.update(updated_tasks)

    # Download all images and masks
    for taskdict in filter(lambda a: a["id"] in updated_tasks.keys(), annotations):
        casefolder = data_folder / f"{taskdict['id']}"
        logging.info(f"Updating case on disk for annotation {taskdict['id']} in {casefolder}")
        # clear if it already exists
        remove_folder_blocking_if_exists(casefolder)
        casefolder.mkdir(parents=True, exist_ok=True)
        case_writer(taskdict, casefolder)

    tasks_updated_cachepath.write_text(
        json.dumps(
            tasks_updated_at,
            indent=2,
            default=lambda x: datetime.datetime.strftime(x, dateformat),
        )
    )
    (data_folder / "mtlfiledataset.json").write_text(json.dumps(project.get_params(), indent=2))


def binary_mask_to_result(mask: np.ndarray, class_name: str, brush_name: str, score=None, image_tag="image") -> dict:
    mask_255 = (mask > 0).astype(np.uint8) * 255
    flat_mask = np.repeat(mask_255.ravel(), 4)
    rle = encode_rle(flat_mask)
    res = {
        "id": str(uuid.uuid4())[0:8],
        "type": "brushlabels",
        "value": {"rle": rle, "format": "rle", "brushlabels": [class_name]},
        "to_name": image_tag,
        "from_name": brush_name,
        "image_rotation": 0,
        "original_width": mask.shape[1],
        "original_height": mask.shape[0],
    }
    if score is not None:
        res["score"] = score
    return res


def mask_to_annotation(
    mask: np.ndarray,
    class_names: list[str],
    brush_names: list[str],
    ignore_index: int | None = 0,
) -> list:
    """
    - mask: a numpy array with shape (H, W) containing the class indices

    project.create_annotation(
        task_id=project.tasks[-1]["id"],
        result=mask_to_annotation(mask, ["notannotated", "foreground"])
    )
    """
    return [
        binary_mask_to_result(mask == i, class_name, brush_name)
        for i, (class_name, brush_name) in enumerate(zip(class_names, brush_names))
        if i in mask and (ignore_index is None or i != ignore_index)
    ]


def download_image(
    url: str | io.BytesIO,
    ensure_rgb=True,
    local_file_prefix="/data/local-files/?d=",
) -> Image.Image:
    if isinstance(url, io.BytesIO):
        img_bytes = url
    elif url.startswith("data:image"):
        bytestring = url
        img_bytes = io.BytesIO(base64.b64decode(bytestring.split(",")[1]))
    elif url.startswith(local_file_prefix):
        img_bytes = BytesIO(DistributedPath.from_string(url[len(local_file_prefix) :]).upath().read_bytes())
    elif (image_upath := DistributedPath.from_string(url).upath()).exists():
        img_bytes = BytesIO(image_upath.read_bytes())
    else:
        r = requests.get(url)
        if r.status_code != 200:
            raise ValueError(f"Could not download image from {url}, got status code {r.status_code}")
        img_bytes = io.BytesIO(r.content)

    pil_image: Image.Image = Image.open(img_bytes)
    # Some images might have EXIF transformations
    if (transposed := ImageOps.exif_transpose(pil_image)) is not None:
        pil_image = transposed
    if ensure_rgb:
        pil_image = pil_image.convert("RGB")

    return pil_image
