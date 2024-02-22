import time
import json
import logging
import requests
import datetime
import uuid
from typing import Optional, Any
import requests
from urllib.parse import urlparse, urljoin
import os
from io import BytesIO
from PIL import Image
import imageio.v3 as imageio
from pathlib import Path
import tempfile
import numpy as np
import io
import base64

try:
    from label_studio_sdk import Client, Project
    from label_studio_converter.brush import (
        decode_rle,
        encode_rle,
        decode_from_annotation,
    )
except ImportError:
    Client, Project = Any, Any

from mmm.typing_utils import get_colors
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


def get_image_urls(n: int) -> list[str]:
    return [f"https://picsum.photos/id/{i}/200/300" for i in range(1, n + 1)]


def add_npimage_to_project(project: Project, img: np.ndarray, metainfo: Optional[dict] = None):
    """
    Writes the image to a temporary file
    """
    tmpfile = tempfile.NamedTemporaryFile(suffix=".jpg")
    imageio.imwrite(tmpfile.name, img)
    imported_ids = project.import_tasks(Path(tmpfile.name))
    if metainfo is None:
        metainfo = {}

    assert "shape" not in metainfo
    metainfo["shape"] = img.shape

    project.update_task(imported_ids[0], meta=metainfo)
    return imported_ids


def brush_annotation_to_npy(result) -> tuple[str, np.ndarray]:
    """
    Expects something like:

    {
      'original_width': 2678,
      'original_height': 2954,
      'image_rotation': 0,
      'value': {'format': 'rle',
        'rle': [],
        'brushlabels': ['lung']},
      'id': 'aaOhNtmESo',
      'from_name': 'tag',
      'to_name': 'image',
      'type': 'brushlabels',
      'origin': 'manual'
    }
    """
    width, height = result["original_width"], result["original_height"]
    classnames = result["value"]["brushlabels"]
    rle = decode_rle(result["value"]["rle"])
    # For unknown reasons, the rle is decoded into four channels
    mask_channels = np.reshape(rle, [height, width, 4])

    mask = mask_channels[:, :, 3]

    assert result["image_rotation"] == 0
    assert len(classnames) == 1
    return classnames[0], mask


def convert_task_to_seglabel(task, class_names: list[str], prefill_value: int = 0) -> np.ndarray:
    labels = [
        brush_annotation_to_npy(result)
        for annotation in task["annotations"]
        for result in annotation["result"]
        if result["type"] == "brushlabels" and result["value"]["brushlabels"][0] in class_names
    ]
    if not labels:
        return None

    # Combine the labels to a single mask
    mask = np.zeros(labels[0][1].shape, dtype=np.int64)
    if prefill_value != 0:
        mask.fill(prefill_value)
    for class_name, annotation_mask in labels:
        # Plus one, because otherwise its impossible to distinguish between background and foreground
        mask[annotation_mask > 0] = class_names.index(class_name)

    # Visualize the mask using
    # Image.fromarray(mask * 50).save("mask.png")
    return mask


def binary_mask_to_result(mask: np.ndarray, class_name: str, brush_name: str, score=None):
    mask_255 = (mask > 0).astype(np.uint8) * 255
    flat_mask = np.repeat(mask_255.ravel(), 4)
    rle = encode_rle(flat_mask)
    res = {
        "id": str(uuid.uuid4())[0:8],
        "type": "brushlabels",
        "value": {"rle": rle, "format": "rle", "brushlabels": [class_name]},
        "origin": "manual",
        "to_name": "image",
        "from_name": brush_name,
        "image_rotation": 0,
        "original_width": mask.shape[1],
        "original_height": mask.shape[0],
    }
    if score is not None:
        res["score"] = score
    return res


def mask_to_annotation(mask, class_names: list[str], brush_names: list[str], ignore_index: int | None = 0) -> list:
    """
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


def download_image(url: str, attempts: int = 5, backoff: float = 1.0, ls_client=None, ensure_rgb=True) -> np.ndarray:
    if url.startswith("data:image"):
        bytestring = url
        img_bytes = io.BytesIO(base64.b64decode(bytestring.split(",")[1]))
        np_img: np.ndarray = imageio.imread(img_bytes)
    elif not bool(urlparse(url).netloc):
        # relative URL, use ls_client
        assert ls_client is not None, f"Relative URL {url} but no ls_client provided"
        r = ls_client.make_request("GET", url)
        np_img: np.ndarray = imageio.imread(r.content)
    else:
        try:
            # Otherwise, it should be a valid URL
            np_img: np.ndarray = imageio.imread(url)
        except Exception as e:
            logging.debug(f"Problem opening {url} as numpy image in attempt {attempts}: {e}")
            if attempts == 0:
                logging.error(f"Error opening {url} as numpy image")
                logging.error(f"Cannot convert {url} to an image with the following exception: {e}")
                raise e
            else:
                time.sleep(backoff)
                return download_image(url, attempts - 1, backoff * 2)

    if ensure_rgb:
        if len(np_img.shape) == 2:
            np_img = np.repeat(np_img[:, :, None], 3, axis=2)
        elif np_img.shape[2] == 4:
            np_img = np_img[:, :, :3]
        elif np_img.shape[-1] == 1:
            np_img = np.repeat(np_img, 3, axis=2)
        else:
            assert np_img.shape[2] == 3 and len(np_img.shape) == 3, f"Image shape {np_img.shape} not understood"

    return np_img


def download_image_from_task(url: str, token: str, task: dict, image_name: str = "image") -> Image:
    headers = {"Authorization": f"Token {token}"}
    download_url = f"{url}{task['data'][image_name]}"
    response = requests.get(download_url, headers=headers)
    # Create PIL image from response
    image = Image.open(BytesIO(response.content))

    # Ensure the image is RGB, sometimes they come in grayscale. Check if there are 3 channels and if not, convert
    if image.mode != "RGB" or len(image.getbands()) != 3:
        image = image.convert("RGB")

    return image
