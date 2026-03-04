"""
You can use the common S3 defaults by using the environment variables: S3URL, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY.
Then, you can interact with s3 using `minio_client: Minio = mmm.settings.mtl_settings.s3`.
"""

from __future__ import annotations

import typing
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image

from mmm.utils import disk_cacher

try:
    from minio import Minio
except ImportError:
    if not typing.TYPE_CHECKING:
        Minio = None

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import make_dataset


@disk_cacher(cache_path="shared")
def make_dataset_cached(*args, **kwargs):
    return make_dataset(*args, **kwargs)


class S3ImageFolder(ImageFolder):
    """
    Listing many files with S3 is slow. This class behaves like ImageFolder, but it caches the file list.
    """

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: dict[str, int],
        extensions: tuple[str, ...] | None = None,
        is_valid_file: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, int]]:
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset_cached(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)


def download_object(client: Minio, bucket_name: str, object_name: str) -> BytesIO:
    """
    Downloads an object from the bucket without saving it to disk.
    """
    file_obj = BytesIO()
    try:
        response = client.get_object(
            bucket_name,
            object_name,
        )

        for d in response.stream(amt=1024 * 1024):
            file_obj.write(d)

        file_obj.seek(0)
        return file_obj

    finally:
        if response is not None:
            response.close()
            response.release_conn()


@disk_cacher(cache_path="shared")
def index_files(bucket_name: str, p: Path, recursive=True) -> list:
    """
    Indexes files in a bucket.
    """
    from mmm.settings import mtl_settings

    # The prefix consists of the path except for the first part, which needs to be the bucket name.
    prefix = "/".join(p.parts[2 if p.is_absolute else 1 :])

    bucket_path = Path("".join(p.parts[: 2 if p.is_absolute else 1]))
    return [
        bucket_path / object.object_name
        for object in mtl_settings.s3.list_objects(bucket_name, prefix=prefix, recursive=recursive)
    ]


@disk_cacher(cache_path="shared")
def cacheglob(path: Path, pattern: str) -> list:
    return list(path.glob(pattern))


def upload_img(
    mclient: Minio,
    bucket: str,
    prefix,
    img: Image,
    img_name,
    format: str = "PNG",
    base_url: str = "http://s3.datanodefec:9500",
    data_key: str = "image",
):
    # Convert image to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format=format)
    img_bytes.seek(0)
    r = mclient.put_object(bucket, f"{prefix}/{img_name}", img_bytes, length=img_bytes.getbuffer().nbytes)
    return {"data": {data_key: f"{base_url}/{bucket}/{prefix}/{img_name}"}}
