"""
You can use the common S3 defaults by using the environment variables: S3URL, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY.
Use this code:

```python
client: Minio = Minio(*get_args(), **get_kwargs())
```
"""

from __future__ import annotations
from typing import Callable
import os
from pathlib import Path
from mmm.utils import disk_cacher
from mmm.BaseModel import BaseModel
from io import BytesIO
from PIL import Image

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    pass

from torchvision.datasets.folder import make_dataset
from torchvision.datasets import ImageFolder


@disk_cacher(cache_path="shared")
def make_dataset_cached(*args, **kwargs):
    return make_dataset(*args, **kwargs)


class S3ImageFolder(ImageFolder):
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


def get_args():
    return (
        os.getenv("S3URL", default="http://localhost:9000").replace("http://", "").replace("https://", ""),
        os.getenv("AWS_ACCESS_KEY_ID", default="minioadmin"),
        os.getenv("AWS_SECRET_ACCESS_KEY", default="minioadmin"),
    )


def get_kwargs():
    return {
        "secure": os.getenv("S3URL", default="http://localhost:9000").startswith("https://"),
    }


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
    client = Minio(*get_args(), **get_kwargs())
    # The prefix consists of the path except for the first part, which needs to be the bucket name.
    prefix = "/".join(p.parts[2 if p.is_absolute else 1 :])

    bucket_path = Path("".join(p.parts[: 2 if p.is_absolute else 1]))
    return [
        bucket_path / object.object_name
        for object in client.list_objects(bucket_name, prefix=prefix, recursive=recursive)
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
    r = mclient.put_object(bucket, f"{prefix}/{img_name}.png", img_bytes, length=img_bytes.getbuffer().nbytes)
    return {"data": {data_key: f"{base_url}/{bucket}/{prefix}/{img_name}.png"}}


class S3Path(BaseModel):
    bucket: str
    path: str

    @classmethod
    def from_str(cls, s3path: str) -> S3Path:
        bucket, path = s3path.split("://")[1].split("/", 1)
        return cls(bucket=bucket, path=path)

    def download(self, client: Minio | None = None) -> BytesIO:
        """
        Uses default environment variables to connect to the S3 bucket if client is None.
        """
        if client is None:
            client = Minio(*get_args(), **get_kwargs())
        return download_object(client, self.bucket, self.path)

    def exists(self, client: Minio | None = None) -> bool:
        if client is None:
            client = Minio(*get_args(), **get_kwargs())
        try:
            return client.bucket_exists(self.bucket) and client.stat_object(self.bucket, self.path)
        except S3Error:
            return False
