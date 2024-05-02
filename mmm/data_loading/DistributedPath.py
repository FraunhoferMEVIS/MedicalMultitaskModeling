from __future__ import annotations
import urllib
import os
from fsspec.core import OpenFile
import fsspec
from upath import UPath
from pydantic import Field, BaseModel, model_validator


class DistributedPath(BaseModel):
    """
    Wraps fsspec with a pydantic model:

    >>> from mmm.data_loading.DistributedPath import DistributedPath
    >>> dp = DistributedPath(uri="s3://dataroot/histoartefact_preannotation/SynD_0_001.svs")
    >>> fs, f = dp.fs(), dp.file()

    Can be used with paths like "s3://bucketname/file", "/path/to/file" and other fsspec URIs.
    If no options are provided, it will default to Minio settings: S3URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.

    Use `upath` for a pathlib like interface provided by `universal_pathlib` package:

    >>> assert len(list(dp.upath().parent.glob("*"))) > 0

    """

    uri: str = Field(..., description="FSSpec URI to the file")
    options: dict = Field({}, description="Options to pass to fsspec")

    @model_validator(mode="before")
    @classmethod
    def serialize_from_string(cls, data):
        if isinstance(data, str):
            data = DistributedPath(uri=data).model_dump()

        return data

    @model_validator(mode="after")
    def fill_s3_defaults(self):
        if self.uri.startswith("s3://") and not self.options:
            self.options = {
                "endpoint_url": os.getenv("S3URL", default="http://localhost:9000"),
                "key": os.getenv("AWS_ACCESS_KEY_ID", default="minioadmin"),
                "secret": os.getenv("AWS_SECRET_ACCESS_KEY", default="minioadmin"),
            }

        return self

    def file(self, *args, **kwargs) -> OpenFile:
        return fsspec.open(self.uri, *args, **kwargs, **self.options)

    def upath(self, *args, **kwargs) -> UPath:
        return UPath(self.uri, *args, **kwargs, **self.options)

    def fs(self):
        return fsspec.filesystem(self.get_protocol(self.uri), **self.options)

    def get_protocol(self, path):
        # Parse the path using urllib.parse
        parsed_path = urllib.parse.urlparse(path)

        # Extract the scheme (protocol) from the parsed path
        return parsed_path.scheme if parsed_path.scheme else "file"

    def exists(self) -> bool:
        return self.fs().exists(self.uri)
