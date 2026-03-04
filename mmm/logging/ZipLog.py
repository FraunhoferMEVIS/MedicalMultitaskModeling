import tempfile
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import wandb
from m3_sdk.DistributedPath import DistributedPath

from mmm.settings import mtl_settings


class AddFilesContext(tempfile.TemporaryDirectory):
    def __init__(self, zip_file: zipfile.ZipFile, exit_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_file = zip_file
        self.exit_cb = exit_cb

    def __enter__(self) -> Path:
        log_dir = super().__enter__()
        return Path(log_dir)

    def __exit__(self, exc, value, tb):
        for file_path in Path(self.name).rglob("*"):
            self.zip_file.write(
                file_path,
                arcname=file_path.relative_to(self.name),
            )
        self.zip_file.close()
        self.exit_cb()
        return super().__exit__(exc, value, tb)


class ZipLog:
    """
    >>> from mmm.logging.ZipLog import ZipLog, DistributedPath; import tempfile

    If not set, extracted from settings
    >>> d = tempfile.TemporaryDirectory()
    >>> cloud_path = DistributedPath.from_string(d.name).joinpath("ziplog_doctest.zip")

    All files added within the `add_files` context will be zipped and uploaded to `upload_path` on exit:
    >>> with (log := ZipLog(upload_path=cloud_path)).add_files() as logdir:
    ...     _ = logdir.joinpath("log.txt").write_text("Top-level log file.")

    Build the artifact and that can be used with `wandb.log_artifact(artifact)`:
    >>> artifact = log.build_artifact()

    Optionally, log an instruction `wandb.log("predictions", log.build_instruction())`:
    >>> _ = log.build_instruction()
    """

    def __init__(self, mode=zipfile.ZIP_STORED, upload_path: DistributedPath | None = None):
        self.zip_buffer = BytesIO()
        self.zip_file = zipfile.ZipFile(self.zip_buffer, "w", compression=mode)
        if upload_path is None:
            from mmm.settings import mtl_settings

            upload_path = mtl_settings.default_log_folder.joinpath(f"log_{uuid.uuid4().hex}.zip")

        self.upload_path: DistributedPath = upload_path

    def add_files(self):
        return AddFilesContext(self.zip_file, exit_cb=self.upload)

    def upload(self):
        self.zip_buffer.seek(0)
        self.upload_path.upath().write_bytes(self.zip_buffer.read())

    def build_instruction(self) -> wandb.Html:
        # columns = ["View Link", "Log File Location"]
        html = wandb.Html(
            data=f"""<pre>{self.upload_path.uri}</pre>
<a href="{mtl_settings.st_app_base}?logzip={self.upload_path.uri}" target="_blank">View Log in Viewer on {mtl_settings.st_app_base}</a>
"""
        )
        return html

    def build_artifact(
        self,
        name: str | None = None,
        kind: str = "prediction",
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        incremental: bool = False,
        use_as: str | None = None,
    ):
        if name is None:
            name = self.upload_path.get_filestem_suggestion()

        res = wandb.Artifact(
            name=name,
            type=kind,
            description=description,
            metadata=metadata,
            incremental=incremental,
            use_as=use_as,
        )
        res.add_reference(self.upload_path.uri)
        return res
