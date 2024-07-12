"""
Wraps an nn.ModuleDict that was created using `trainer.save_blocks_native(...)`.
"""

import logging
from pathlib import Path
import torch
import torch.nn as nn

from mmm.data_loading.DistributedPath import DistributedPath
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.utils import get_default_cachepath

MMM_MODELS = {"encoder-1.0.4.pt.zip": "https://owncloud.fraunhofer.de/index.php/s/Q5o1uD5eHL7tr3X/download"}
DEFAULT_MODEL = "encoder-1.0.4.pt.zip"


class NativeBlocks:
    """
    Wraps an nn.ModuleDict that was created using `trainer.save_blocks_native(...)`.
    """

    def __init__(self, modules_path: DistributedPath | str, device_identifier: str = "cuda") -> None:
        if isinstance(modules_path, str):
            modules_path = DistributedPath(uri=modules_path)
        if modules_path.uri.startswith("http"):
            # Download file to cache
            cache_folder = get_default_cachepath("models")
            model_name = {v: k for k, v in MMM_MODELS.items()}.get(modules_path.uri, Path(modules_path.uri).name)
            if not model_name.endswith(".pt.zip"):
                model_name = f"{model_name}.pt.zip"
            if not (target_path := cache_folder / model_name).exists():
                logging.info(f"Model does not exist at {target_path}, downloading...")
                torch.hub.download_url_to_file(modules_path.uri, target_path)
            else:
                logging.info(f"Model already exists at {target_path}, not redownloading")
            modules_path = DistributedPath(uri=str(target_path), options=modules_path.options)

        self.device = device_identifier
        if modules_path.uri.endswith(".pt.zip"):
            import zipfile
            import tempfile

            # Extract the zip file to a temporary directory
            with zipfile.ZipFile(modules_path.upath(), "r") as zip_ref:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    zip_ref.extractall(tmpdirname)
                    modules_path = DistributedPath(uri=tmpdirname, options=modules_path.options)
                    # Load the modules from the temporary directory
                    self.torch_modules: nn.ModuleDict = nn.ModuleDict()
                    for file in modules_path.upath().glob("*.pt"):
                        try:
                            self.torch_modules[file.stem] = torch.load(file).to(device_identifier)
                        except Exception as e:
                            logging.error(f"Did not load {file} due to {e}")

        else:
            self.torch_modules = torch.load(modules_path.upath()).to(device_identifier)

    def get_sharedblock_keys(self) -> list[str]:
        return [k for k, v in self.torch_modules.items() if isinstance(v, SharedBlock)]

    def get_task_keys(self) -> list[str]:
        return [k for k, v in self.torch_modules.items() if isinstance(v, MTLTask)]

    def get_device(self) -> str:
        return self.device

    def __getitem__(self, key):
        return self.torch_modules[key]

    def keys(self):
        return self.torch_modules.keys()

    @staticmethod
    def save_to_disk(file_path: DistributedPath, d: nn.ModuleDict):
        """
        Stores the modules in a folder with one file per module, where the filename is the f"{key}.pt".
        The filename could be the unique name returned by `module.get_name()`.
        """
        if file_path.uri.endswith(".pt.zip"):
            import zipfile
            import tempfile

            with zipfile.ZipFile(file_path.upath(), "w") as zip_ref:
                for k, v in d.items():
                    with tempfile.NamedTemporaryFile() as tmpfile:
                        torch.save(v, tmpfile.name)
                        zip_ref.write(tmpfile.name, f"{k}.pt")
                return file_path
        else:
            # Save the whole dict to a single file
            torch.save(d, file_path.upath())
