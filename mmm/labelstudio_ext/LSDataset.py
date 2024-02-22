import torch
import json
import numpy as np
from typing import Any
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import imageio.v3 as imageio
from torchvision.datasets.folder import default_loader
from mmm.BaseModel import BaseModel
from mmm.utils import disk_cacher

# Enable extra fields such that not everything from labelstudio needs to be modeled
from pydantic import BaseModel as RawBaseModel, ConfigDict

from .projects import LSProject
from mmm.labelstudio_ext.utils import download_image, convert_task_to_seglabel


class LSCase(RawBaseModel):  # Not use mmm.BaseModel to allow for extra fields
    model_config = ConfigDict(extra="allow")
    id: int
    task: dict
    project: int
    result: list[dict]


class LSDataset(Dataset):
    """
    Wraps a target storage of labelstudio
    """

    class Config(BaseModel):
        abs_folder_path: str
        # only_for_project_id: None | str = None
        project: LSProject.Config = LSProject.Config(name="syndifact")

    def __init__(self, args: Config):
        self.cfg = args
        self.folder_path = Path(self.cfg.abs_folder_path)
        self.case_paths: list[Path] = self.update_from_disk()

    def update_from_disk(self) -> list[Path]:
        self.case_paths = list(self.folder_path.iterdir())

        return self.case_paths

    def __len__(self):
        self.update_from_disk()
        return len(self.case_paths)

    def __getitem__(self, index) -> dict:
        case_path = self.case_paths[index]
        case_dict = json.loads((case_path).read_text())
        return LSCase(**case_dict)

    @staticmethod
    def loadmcsemseg(x: LSCase, project: LSProject):
        """
        can be used as a src_transform like `partial(loadmcsemseg, project=ls_dataset.project)`
        """

        @disk_cacher(cache_path="shared")
        def download_image_cached(url):
            """
            Labelstudio often only stores the url to the image, so we need to download it.
            During training that can be a bottleneck, so we cache it by default.
            """
            return download_image(url)

        img = download_image_cached(x.task["data"]["image"])
        seglabel = convert_task_to_seglabel(
            {
                "id": x.task["id"],
                "annotations": [{"result": x.result}],
                "data": {"image": x.task["data"]["image"]},
                "meta": {"labelstudioid": x.task["id"]},
            },
            project.ls.parsed_label_config["histoarteseg"]["labels"],
            prefill_value=-1,
        )
        return {
            "image": F.to_tensor(img),
            "label": torch.from_numpy(seglabel).long(),
            "meta": {
                "labelstudioid": x.task["id"],
                "lslink": f"/projects/{project.ls.id}/data?task={x.task['id']}",
            },
        }
