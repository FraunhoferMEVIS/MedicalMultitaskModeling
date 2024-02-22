import torch
import json
import numpy as np
from typing import Any
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import imageio.v3 as imageio
from torchvision.datasets.folder import default_loader


class DiskDataset(Dataset):
    def __init__(self, folder_path, meta_file="mtlfiledataset.json"):
        self.folder_path = folder_path
        self.case_paths: list[Path] = self.update()
        self.dataset_meta: dict = json.loads((self.folder_path / meta_file).read_text())

    def update(self) -> list[Path]:
        self.case_paths = list(filter(lambda x: x.is_dir(), self.folder_path.iterdir()))
        return self.case_paths

    def __len__(self):
        self.update()
        return len(self.case_paths)

    def __getitem__(self, index) -> dict[str, Any]:
        folder_path = self.case_paths[index]
        res = {}
        # Load jpg as Tensors
        for img_path in folder_path.glob("*.jpg"):
            res[img_path.stem] = F.to_tensor(default_loader(img_path))
        # Load npz using numpy
        for npz_path in folder_path.glob("*.npz"):
            res[npz_path.stem] = torch.from_numpy(np.load(npz_path)["arr_0"])
        # Load case.json into meta key
        res["meta"] = json.load((folder_path / "case.json").open())
        res["meta"]["case_path"] = str(folder_path)
        return res
