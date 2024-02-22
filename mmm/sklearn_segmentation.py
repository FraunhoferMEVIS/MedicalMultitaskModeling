from typing import Any
from pathlib import Path
import numpy as np
from io import BytesIO
import requests
from PIL import Image
import imageio.v3 as imageio
import torchvision.transforms.functional as F
import torch
from sklearn.ensemble import RandomForestClassifier
import albumentations as A
from torchvision.transforms import InterpolationMode

# Enable optional dependencies
try:
    from label_studio_sdk import Client, Project
except ImportError:
    Client, Project = Any, Any

from mmm.interactive import pipes


@torch.inference_mode()
def img_to_segmentationfeatures(img: torch.Tensor, shape, encoder, decoder):
    inps = img.to(encoder.torch_device).unsqueeze(0)
    pyr, _ = encoder(inps)
    features = decoder(pyr)
    if shape != features.shape[1:]:
        features = F.resize(features, shape, interpolation=InterpolationMode.BILINEAR)
    return features.cpu()


def case_to_sklearn(case: dict[str, any], features: torch.Tensor, ignore_index=None):
    if ignore_index is None:
        # All rows and columns are relevant
        rows, cols = torch.where(torch.ones_like(case["label"]))
    else:
        rows, cols = torch.where(case["label"] != ignore_index)
    X = features[:, rows, cols].T
    y = case["label"][rows, cols]
    return X, y


def cases_to_sklearn(cases: list[dict[str, any]], features: torch.Tensor, ignore_index=0):
    cases = [case_to_sklearn(case, features[i], ignore_index) for i, case in enumerate(cases)]
    X = torch.cat([case[0] for case in cases])
    y = torch.cat([case[1] for case in cases])
    return X.numpy(), y.numpy()


if __name__ == "__main__":
    import pydantic

    pydantic.class_validators._FUNCS.clear()
    import streamlit as st
    from mmm.logging.st_ext import blend_with_mask

    LSURL = "http://localhost:8080"
    LSTOKEN = "2ac7ff7ae38466ebd19122ae019d11092176da2c"
    CLASS_NAMES = ["nonannotated", "background", "heart", "lung"]
    MODULES_PATH = Path("/host-home/shares/imageData/deep_learning/output/jschaefer/univ_sh_modules/")

    lsclient = Client(LSURL, LSTOKEN)
    project: Project = lsclient.get_project(lsclient.get_projects()[0].id)
    st.title(f"Project with {len(project.tasks)} tasks")

    preparer = pipes.Alb([A.Resize(224, 224)])

    # build cases from annotations
    cases = build_all_cases_for_project(project, transform=preparer)

    encoder = torch.load(MODULES_PATH / "encoder.pt")
    decoder = torch.load(MODULES_PATH / "decoder.pt")
    features = img_to_segmentationfeatures([case["image"] for case in cases], cases[0]["label"].shape, encoder, decoder)
    X, y = cases_to_sklearn(cases, features)

    rf = RandomForestClassifier().fit(X, y)

    fullX, _ = case_to_sklearn(cases[0], features[0], ignore_index=None)
    preds = rf.predict(fullX).reshape(cases[0]["label"].shape)
    st.write(fullX.shape)
    st.write(preds.shape)
    blend_with_mask(cases[0]["image"], torch.from_numpy(preds), classes=CLASS_NAMES, st_key="preds")
