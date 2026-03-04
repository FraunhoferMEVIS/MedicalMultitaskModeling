import os
import pytest
from pathlib import Path

assert (_env_var := os.getenv("MMM_TEST_DATA_DIR")) is not None
TEST_DATA_DIR = Path(_env_var)


IMAGE_IDS = ["naturaljpg", "histojpg"]  # , "s3png"]
IMAGES = [
    _env_var + "/ILSVRC2012_test_00010192.JPEG",
    _env_var + "/norm.16846.jpg",
    # "http://s3.datanodefec:9500/dataroot/shapedata/img_crag35936fc4-fc4c-4ca6-8616-4a9d6b39a098512.png",
]
IMAGE_PROPERTIES = [{"HW": (365, 500)}, {"HW": (612, 610)}]  # {"HW": (512, 512)}]


@pytest.fixture(ids=IMAGE_IDS, params=IMAGES)
def image_url(request) -> str:
    return request.param


WSI_IDS = ["s3tiff", "localtiff"]
WSIS = [
    "s3://project-histo/gigi/semicol_val_2pT4GFzn.ome.tiff",
    _env_var + "/4b77f015acc40c7e39470f3ebb658818.tiff",
]


@pytest.fixture(ids=WSI_IDS, params=WSIS)
def gigapixel_image(request) -> str:
    return request.param


CLF_VOLUME_IDS = ["luna_25"]
# 100012,1.2.840.113654.2.55.240231128564881525363489796879328810792,19990102,[...],1,100012_1_19990102,100012_1,1,61,Female
CLF_VOLUMES = [(_env_var + "/luna25_100012_1_19990102.nii.gz", "malignant")]

VOLUME_IDS = ["amos_HW", "brats_HWD"]
VOLUMES = [
    # Image, Mask, NumClasses, TaskName, VolumeInference
    (
        _env_var + "/amos_0008_image.nii.gz",
        _env_var + "/amos_0008_label.nii.gz",
        16,
        "amos22ctseg",
        "HW",  # the volume is not cubical
    ),
    (
        _env_var + "/BraTS20_Training_001_t1.nii.gz",
        _env_var + "/BraTS20_Training_001_seg.nii.gz",
        5,
        "brats2020seg",  # background, 4 classes
        "HWD",
    ),
]


@pytest.fixture(ids=VOLUME_IDS, params=VOLUMES)
def img_mask_3d(request) -> str:
    return request.param


# Teddy image with RLE segmentation and classification label
LABELED_SUBJECT_DIR = TEST_DATA_DIR / "data_interface" / "teddy_with_seg_clf"
