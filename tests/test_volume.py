from pathlib import Path

from einops import rearrange
from mmm.api.WorkerState import WorkerState
from mmm.mtl_modules.tasks.SemSegTask import SemSegTask
from mmm.transforms import ApplyToKey
from mmm.volume3d import SegMetric3D, Tomo3DProcessor, find_regions, render_regions_to_volume, Volume3DInference
from monai.transforms import LoadImage, EnsureChannelFirst
import numpy as np
import pytest
import torch

from tests.test_data import TEST_DATA_DIR, img_mask_3d


@pytest.fixture(scope="session")
def ws() -> WorkerState:
    from mmm.api.WorkerState import ws

    return ws


def test_loading_padding(img_mask_3d):
    img_path, mask_path, num_classes, task_name, volume_inference = img_mask_3d
    processor = Tomo3DProcessor(
        args=Tomo3DProcessor.Config(fix_orientation="never", divisable_by=(factors := (128, 128, 128))),
        augs_constructor=None,
        with_segmask=True,
    )
    img, _ = processor.image_loader(img_path)
    original_img_affine = img.meta["affine"].clone()
    img = EnsureChannelFirst()(img)
    assert ~(original_img_affine != img.meta["affine"]).any()

    mask, _ = processor.mask_loader(mask_path)
    assert ~(original_img_affine != mask.meta["affine"]).any()

    needs_padding = True in [
        img.shape[i + 1] % factor != 0 or mask.shape[i] % factor != 0 for i, factor in enumerate(factors)
    ]

    # Apply padding which should change the affine if the image was not already divisable
    out = processor({"image": img, "label": mask, "meta": {}})
    print(out)

    if needs_padding:
        assert (
            out["meta"]["monai_meta"]["image"]["affine"] != original_img_affine
        ).any(), "Affine should change after padding"
    for i, factor in enumerate(factors):
        assert out["image"].shape[i + 1] % factor == 0, f"Image shape {out['image'].shape} is not divisable by {factor}"
        assert out["label"].shape[i] % factor == 0, f"Mask shape {out['label'].shape} is not divisable by {factor}"


def test_volume_prediction_single(ws: WorkerState):
    img_path = TEST_DATA_DIR.joinpath("lung_074_image.nii.gz")
    inference = Volume3DInference()
    slice_generator, chwd, main_image_affine = Volume3DInference.load_volume(inference.tomo_processor, [img_path])
    print(slice_generator)
    results = inference(
        model=ws.fm, task=ws.fm["msdlung"], slices=list(slice_generator), chwd=chwd, main_image_affine=main_image_affine
    )

    metric = SegMetric3D(SegMetric3D.Config())
    mask_loader = LoadImage(dtype=np.int64, image_only=False, simple_keys=True)
    metrics = metric(
        y_pred=results["mask"].long().unsqueeze(0),
        y_true=mask_loader(TEST_DATA_DIR.joinpath("lung_074_label.nii.gz"))[0].unsqueeze(0),
        num_classes=2,
    )
    assert metrics["dice"].min() > 0.8


def test_volume_prediction_boxes(ws: WorkerState):
    def boxes_from_mask(mask: torch.Tensor, min_size: float = 25.0):
        from skimage.measure import label, regionprops

        labeled_array, num_features = label(mask, connectivity=1, return_num=True)
        res = []
        for region in regionprops(labeled_array):
            if region.area > min_size:
                res.append(
                    {
                        "box": region.bbox,
                        "size": region.area,
                        "class": "lesion",
                    }
                )
        return {
            "lesions": res,
        }

    def overlap_boxes(box1, box2):
        x_start_1, y_start_1, z_start_1, x_end_1, y_end_1, z_end_1 = box1
        x_start_2, y_start_2, z_start_2, x_end_2, y_end_2, z_end_2 = box2

        x_overlap = max(0, min(x_end_1, x_end_2) - max(x_start_1, x_start_2))
        y_overlap = max(0, min(y_end_1, y_end_2) - max(y_start_1, y_start_2))
        z_overlap = max(0, min(z_end_1, z_end_2) - max(z_start_1, z_start_2))
        return x_overlap > 0 and y_overlap > 0 and z_overlap > 0

    img_path = TEST_DATA_DIR.joinpath("lung_074_image.nii.gz")
    inference = Volume3DInference(instances="boxes")
    slice_generator, chwd, main_image_affine = Volume3DInference.load_volume(inference.tomo_processor, [img_path])
    print(slice_generator)
    results = inference(
        model=ws.fm,
        task=ws.fm["msdlung"],
        slices=list(slice_generator),
        chwd=chwd,
        main_image_affine=main_image_affine,
    )

    mask, mask_header = LoadImage(dtype=np.int64, image_only=False, simple_keys=True)(
        TEST_DATA_DIR.joinpath("lung_074_label.nii.gz")
    )
    gt_boxes = boxes_from_mask(mask.numpy())
    assert len(gt_boxes["lesions"]) == 1, "Expected one lesion in the ground truth mask"
    gt_box = gt_boxes["lesions"][0]["box"]

    assert True in [overlap_boxes(gt_box, b["box"]) for b in results["boxes"]]
