import logging
import random
import pytest
import torch
from mmm.api.models import GigapixelImage, MSubject, extract_instances_gigapixel
from mmm.nic_utility import WSIWindows

from tests.test_data import gigapixel_image, WSIS


def test_repr_extaction_image(gigapixel_image):
    reprs = list(extract_instances_gigapixel(GigapixelImage(url=gigapixel_image, max_instances=5)))
    assert len(reprs) == 5


def test_repr_extraction_subject():
    img1 = GigapixelImage(url=WSIS[0], max_instances=5)
    img2 = GigapixelImage(url=WSIS[1], max_instances=5)
    subj = MSubject(id=0, data={"image": [img1, img2]})
    reprs = list(subj.extract_instances(subj))
    assert len(reprs) == 10


def test_openslide_integration(gigapixel_image):
    from tiffslide import TiffSlide

    try:
        import openslide
    except ImportError:
        pytest.skip("openslide not installed")

    wsi = GigapixelImage(url=gigapixel_image, levels=[2])
    tiffslide_slide = TiffSlide(wsi.url.file().open())
    wsi.load_sample_geojson(tiffslide_slide)
    list_levels_hw_sizes = list(wsi.build_window_gen(tiffslide_slide))

    windows_tiffslide = WSIWindows(list_levels_hw_sizes, gigapixel_image)
    windows_openslide = WSIWindows(list_levels_hw_sizes, gigapixel_image)
    if wsi.url.get_protocol() != "file":
        # Expect Exception with pytest
        with pytest.raises(Exception):
            windows_openslide.slide_instance = WSIWindows.open_slide(gigapixel_image, force_openslide=True)
    else:
        windows_openslide.slide_instance = WSIWindows.open_slide(gigapixel_image, force_openslide=True)

        assert isinstance(windows_openslide.slide_instance, openslide.OpenSlide)

        random_index = random.randint(0, len(windows_tiffslide) - 1)
        patch_tiffslide, ctx_tiffslide = windows_tiffslide[random_index]
        assert isinstance(windows_tiffslide.slide_instance, TiffSlide)
        patch_openslide, ctx_openslide = windows_openslide[random_index]
        assert ctx_tiffslide == ctx_openslide
        assert torch.max(patch_tiffslide["image"]) == torch.max(patch_openslide["image"])
        assert patch_tiffslide["image"].shape == patch_openslide["image"].shape

        import timm

        # import numpy as np
        # from PIL import Image
        # pil_patch_tiffslide = Image.fromarray(np.uint8(patch0_tiffslide["image"].permute(1, 2, 0).numpy() * 255))
        # pil_patch_tiffslide.save(f"{wsi.url.get_filestem_suggestion()}_tiffslide.png")
        # pil_patch_openslide = Image.fromarray(np.uint8(patch0_openslide["image"].permute(1, 2, 0).numpy() * 255))
        # pil_patch_openslide.save(f"{wsi.url.get_filestem_suggestion()}_openslide.png")

        with torch.inference_mode():
            model = timm.create_model("convnext_femto", pretrained=True, features_only=True).eval()
            # Use the last feature map for the comparison (still has a small spatial resolution)
            repr_tiffslide = model(patch_tiffslide["image"].unsqueeze(0))[-1][0]
            repr_openslide = model(patch_openslide["image"].unsqueeze(0))[-1][0]

            # Count the number of non-zero elements in the difference
            diff = torch.abs(repr_tiffslide - repr_openslide) < 1e-3
            number_correct, number_incorrect = diff.sum().item(), (~diff).sum().item()
            if number_incorrect > 0:
                logging.warning(f"openslide and tiffslide representations differ in {number_incorrect} elements")
            assert number_correct > number_incorrect, f"{number_correct=}, {number_incorrect=}"
