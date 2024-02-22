from typing import Any, Callable, TypeVar

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from mmm.mtl_types import AnnotatedImage, RGBImage, ImageCaption
from mmm.data_loading.MTLDataset import MTLDataset, SrcCaseType


class ImageCaptionDataset(MTLDataset):
    """
    Each case consists of an image and a list of captions.
    """

    @staticmethod
    def batch_collater() -> Callable:
        """
        For detection tasks the common batch format seems to be a list of cases.
        By default, collate_fn overwriters work with a list of cases.
        """
        # No inputs are stacked, detection tasks want lists!
        batch_type = TypeVar("batch_type", bound=list[dict[str, Any]])

        def f(x: batch_type) -> batch_type:
            return x

        return f

    def __init__(self, src_ds: Dataset[SrcCaseType], collate_fn: Callable | None = None, *args, **kwargs) -> None:
        if collate_fn is not None:
            collate_fn = transforms.Compose([collate_fn, self.batch_collater()])
        else:
            collate_fn = self.batch_collater()
        super().__init__(src_ds, ["image", "captions"], ["meta"], collate_fn=collate_fn, *args, **kwargs)

    def verify_case(self, d: SrcCaseType) -> None:
        self.assert_image_data_assumptions(d["image"])
        assert "captions" in d, "ImageCaptionDataset requires a 'captions' field"
        assert isinstance(d["captions"], list), "ImageCaptionDataset requires a list of captions"

    def st_case_viewer(self, case: SrcCaseType, i: int = -1) -> None:
        from mmm.logging.st_ext import stw

        stw(AnnotatedImage.from_untyped(case))

    def _compute_batchsize_from_batch(self, batch: SrcCaseType) -> int:
        return len(batch)

    def _visualize_batch_case(self, batch: SrcCaseType, i: int) -> None:
        from mmm.logging.st_ext import stw

        stw(
            AnnotatedImage.from_untyped(
                {
                    "image": batch[i]["image"],
                    "captions": batch[i]["captions"],
                }
            )
        )
