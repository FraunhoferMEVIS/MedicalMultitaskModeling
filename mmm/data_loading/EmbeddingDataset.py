import json
from typing import Any, Callable, TypeVar

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from mmm.data_loading.MTLDataset import MTLDataset, SrcCaseType


class EmbeddingDataset(MTLDataset):
    """
    Each case consists of an image with one or more embeddings.

    For some tasks it might need to be required to also include negatives.
    """

    def __init__(self, *args, batch_visualizer: Callable | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_visualizer = batch_visualizer

    @staticmethod
    def get_mandatory_keys() -> list[str]:
        return super(EmbeddingDataset, EmbeddingDataset).get_mandatory_keys() + ["image", "embeddings"]

    @staticmethod
    def get_optional_keys():
        return super(EmbeddingDataset, EmbeddingDataset).get_optional_keys() + ["negatives"]

    def verify_case(self, d: SrcCaseType) -> None:
        self.assert_image_data_assumptions(d["image"])
        assert "embeddings" in d, "EmbeddingDataset requires an 'embeddings' field"
        assert len(d["embeddings"].shape) == 2, f"Expected embeddings to have 2 dimensions, got {d['embeddings'].shape}"
        assert d["embeddings"].dtype == torch.float32, f"Expected embeddings to be float32, got {d['embeddings'].dtype}"
        if "negatives" in d:
            assert (
                len(d["negatives"].shape) == 2
            ), f"Expected negatives to have 2 dimensions, got {d['negatives'].shape}"
            assert (
                d["negatives"].dtype == torch.float32
            ), f"Expected negatives to be float32, got {d['negatives'].dtype}"

    @staticmethod
    def describe_emb(z):
        from mmm.logging.st_ext import blend_with_mask, stw

        return f"{z.shape}, mean: {z.mean():.2f}, std: {z.std():.2f}, min: {z.min():.2f}, max: {z.max():.2f}"

    def st_case_viewer(self, ls: list[dict[str, Any]], i: int = -1) -> None:
        from mmm.logging.st_ext import Image2D, M3Image, m3_image, st

        m3_image(
            data=M3Image.Data(
                images=[
                    Image2D.from_tensor(
                        img=d["image"],
                        desc=json.dumps(d["meta"], indent=2, default=str) if "meta" in d else None,
                        caption=self.describe_emb(d["embeddings"]),
                    )
                    for d in ls
                ],
            ),
            key=f"img{i}_original",
        )

    def _compute_batchsize_from_batch(self, batch: SrcCaseType) -> int:
        return len(batch)

    def _visualize_batch_case(self, batch: SrcCaseType, i: int) -> None:
        from mmm.logging.st_ext import blend_with_mask, stw

        blend_with_mask(batch["image"][i], None, st_key=f"image_{i}")
        z = batch["embeddings"][i]
        stw(self.describe_emb(z))
        if "negatives" in batch:
            stw("Negatives:" + self.describe_emb(batch["negatives"][i]))

        stw(batch.get("meta", [{}] * i)[i], st_prefix=f"case_{i}_meta_")
