from __future__ import annotations

from torch.utils.data import Dataset

from .MTLDataset import MTLDataset, SrcCaseType


class RegressionDataset(MTLDataset):
    def __init__(self, src_ds: Dataset[SrcCaseType], *args, **kwargs) -> None:
        super().__init__(src_ds, *args, **kwargs)

    @staticmethod
    def get_mandatory_keys() -> list[str]:
        return super(RegressionDataset, RegressionDataset).get_mandatory_keys() + ["image", "target"]

    def verify_case(self, case):
        super().verify_case(case)
        self.assert_image_data_assumptions(case["image"])
        # targets should be of type float
        assert isinstance(case["target"], float), "Target should be of type float"

        if "meta" in case and "event" in case["meta"]:
            assert isinstance(case["meta"]["event"], int), "Event should be of type int"
            assert case["meta"]["event"] in [0, 1], "Event should be either 1 if event happened or 0 if not"

    def get_input_output_tuple(self, batch: Dict[str, Any]) -> Tuple[Any, ...]:
        return batch["image"], batch["target"]

    def st_case_viewer(self, case: Dict[str, Any], index: int = -1) -> None:
        from mmm.logging.st_ext import blend_with_mask, st

        st.write(f"Target: {case['target']}")
        im = case["image"]
        blend_with_mask(im, None, caption_suffix=f"Shape: {im.shape}", st_key=f"c{index}")
        st.write(case)

    def _compute_batchsize_from_batch(self, batch: Dict[str, Any]) -> int:
        return batch["image"].shape[0]

    def _visualize_batch_case(self, batch: Dict[str, Any], i: int) -> None:
        from mmm.logging.st_ext import blend_with_mask, st

        patch = batch["image"][i]
        st.write(f"Target: {batch['target'][i]}")
        blend_with_mask(
            patch,
            None,
            caption_suffix=f"{i}/{self._compute_batchsize_from_batch(batch)}: {patch.shape}",
            st_key=f"b{i}",
        )
