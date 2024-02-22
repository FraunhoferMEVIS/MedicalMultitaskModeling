"""
Utilities for streamlit. Use `stw` to visualize objects.

Importing this module will automatically set our recommended settings.
"""

import inspect
from typing import Optional, List, Callable, Dict, Tuple, Any
import traceback
import os
from functools import partial
from tqdm import tqdm
import random
import torch
import numpy as np
import cv2

try:
    import streamlit as st
except ImportError:
    st = None

from torch.utils.data import Dataset, DataLoader
from mmm.data_loading.MTLDataset import MTLDataset, DatasetStyle
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.DataSplit import DataSplit
from mmm.typing_utils import get_colors
from mmm.settings import mtl_settings

if st is not None:
    st.set_page_config(layout="wide")


def blend_with_mask(
    im_channels_first: torch.Tensor,
    mask_no_channels: Optional[torch.Tensor],
    caption_suffix: str = "",
    classes: Optional[List[str]] = None,
    st_key="default_blend_with_masks_key",
    boxes: Optional[Tuple[torch.FloatTensor, torch.LongTensor]] = None,
) -> None:
    """
    Shows images like they are expected by our neural networks.

    Images:
    - 2D: [C, width, height]; dtype: float32
    - 3D: [C, width, height, slices]: dtype: float32

    Mask:
    - 2D: [width, height]; dtype: long
    - 3D: [width, height, slices]; dtype: long
    """
    import streamlit as st

    caption = f"Shape: {tuple(im_channels_first.shape)}, "
    caption += f"Min: {torch.min(im_channels_first):.2f}, Max: {torch.max(im_channels_first):.2f}"

    if len(im_channels_first.shape) == 4:
        # 3D!
        slider_key = f"{st_key}_sliceslider"

        if slider_key not in st.session_state:
            st.session_state[slider_key] = random.randint(0, im_channels_first.shape[-1])

        st.slider(
            label="Select slice of image",
            min_value=0,
            max_value=im_channels_first.shape[-1] - 1,
            key=slider_key,
        )
        im_channels_first = im_channels_first[..., st.session_state[slider_key]]

        if mask_no_channels is not None:
            mask_no_channels = mask_no_channels[..., st.session_state[slider_key]]

    if torch.max(im_channels_first) > 1.0 or torch.min(im_channels_first) < 0.0:
        st.error("Pixel range is expected to be normalized into [0, 1]. Applying auto fix")
        im_channels_first = im_channels_first / 255.0
    im = im_channels_first.numpy().astype(np.float32)

    # Printing images happens channels last, while the library uses channels first
    im = np.moveaxis(im, 0, -1)

    # Writing annotations on top of the image without copying first can lead to errors
    im = im.copy()

    # For printing a color image is required. Scale the channels to 3
    if im.shape[-1] == 1:
        im = np.concatenate([im] * 3, axis=-1)

    if classes is None:
        classes_ind_name = [
            (mtl_settings.ignore_class_value, "Unlabeled"),
            (0, "BG"),
            (1, "FG"),
        ]
    else:
        classes_ind_name = [(mtl_settings.ignore_class_value, "Unlabeled")] + [(i, c) for i, c in enumerate(classes)]
    colors = get_colors(len(classes_ind_name))

    if mask_no_channels is not None:
        assert len(im_channels_first.shape) - len(mask_no_channels.shape) == 1, "Image needs channels, mask not"

        # Count the pixels of each class in the mask
        class_indices, class_counts = torch.unique(mask_no_channels, return_counts=True)
        for i, c in classes_ind_name:
            if i in class_indices:
                num_pixels = class_counts[class_indices == i][0]
            else:
                num_pixels = 0

            if num_pixels > 0:
                colorrgb = f"{int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)}"
                col1, col2 = st.columns(2)
                with col1:
                    st.checkbox(
                        c,
                        key=f"{st_key}_{c}",
                        value=(len(classes_ind_name) == 1) or (i >= 0),
                    )
                with col2:
                    st.markdown(
                        f"<span style='color: rgb({colorrgb})'>{num_pixels} pixels</span>",
                        unsafe_allow_html=True,
                    )

            # Display the color of the class using markdown
            # st.markdown(f"<span style='color: rgb({colorrgb})'>asdf</span>", unsafe_allow_html=True)

        mask_no_channels = mask_no_channels.numpy()
        # color_overlay = np.zeros((mask_no_channels.shape[0], mask_no_channels.shape[1], 3))
        color_overlay = im.copy()
        for class_i, class_name in classes_ind_name:
            if f"{st_key}_{class_name}" in st.session_state and st.session_state[f"{st_key}_{class_name}"]:
                color_overlay[mask_no_channels == class_i] = np.asarray(colors[class_i], dtype=np.float32)

        # For each class, a binary mask is added in a distinct color
        # masks = []
        # for class_i, class_name in enumerate(classes):
        #     if st.session_state[f"{st_key}_{class_name}"]:
        #         binstr = f"{class_i:03b}"
        #         st.write(f"{class_name}, {binstr}")
        #         cc = []
        #         for c in binstr:
        #             if c == "1":
        #                 m_c = np.zeros_like(mask_no_channels)
        #                 m_c[mask_no_channels == class_i] = 1
        #                 cc.append(m_c)
        #             else:
        #                 cc.append(np.zeros_like(mask_no_channels))
        #         m = np.stack(cc, axis=-1).astype(np.float32)  # type: ignore
        #         # st.image(m)
        #         masks.append(m)

        # mm: np.ndarray = sum(masks) if masks else np.zeros_like(im)      # type: ignore

        im = cv2.addWeighted(im, 0.7, color_overlay.astype(np.float32), 0.3, 0)

    # Detection boxes
    if boxes is not None:
        boxes_tensor, box_labels = boxes
        for i, box in enumerate(boxes_tensor.tolist()):
            box_label: int = int(box_labels[i].item())
            thickness = max(1, im.shape[0] // 500)
            cv2.rectangle(
                im,
                list(map(int, box[:2])),
                list(map(int, box[2:])),
                colors[box_label],
                thickness,
            )

            fontscale: float = (im.shape[1] + im.shape[2]) / 1000
            fontsize_px: int = int(fontscale * 40)
            box_width = box[2] - box[0]
            chars_fitting = int(box_width / fontsize_px) * 2
            box_text = classes[box_label][:chars_fitting]
            cv2.putText(
                im,
                text=box_text,
                org=(int(box[0]), int(box[1]) + fontsize_px),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                color=colors[box_label],
                fontScale=fontscale,
            )

            # cv2 putText to write the label of the box
            # cv2.putText(im, f"{box_label}", (box[0], box[1]),
            #             cv2.FONT_HERSHEY_SIMPLEX, thickness, colors[box_label], thickness=thickness)

    st.image(im, caption=f"{caption}\n{caption_suffix}", clamp=True)


def _cohort_explorer(cohort: TrainValCohort) -> None:
    # Streamlit would otherwise resample cross validation splits
    import streamlit as st

    cohort.prepare_epoch(epoch=0)

    if split_name_selection := st.sidebar.selectbox("TrainOrVal", ["Training", "Validation"]):
        split_name: str = split_name_selection
    else:
        split_name: str = "Training"
    train_val_index = 0 if split_name == "Training" else 1

    def dataset_explorer():
        ds = cohort.datasets[train_val_index]
        st.title(f"{split_name} dataset:")

        if ds.get_dataset_style() is DatasetStyle.MapStyle:
            dataset_len = len(ds)  # type: ignore
            case_index = st.slider("select case", max_value=dataset_len - 1)
            case = ds.get_untransformed_case(case_index)
            ds.st_case_viewer(case, case_index)

            if st.button(label="verify all cases"):

                class VerifyingDataset(Dataset):
                    def __init__(self, src):
                        self.src = src

                    def __len__(self):
                        return len(self.src)

                    def __getitem__(self, i: int):
                        try:
                            case = self.src[i]
                            assert self.src.verify_case(case), f"case {case} could not be verified"
                            return case
                        except Exception as e:
                            print(i)
                            print(e)
                            st.write(i)
                            st.write(e)
                            traceback.print_exc()

                prog_bar = st.progress(0.0)
                st.write(f"Verifying with {len(os.sched_getaffinity(0))} workers")
                dl = DataLoader(
                    VerifyingDataset(ds),
                    shuffle=True,
                    num_workers=len(os.sched_getaffinity(0)),
                    batch_size=128,
                    collate_fn=lambda ls: ls,
                )
                for i, _ in enumerate(tqdm(dl)):
                    prog_bar.progress(i / (dataset_len // 128))
                st.balloons()
        else:
            with st.form("iterator demo"):
                max_items = int(st.number_input("Max iterations", step=1, value=10))
                display_every = int(st.number_input("Display every N case", step=1, value=2))
                submitted = st.form_submit_button("Reload iterator")

            # @st.cache(allow_output_mutation=True)
            # def cache_ds_iterator():
            #     return iter(ds)
            if submitted:
                for i, case in enumerate(ds):
                    ds.verify_case(case)
                    # case = next(iter(ds))
                    if i % display_every == 0:
                        ds.st_case_viewer(case, i)
                    if i > max_items:
                        break

    def batch_explorer():
        ds = cohort.datasets[train_val_index]
        test_batch = cohort.get_random_batch(DataSplit.from_index(train_val_index))
        ds.visualize_batch(test_batch)

    pages = {"Dataset explorer": dataset_explorer, "Batch explorer": batch_explorer}

    demo_name = st.sidebar.selectbox("Choose", list(pages.keys()))
    if demo_name is not None:
        pages[demo_name]()
    else:
        st.write("Select a page")


def side_by_side(img_1: torch.Tensor, img_2: torch.Tensor):
    """
    Shows two images side by side.
    Only implemented for 2D images [C,W,H], dtype: float32
    """
    import streamlit as st

    if torch.max(img_1) > 1.0 or torch.min(img_1) < 0.0:
        st.error("Pixel range is expected to be normalized into [0, 1]. Applying auto fix to first image")
    if torch.max(img_2) > 1.0 or torch.min(img_2) < 0.0:
        st.error("Pixel range is expected to be normalized into [0, 1]. Applying auto fix to second image")

    img_1: np.ndarray = img_1.numpy().astype(np.float32)
    img_2: np.ndarray = img_2.numpy().astype(np.float32)

    img_1 = np.moveaxis(img_1, 0, -1)
    img_2 = np.moveaxis(img_2, 0, -1)
    divider = np.zeros(img_1.shape)[:, :15, :]
    # Concatenating two images side by side

    img = np.concatenate((img_1, divider, img_2), axis=1)

    st.image(
        img,
        caption=f"two views of the same image. each of shape {img_1.shape}",
        clamp=True,
    )


def stw(obj: Any, st_prefix: str = "") -> None:
    """
    Takes an object and tries to visualize it.

    If the package did not define a special rule for that object we call streamlit's write.
    """
    if hasattr(obj, "_st_repr_"):
        # If the object's st_repr has a keyword parameter st_prefix, use the global counter
        if "st_prefix" in inspect.signature(obj._st_repr_).parameters:
            st_repr = obj._st_repr_(st_prefix=st_prefix)
        else:
            st_repr = obj._st_repr_()
        if st_repr is not None:
            st.markdown(
                st_repr,
                unsafe_allow_html=True,
            )
        return None
    return st.write(obj)


def multi_cohort_explorer(cohorts: Dict[str, Callable[[], TrainValCohort]]):
    # please use chromium-based browsers for correct image display
    # torch.manual_seed(0)
    # np.random.seed(0)
    # random.seed(0)
    import streamlit as st

    cohort_name = st.sidebar.selectbox("Choose cohort", list(cohorts.keys()))

    if cohort_name:
        stw(cohorts[cohort_name]())
    else:
        st.write(f"Select a Cohort")
