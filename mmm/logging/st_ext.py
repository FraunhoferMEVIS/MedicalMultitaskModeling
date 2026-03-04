"""
Utilities for streamlit. Use `stw` to visualize objects.

Importing this module will automatically set our recommended settings.
"""

import inspect
import json
import os
import random
import traceback
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, cast

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import streamlit as st
except ImportError:
    if not TYPE_CHECKING:
        st = None
    else:
        raise  # Happens during type checking and avoids false positives

from m3_sdk.DistributedPath import DistributedPath
from m3_sdk.geojson import get_colors
from m3_sdk.utils import rgbnumpy_to_base64

from mmm.data_loading.MTLDataset import DatasetStyle
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.DataSplit import DataSplit
from mmm.logging.st_m3image import Image2D, ImageOverlay, ImageOverlays, M3Image, m3_image
from mmm.settings import mtl_settings
from mmm.utils import remove_folder_blocking_if_exists

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
        st.info("For training, pixel range is expected to be normalized into [0, 1]!")
        im_channels_first = (im_channels_first - torch.min(im_channels_first)) / (
            torch.max(im_channels_first) - torch.min(im_channels_first)
        )
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

    cohort_names = list(cohorts.keys())
    cohort_name = st.sidebar.selectbox("Choose cohort", cohort_names)

    if cohort_name:
        stw(cohorts[cohort_name](), st_prefix=cohort_name)
    else:
        st.write(f"Select a Cohort")


def download_zip_to_workdir(zip_url: str, workdir: Path) -> None:
    remove_folder_blocking_if_exists(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    workdir.joinpath("temp.zip").write_bytes(DistributedPath(uri=zip_url).upath().read_bytes())
    with zipfile.ZipFile(workdir.joinpath("temp.zip"), "r") as zip_ref:
        zip_ref.extractall(workdir)
    workdir.joinpath("temp.zip").unlink()


def st_download_sidebar():
    with st.sidebar:
        workdir = Path(
            st.text_input(
                "Working directory",
                value="/ephemeral",
                help="Local path to working directory where logzips are extracted to",
            )
        )
        if not workdir.exists() and st.button("Create working directory"):
            workdir.mkdir(parents=True, exist_ok=True)

        if workdir.exists():
            numfiles = len(list(workdir.rglob("*")))
            st.markdown(f"Working directory `{workdir}` currently contains {numfiles} files")

            with st.form("zipload"):
                # Display recents that were saved to st.session_state["recent_logzips"]
                for zip_path in st.session_state.get("recent_logzips", []):
                    st.markdown(f"- {zip_path}")
                zip_path = st.text_input("From zip", value="")
                if st.form_submit_button("Load"):
                    # st.query_params.from_dict({"logzip": zip_path, "working_dir": str(workdir.resolve())})
                    st.session_state["recent_logzips"] = st.session_state.get("recent_logzips", []) + [zip_path]
                    download_zip_to_workdir(zip_path, workdir)
                    st.rerun()

            if zip_path:
                try:
                    st.download_button(
                        label="Download ZIP",
                        data=DistributedPath(uri=zip_path).upath().read_bytes(),
                        file_name=Path(zip_path).name if "/" in zip_path else "logzip.zip",
                        mime="application/zip",
                    )
                except Exception as e:
                    st.warning(f"Could not download zip: {e}")
    return workdir if (workdir and workdir.exists()) else None


def st_groupselector(group_folders: list[Path]):
    def display_option(p: Path):
        return f"{p.name} ({len(list(filter(Path.is_dir, p.iterdir())))} items)"

    selected_group = st.selectbox(
        "Select group folder",
        options=group_folders,
        format_func=display_option,
    )
    return selected_group


def st_batchselector(batch_path: Path):
    """
    Visualizes a batch that was already extracted to disk and can therefore be loaded from a Path.
    """
    item_folders = [item_folder for item_folder in batch_path.iterdir() if item_folder.is_dir()]
    batch_indices = [int(f.name.split("_")[-1]) for f in item_folders]

    df = pd.DataFrame(
        [
            {
                "thumbnail": rgbnumpy_to_base64(iio.imread(f.joinpath("input_image.jpg"))),
                "image_folder": str(f.absolute()),  # str because serialization otherwise fails
                "batch_index": batch_index,
            }
            for f, batch_index in zip(item_folders, batch_indices)
        ]
    )
    # sort by batch index
    df = df.sort_values(by="batch_index").reset_index(drop=True)

    st_df = st.dataframe(
        df,
        column_config={
            "thumbnail": st.column_config.ImageColumn("Preview Image", help="Streamlit app preview screenshots")
        },
        on_select="rerun",
        selection_mode="multi-row",
        row_height=50,
        # hide_index=True,
        # on_change=lambda: print('onchange')
    )
    # st.write(df.selection)
    return [df.iloc[row] for row in st_df.selection["rows"]]  # type: ignore


def heatmap_with_sums_chart(
    a: np.ndarray,
    *,
    row_title: str = "Row",
    col_title: str = "Column",
    scheme: str = "viridis",
    sum_label: str = "Sum",
):  # -> alt.LayerChart:
    """Build an Altair heatmap with an extra Sum row/column and annotated values."""
    import altair as alt

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Expected a square 2D array (n x n).")

    n = a.shape[0]

    # Use only original cells for the color scale (exclude sums).
    vmin = float(np.nanmin(a))
    vmax = float(np.nanmax(a))

    # Extend with Sum row/col
    row_sums = a.sum(axis=1)
    col_sums = a.sum(axis=0)
    grand_total = a.sum()
    ext = np.block(
        [
            [a, row_sums[:, None]],
            [col_sums[None, :], np.array([[grand_total]])],
        ]
    )

    row_labels = [str(i) for i in range(n)] + [sum_label]
    col_labels = [str(i) for i in range(n)] + [sum_label]

    df = pd.DataFrame(ext, index=row_labels, columns=col_labels)
    long = df.rename_axis("row").reset_index().melt(id_vars="row", var_name="col", value_name="value")
    long["is_sum"] = (long["row"] == sum_label) | (long["col"] == sum_label)

    base = alt.Chart(long).encode(
        x=alt.X("col:O", sort=col_labels, title=col_title),
        y=alt.Y("row:O", sort=row_labels, title=row_title),
    )

    # Split into two rect layers so the legend/scale reflects only non-sum cells.
    rect_values = (
        base.transform_filter("!datum.is_sum")
        .mark_rect()
        .encode(
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(
                    scheme=cast(Any, scheme),
                    domain=[vmin, vmax],
                    nice=True,
                ),
                legend=alt.Legend(
                    title="Value",
                    orient="bottom",
                ),
            ),
            tooltip=[
                alt.Tooltip("row:O", title="Row"),
                alt.Tooltip("col:O", title="Column"),
                alt.Tooltip("value:Q", title="Value"),
            ],
        )
    )

    rect_sums = (
        base.transform_filter("datum.is_sum")
        .mark_rect(color="#f2f2f2")
        .encode(
            tooltip=[
                alt.Tooltip("row:O", title="Row"),
                alt.Tooltip("col:O", title="Column"),
                alt.Tooltip("value:Q", title="Value"),
            ]
        )
    )

    text_values = (
        base.transform_filter("!datum.is_sum")
        .mark_text(baseline="middle", color="white")
        .encode(text=alt.Text("value:Q", format=".2f"))
    )

    text_sums = (
        base.transform_filter("datum.is_sum")
        .mark_text(baseline="middle", color="black", fontWeight="bold")
        .encode(text=alt.Text("value:Q", format=".2f"))
    )

    # Give the chart a real height so it can't collapse to ~0px.
    # (Still scales to container width via width="container".)
    min_h = 320
    target_h = 24 * (n + 1)
    chart_height = max(min_h, target_h)

    return (
        (rect_values + rect_sums + text_values + text_sums)
        .properties(
            width="container",
            height=chart_height,
            autosize=alt.AutoSizeParams(type="fit-x", contains="padding"),
        )
        .configure_view(stroke=None)
    )


def m3_image_from_disk(item_dicts: list[dict], graphs: dict[str, np.ndarray] | None = None):
    if graphs is not None:
        # for g in graphs:
        #     assert g[1].shape == (len(item_dicts), len(item_dicts))
        if (
            graph_name := st.selectbox(label="Select graph to visualize", options=["None"] + list(graphs.keys()))
        ) in graphs:
            st.altair_chart(
                heatmap_with_sums_chart(
                    graphs[graph_name],
                    row_title="From",
                    col_title="To",
                ),
                width="stretch",
            )

    images = []
    for item_dict in item_dicts:
        item_folder = Path(item_dict["image_folder"])
        overlay_meta = json.loads(item_folder.joinpath("overlays_meta.json").read_text())
        img2d = Image2D(
            img=rgbnumpy_to_base64(img_npy := iio.imread(item_folder.joinpath("input_image.jpg"))),
            overlay_groups=[
                ImageOverlays(
                    overlays=[
                        ImageOverlay(
                            data=rgbnumpy_to_base64(iio.imread(item_folder.joinpath(overlay_item["file_path"]))),
                            classname=overlay_item["class_name"],
                        )
                        for overlay_item in overlay_meta["overlay_categories"][overlay_category]
                    ],
                    overlay_type=overlay_category,
                )
                for overlay_category in overlay_meta["overlay_categories"]
            ],
        )

        img2d.caption = f"<span style='color:DarkSeaGreen'>{img_npy.shape}</span><br>"

        if item_folder.joinpath("meta.json").exists():
            img2d.desc = (item_meta_str := item_folder.joinpath("meta.json").read_text())
            item_meta = json.loads(item_meta_str)
            if "context" in item_meta:
                img2d.caption += f"Context: <span style='color:orange'>{item_meta['context']}</span><br>"

        if item_folder.joinpath("caption.txt").exists():
            img2d.caption += item_folder.joinpath("caption.txt").read_text()

        images.append(img2d)

    m3_image(
        key=f"image_{'_'.join([Path(x['image_folder']).name for x in item_dicts])}",
        data=M3Image.Data(
            images=images,
        ),
    )
