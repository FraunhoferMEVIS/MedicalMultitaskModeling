import json
from pathlib import Path

import numpy as np
from m3_sdk.models import Subject
from m3_sdk.Repr import Repr

from mmm.app import label
from mmm.logging.st_ext import (
    Image2D,
    ImageOverlay,
    ImageOverlays,
    M3Image,
    download_zip_to_workdir,
    m3_image,
    m3_image_from_disk,
    st,
    st_batchselector,
    st_download_sidebar,
    st_groupselector,
    stw,
)
from mmm.settings import mtl_settings


def view_single_prediction(logzip: str, local_workdir: str = "/ephemeral/"):
    download_zip_to_workdir(logzip, workdir := Path(local_workdir))
    with st.expander(label="Local working directory"):
        st.success(f"{workdir.absolute()}")
    selected_group = st_groupselector([p for p in workdir.iterdir() if p.is_dir()])

    if (batch_json_path := workdir.joinpath("batch_info.json")).exists() and "group_id_to_index" in (
        batch_info := json.loads(batch_json_path.read_text())
    ):
        group_name = selected_group.name
        group_index = batch_info["group_id_to_index"][group_name]
        graphs = {key: np.array(batch_info["graphs"][key][group_index]) for key in batch_info["graphs"]}
    else:
        graphs = None

    m3_image_from_disk(st_batchselector(selected_group), graphs=graphs)


def predictions_viewer():
    workdir = st_download_sidebar()
    if (workdir is not None) and (group_folders := [p for p in workdir.iterdir() if p.is_dir()]):
        with st.sidebar:
            selected_group = st_groupselector(group_folders)

        m3_image_from_disk(st_batchselector(selected_group))


def readme():
    st.markdown("run `m3 --help` in a terminal for command line usage instructions.")
    st.markdown(Path(__file__).parent.parent.parent.joinpath("README.md").read_text())


def database_explorer():
    dataset_keys = [dname.decode() for dname in mtl_settings.kv.keys("datasets:*")]  # type: ignore
    dataset_key = st.selectbox(
        "Select dataset",
        index=None,
        options=dataset_keys,
        accept_new_options=True,
        placeholder=f"{mtl_settings.subj_prefix}:*",
        format_func=lambda x: f"{x} ({mtl_settings.kv.scard(x)} subjects)",
    )
    st.write(f"Selected dataset key: {dataset_key}")
    if dataset_key is None or dataset_key not in dataset_keys:
        subject_keys = [k.decode() for k in mtl_settings.kv.keys(f"{mtl_settings.subj_prefix}:*")]
    else:
        subject_keys = [f"{mtl_settings.subj_prefix}:{k.decode()}" for k in mtl_settings.kv.smembers(dataset_key)]

    if (subject_key := st.selectbox("Select subject", index=None, options=subject_keys)) is not None:
        st.write(f"Selected subject key: {subject_key}")
        stw(subject := Subject.model_validate_json(mtl_settings.kv.get(subject_key)))  # type: ignore

        if (
            repr_key := st.selectbox(
                "Select representation",
                index=None,
                options=[k.decode() for k in mtl_settings.kv.keys(f"repr:{subject.id}:*")],  # type: ignore
                format_func=lambda r: f"{r} ({len(mtl_settings.kv.hkeys(r))})",
            )
        ) is not None:
            reprs = [Repr.from_bytes(mtl_settings.kv.hget(repr_key, field)) for field in mtl_settings.kv.hkeys(repr_key)]  # type: ignore
            gt_keys = mtl_settings.kv.keys(f"gt:{subject.id}:*:rgbimage")
            if (
                gt_key := st.selectbox(
                    "Select label",
                    index=None,
                    options=[k.decode() for k in gt_keys],  # type: ignore
                    format_func=lambda k: f"{k.split(':')[-2]} ({mtl_settings.kv.hlen(k)})",  # type: ignore
                )
            ) is not None:
                from mmm.api.mtl_adapter import SegmentationAdapter

                gt_reprs = [Repr.from_bytes(mtl_settings.kv.hget(gt_key, field)) for field in mtl_settings.kv.hkeys(gt_key)]  # type: ignore
                all_classes = list(set([class_name for gt in gt_reprs for class_name in gt.meta["class_names"]]))
                stw(f"Classes in this label: {', '.join(all_classes)}")
                masks = [
                    SegmentationAdapter._build_label_for_class_names(
                        all_classes, gt_repr.tensor.long().squeeze(0), gt_repr.meta["class_names"]
                    )
                    for gt_repr in gt_reprs
                ]
            else:
                masks, all_classes = [None] * len(reprs), None
            # stw(repr_keys := mtl_settings.kv.keys(f"repr:{subject.id}:*"))
            images = []
            for repr, mask in zip(reprs, masks):
                image = Image2D.from_tensor(
                    repr.tensor,
                    masks=[mask == i for i in range(len(all_classes))] if all_classes is not None else None,
                    class_names=all_classes,
                    caption=f"<span style='color:orange'>{repr.meta.get('context', ())}</span>",
                    desc=json.dumps(repr.meta),
                )
                images.append(image)
            m3_image(key="demo_image", data=M3Image.Data(images=images))


if __name__ == "__main__":
    st.set_page_config(page_title="M3 Streamlit App", layout="wide", page_icon="🏥")

    if "logzip" in st.query_params:
        view_single_prediction(st.query_params["logzip"])
    else:
        pages = {
            "0_Readme": readme,
            "1_Predictions Viewer": predictions_viewer,
            "2_Database Explorer": database_explorer,
        }

        with st.sidebar:
            if mtl_settings.kv.ping():
                st.success("Connected to DB")
            else:
                st.error("Not connected to DB")
            page_names = sorted(list(pages.keys()))
            page = st.selectbox("Select a page", options=page_names, index=1 if "logzip" in st.query_params else 0)

        pages[page]()
