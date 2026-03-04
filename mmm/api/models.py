from __future__ import annotations

import itertools
import json
import logging
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Generator

import logfire
import numpy as np
import torch
from m3_sdk.DistributedPath import DistributedPath
from m3_sdk.models import Annotation as SDKAnnotation
from m3_sdk.models import BaseResult, GeoMask
from m3_sdk.models import GigapixelImage as SDKGigapixelImage
from m3_sdk.models import Prediction, Result
from m3_sdk.models import Subject as SDKSubject
from m3_sdk.models import Volume3DBox, Volume3DImage, Volume3DMask
from m3_sdk.Repr import Repr

from mmm.api.utils import download_image
from mmm.data_loading.geojson.GeojsonRegionWindows import GeojsonRegionWindows
from mmm.mmm_types.LabelType import LabelType

# moved to SDK, but for backwards still used here (otherwise the imports would be removed for not being used)
BACKWARDS_COMPAT_TYPE = BaseResult, Result, Volume3DBox, Prediction


class GigapixelImage(SDKGigapixelImage):
    extraction: GeojsonRegionWindows = GeojsonRegionWindows(  # type: ignore (M3 version needs to be used, not SDK)
        min_area=0.5,  # only include instances that are at least 50% inside the annotated region
        coordinate_augmentation="none",
        windowsize_augmentation="none",
        augmentation_strength=0.0,
        stepsize=1,
        patch_size=(448, 448),
        shuffle_coordinates_and_levels=True,
    )

    def load_sample_geojson(self, for_slide=None):
        if isinstance(self.sample_geojson, dict):
            return self

        if isinstance(self.sample_geojson, str):
            self.sample_geojson = json.loads(Path(self.sample_geojson).read_text())
        elif isinstance(self.sample_geojson, DistributedPath):
            self.sample_geojson = json.loads(self.sample_geojson.upath().read_text())
        elif self.sample_geojson is None:
            from m3_sdk.geojson import GeoAnno, create_featurecollection

            assert for_slide is not None, f"Either {self.sample_geojson=} or {for_slide=} must be given."
            rect = GeoAnno.rectangle_builder(left_upper=(0, 0), right_lower=for_slide.level_dimensions[0])
            self.sample_geojson = create_featurecollection([rect])

        return self

    def fillin_tilesize_if_none(self, slide=None):
        if self.extraction.patch_size is None:
            self.extraction.patch_size = self.determine_best_tile_edge_length(level=self.levels[0], slide=slide)
            all_patchsizes = [self.determine_best_tile_edge_length(level=l, slide=slide) for l in self.levels]
            if len(set(all_patchsizes)) != 1:
                logging.warning(f"Tile sizes for different levels are not equal: {all_patchsizes=}")
        return self

    def build_window_gen(self, for_slide):
        from mmm.nic_utility import WSIWindows

        return WSIWindows.build_window_gen(
            for_slide, self.extraction, self.levels, self.sample_geojson, self.max_instances
        )

    def determine_best_tile_edge_length(self, level: int, slide=None):
        from mmm.nic_utility import WSIWindows

        slide = WSIWindows.open_slide(self.url) if slide is None else slide

        try:
            return (
                slide.properties[f"tiffslide.level[{level}].tile-height"],
                slide.properties[f"tiffslide.level[{level}].tile-width"],
            )
        except KeyError:
            logging.warning(f"Could not determine tile size for {self.url}. Using default tile size of 224.")
            return (224, 224)

    def _st_repr_(self, st_prefix=""):
        from m3_sdk.geojson import create_featurecollection
        from streamlit_wsi_viewer import WSIViewerSettings as sett
        from streamlit_wsi_viewer import streamlit_wsi_viewer as st_viewer

        from mmm.logging.st_ext import st, stw
        from mmm.nic_utility import WSIWindows

        st_viewer(
            str(self.url.upath()),
            sett(
                viewerOptions={"showFullScreenButton": True},
                annotationLayerOptions={"asPixelCoordinates": True},
                outputSettings={"asPixelCoordinates": True},
            ),
            annotations=json.loads(Path(self.sample_geojson).read_text()),
        )

        with st.form(key=f"{st_prefix}extract_coords"):
            num_coords = st.number_input("Number of coordinates", value=self.max_instances)
            submit = st.form_submit_button("Show extraction coordinates")
        if submit:
            self.max_instances = num_coords
            insts = [d for d in self.extract_instances(MSubject(data={"image": self}), "image")]

            # Build a geojson with rectangles for visualization
            from mmm.data_loading.geojson import GeoAnno

            rects = []
            for inst in insts:
                level, (row, col) = inst["meta"]["level"], inst["meta"]["row_col"]
                stretch_factor = WSIWindows.open_slide(self.url).level_downsamples[level]
                h, w = inst["image"].shape[-2:]
                h, w = h * stretch_factor, w * stretch_factor
                rect = GeoAnno.rectangle_builder(left_upper=(col, row), right_lower=(col + w, row + h))

                rects.append(rect)

            st_viewer(
                str(self.url.upath()),
                sett(
                    viewerOptions={"showFullScreenButton": True},
                    annotationLayerOptions={"asPixelCoordinates": True},
                    outputSettings={"asPixelCoordinates": True},
                ),
                annotations=create_featurecollection(rects),
                # key=f"{st_prefix}extracted_coords_viewer",
            )

            stw(insts[0])
            stw(f"sampled {len(insts)} instances")


class Annotation(SDKAnnotation):
    def extract_gt(
        self, label_name: str, subject: MSubject, labeling, for_instance: Repr | None, adapter_cfg=None
    ) -> Repr:
        """
        Annotation object store a list of results and can extract the most common types of ground truths
        for machine learning tasks from them. Multiple results might result in a single ground truth.

        For example, a multiclass segmentation can be represented by multiple binary segmentations.
        """
        from mmm.api.mtl_adapter import ADAPTERS, LabelingConfig

        all_results = [result for result in self.result if result["from_name"] == label_name]
        # Filter results by their item_index, in which case the gt is only for a specific instance
        results = []
        for result in all_results:
            if result.item_index is not None:
                assert for_instance is not None, f"Instance must be given for {result=} to filter by item_index"
                if result.item_index == for_instance.meta.get("item_index", 0):
                    results.append(result)
            else:
                # If no item_index is given, the result is relevant for the whole subject
                results.append(result)
        if not results:
            return None

        # At this point, results only contain the results for the requested label
        # And all results have the same label type
        label_type: LabelType = LabelType.from_string(LabelingConfig.determine_type_of_label(results[0]))
        adapter = ADAPTERS[label_type]
        res = adapter.build_gt_repr(
            adapter.Config() if adapter_cfg is None else adapter_cfg,
            results=results,
            label_name=label_name,
            subject=subject,
            for_instance=for_instance,
            labeling=labeling,
        )
        res.meta["label_type"] = label_type.name
        return res


ImageType = str | Volume3DImage | GigapixelImage


def extract_instances_image(
    value, s: MSubject | None = None, input_tag: str = "", extra_ctx: tuple = (), with_labels=None
) -> Generator[Repr, None, None]:
    import torchvision.transforms.functional as F

    res = Repr(tensor=F.to_tensor(download_image(value)), meta={"context": "..."})

    if extra_ctx:
        res.meta["context"] = extra_ctx

    yield res


def extract_instances_gigapixel(
    value, s: MSubject | None = None, input_tag: str = "", extra_ctx: tuple = (), with_labels=None, for_slide=None
) -> Generator[Repr, None, None]:
    """
    - A gigapixel image is stored in pyramidical format.
    - Often, specific tiles can be accessed the fastest. If no tile edgelength is given, those tiles are extracted.
    """
    from m3_sdk.geojson import GeoAnno, create_featurecollection
    from rasterio.features import rasterize
    from shapely import GeometryCollection, clip_by_rect, coverage_union, geometry
    from shapely.affinity import scale, translate
    from shapely.geometry import shape

    from mmm.nic_utility import WSIWindows

    if with_labels and s and (anno := s.get_last_updated_annotation()):
        logfire.info("Found {annotation} for {labels}", annotation=anno, labels=with_labels)
        masks_for_labels = [r for r in anno.result if r.from_name in with_labels if isinstance(r, GeoMask)]
        assert len(masks_for_labels) == 1, f"Implement support for multiple GeoMasks per gigapixel image!"
        segmentation_geojson: GeoMask.FeatureCollection = masks_for_labels[0].value

        shapes = [(shape(feat), feat["properties"]["classification"]["name"]) for feat in segmentation_geojson.features]
    else:
        masks_for_labels = []

    for_slide = WSIWindows.open_slide(value.url) if for_slide is None else for_slide

    if value.extraction.patch_size is None:
        value.fillin_tilesize_if_none()

    if value.sample_geojson is not None:  # set by user
        value.load_sample_geojson(for_slide=for_slide)
        sample_geojson = value.sample_geojson
    elif masks_for_labels:  # sample for segmentation masks
        union_shape = coverage_union(*[x[0] for x in shapes])  # Create a MultiPolygon from all shapes
        sample_geojson = create_featurecollection([GeoAnno({"geometry": geometry.mapping(union_shape)})])
    else:  # fallback to full slide
        value.load_sample_geojson(for_slide=for_slide)
        sample_geojson = value.sample_geojson

    # value.build_window_gen(for_slide)
    window_gen = WSIWindows.build_window_gen(
        for_slide, value.extraction, value.levels, sample_geojson, value.max_instances
    )

    window_ds = WSIWindows(window_gen, value.url)

    for slice, level_hw_edges in window_ds:
        next_slice = Repr(
            tensor=slice["image"],
            meta={k: v for k, v in slice.items() if k != "image"},
        )
        next_slice.meta["image"] = value.url.uri
        next_slice.meta["context"] = level_hw_edges + extra_ctx
        next_slice.meta["downsample_fac"] = for_slide.level_downsamples[level_hw_edges[0]]
        if masks_for_labels:
            next_slice._buffer = next_slice._buffer or {}
            next_slice._buffer["geomask_shapes"] = shapes
        yield next_slice


def extract_instances_volume3d(
    value, s: SDKSubject | None, input_tag: str = "", extra_ctx: tuple = (), with_labels: list[str] | None = None
):
    """
    >>> from tests.test_data import VOLUMES; from m3_sdk.models import Volume3DImage, Subject
    >>> s = Subject(data={"ct": Volume3DImage(url=VOLUMES[0][0])})
    >>> slices = extract_instances_volume3d(s.data["ct"], s)
    >>> first_slice = next(slices)
    >>> assert first_slice.tensor.shape[0] == 3, "Extracts 3 channels"
    >>> assert first_slice.meta["context"][0] == 0, "Extracts the slice index as the first context element"
    """
    with_segmask: None | Volume3DMask = None
    if with_labels and s and (anno := s.get_last_updated_annotation()):
        results_for_labels = [r for r in anno.result if r.from_name in with_labels if isinstance(r, Volume3DMask)]
        assert len(results_for_labels) <= 1, f"Found {len(results_for_labels)=}, multiple masks not implemented"
        with_segmask = results_for_labels[0] if results_for_labels else None
    from monai.transforms.spatial.array import ResampleToMatch

    from mmm.volume3d import Tomo3DProcessor

    tomo_processor = Tomo3DProcessor(
        Tomo3DProcessor.Config(
            rotate_3d="never" if value.rotate is None else value.rotate,
            fix_orientation="never" if value.fix_orientation is None else value.fix_orientation,
            divisable_by=value.divisable_by,
            spacing=value.resample_to_spacing,
        ),
        None,
        with_segmask is not None,
    )

    main_url = value.url[0] if isinstance(value.url, list) else value.url
    secondary_urls = value.url[1:] if isinstance(value.url, list) else None

    with logfire.span("Loading image data for {subject_id}", subject_id=s.id if s is not None else None):
        arr, header = tomo_processor.image_loader(main_url.file().path)
        arr = arr.unsqueeze(0)
        logfire.debug("Loaded image shape: {shape}", shape=arr.shape)

        # Hack to fix Mevislab nifti files, only works for single images
        if len(arr.shape) > 4:
            logfire.warning(
                "Reshaping {old_shape} to {new_shape} {main_url}",
                old_shape=arr.shape,
                new_shape=arr[..., 0, 0, 0].shape,
                main_url=main_url,
            )
            arr = arr[..., 0, 0, 0]
        if len(arr.shape) < 4:  # or arr.shape[3] == 1:
            logfire.warning("Skipping {main_url} because of shape {shape}", main_url=main_url, shape=arr.shape)
            return None
        case_dict: dict[str, Any] = {"image": arr}

        if secondary_urls is not None:
            secondary_images = []
            for secondary_url in secondary_urls:
                sec_image, sec_header = tomo_processor.image_loader(secondary_url.file().path)
                sec_image = sec_image.unsqueeze(0)
                logfire.debug("Loaded secondary image shape: {shape}", shape=sec_image.shape)
                if sec_image.shape != arr.shape:
                    sec_image = ResampleToMatch()(img=sec_image, img_dst=arr)
                secondary_images.append(sec_image)
            case_dict = {"image": [arr] + secondary_images}

    if with_segmask is not None:
        mask_arr = tomo_processor.mask_loader(with_segmask.value.file().path)[0]
        case_dict["label"] = mask_arr

    processed_case = tomo_processor(case_dict)
    if secondary_urls is not None:
        processed_affine = processed_case["meta"]["monai_meta"]["image0"]["affine"].tolist()
    else:
        processed_affine = processed_case["meta"]["monai_meta"]["image"]["affine"].tolist()

    for i, slice in enumerate(tomo_processor.extract_slices(processed_case)):
        repr = Repr(tensor=tomo_processor.repeat_channels(slice["image"]), meta=slice.get("meta", {}))
        repr.meta["affine"] = processed_affine
        if s is not None:
            repr.meta["msubject"] = s
        repr.meta["context"] = slice["meta"]["context"] + extra_ctx

        # If the mask should be extracted add a pointer to the mask to the representation within the private attr
        if with_segmask is not None:
            repr._buffer = {} if repr._buffer is None else repr._buffer
            repr._buffer[with_segmask.from_name] = processed_case["label"]  # type: ignore

        yield repr


Extractors: dict[
    type[ImageType], Callable[[Any, MSubject, str, tuple, list[str] | None], Generator[Repr, None, None]]
] = {
    str: extract_instances_image,
    Volume3DImage: extract_instances_volume3d,
    GigapixelImage: extract_instances_gigapixel,
}


def _extract_instances_for_input(
    s: MSubject, labeling, input_tag: str, with_labels: list[str] | None = None
) -> Generator[Repr, None, None]:
    """
    Each input element can be converted into a single or multiple instances.
    """

    def yield_with_itemindex(generator, item_index):
        for repr in generator:
            repr.meta["item_index"] = item_index
            yield repr

    input_values = s.data[input_tag]
    if is_single_data := not isinstance(input_values, list):
        input_values = [input_values]

    gs = []
    for i, v in enumerate(input_values):
        # Some tasks might want to use positional encoding which can use this context
        extra_ctx = tuple() if is_single_data else (i,)  # (f"{input_tag}_{i}",)

        if s.meta is not None and "contexts" in s.meta:
            extra_ctx += tuple(ctx) if isinstance(ctx := s.meta["contexts"][i], (tuple, list)) else (ctx,)

        g = Extractors[type(v)](v, s, input_tag, extra_ctx, with_labels)
        # if isinstance(v, str):
        #     g = t._extract_instances_image(t, v, extra_ctx=extra_ctx, with_labels=with_labels)
        # elif isinstance(v, (GigapixelImage, Volume3DImage)):
        #     g = v.extract_instances(s, input_tag, extra_ctx=extra_ctx, with_labels=with_labels)
        # else:
        #     raise NotImplementedError(f"{i=}: Input type of {v} not implemented")
        gs.append(g if is_single_data else yield_with_itemindex(g, i))

    # Combine the generators into one, zip_longest fills with None if one of the generators is exhausted
    for slice in (x for one_of_all_g_tuple in zip_longest(*gs) for x in one_of_all_g_tuple if x is not None):
        slice.meta["data_key"] = input_tag
        slice.meta["msubject"] = s
        if "group_id" not in slice.meta:
            slice.meta["group_id"] = s.id
        if with_labels is not None:
            # Put the representation labels in the meta key in the form of another representation
            slice = Repr.extract_subject_labels(slice, labeling, extract_labels=with_labels)
        yield slice


class MSubject(SDKSubject):
    # private Pydantic attribute for storing the downloaded data items in memory
    # _data_mem: dict[str, Any] = {}

    # Make MSubject parse into the m3 types, not the SDK types
    data: dict[str, list[ImageType] | ImageType] = {}  # type: ignore Override until refactor to SDK is done
    annotations: list[Annotation] | None = None  # type: ignore Override until refactor to SDK is done

    def make_static(self, downloader: Callable[[str], torch.Tensor] | None = None):
        """
        Tries to replace references such as S3 or local paths with embedded data e.g. by encoding images as base64.

        This can be useful for visualization or labeling without infrastructure.
        """
        from m3_sdk.utils import rgbnumpy_to_base64

        if downloader is None:
            from mmm.api.utils import download_image

            downloader = download_image

        def image_to_base64(img_path: str):
            pil_image = downloader(img_path)
            np_rgb_image = np.array(pil_image)
            return rgbnumpy_to_base64(np_rgb_image)

        for input_key in self.data.keys():
            if isinstance(self.data[input_key], str):
                self.data[input_key] = image_to_base64(self.data[input_key])
            elif isinstance(self.data[input_key], list):
                self.data[input_key] = [image_to_base64(u) if isinstance(u, str) else u for u in self.data[input_key]]
        return self

    @staticmethod
    def extract_instances(
        subj: MSubject,
        labeling=None,
        with_labels: list[str] | None = None,
        max_instances_per_input=None,
    ) -> Generator[Repr, None, None]:
        """
        Extracts the instances from the task.
        """
        for data_key in subj.data.keys():
            for instance in itertools.islice(
                _extract_instances_for_input(subj, labeling, data_key, with_labels),
                max_instances_per_input,
            ):
                yield instance

    # @staticmethod
    # def get_results_for_label(subj: MSubject, o_key: str, use_only_last_annotation=True):
    #     if not subj.annotations:
    #         return []

    #     if use_only_last_annotation:
    #         annos = [subj.get_last_updated_annotation()]
    #     else:
    #         annos = subj.annotations
    #     rs = [result for annotation in annos for result in annotation.result if result["from_name"] == o_key]
    #     if not rs:
    #         return None

    #     return rs, annos

    # def get_all_inputs(self):
    #     res = []
    #     for output_cfg in self.labeling_config.get_parsed().values():
    #         for input_element in output_cfg["inputs"]:
    #             identical = [
    #                 x for x in res if x["value"] == input_element["value"] and x["type"] == input_element["type"]
    #             ]
    #             if not identical:
    #                 res.append(input_element)
    #     return res
