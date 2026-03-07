"""
Wraps an nn.ModuleDict that was created using `trainer.save_blocks_native(...)`.
"""

from __future__ import annotations

import logging
import tempfile
import zipfile
from typing import Any, Literal, Type

import logfire
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from m3_sdk.DistributedPath import DistributedPath
from m3_sdk.types import CompressType
from pydantic import Field

from mmm.api.models import Repr
from mmm.BaseModel import BaseModel
from mmm.mmm_types.GroupUsage import GroupUsage, MaskingStrategy
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.mtl_modules.tasks.SemSegTask import SemSegTask
from mmm.utils import get_default_cachepath

from .FoundationModel import FoundationModel

M3_MODELS = {
    "encoder-1.0.4.pt.zip": "https://data.rslide.org/encoder-1.0.4.pt.zip",
    "tc-nopanda.pt.zip": "https://data.rslide.org/tc-nopanda.pt.zip",
    "unicorn-encoder.pt.zip": "https://drive.usercontent.google.com/download?id=1MShT_e-Srb6tHyRpcA3DvNNHyRaihOx_&export=download&confirm=t",
    "wsc-mtl-tiny.pt.zip": "https://huggingface.co/FraunhoferMEVIS/WholeSlideConcepts/resolve/main/WSC-MTL-tiny.pt.zip?download=true",
}
DEFAULT_MODEL = "encoder-1.0.4.pt.zip"
TC_NOPANDA = "tc-nopanda.pt.zip"
UNICORN_ENCODER = "unicorn-encoder.pt.zip"
WSC_MTL_TINY = "wsc-mtl-tiny.pt.zip"


class M3Model(FoundationModel):
    """
    Wraps an nn.ModuleDict that was created using `trainer.save_blocks_native(...)`.

    Our recommended inference and training routines are implemented here.
    For special use cases, you might need to overwrite methods such as `compress_instances`.
    """

    class Config(BaseModel):
        encoder_key: str = "encoder"
        squeezer_key: str = "squeezer"
        decoder_key: str = "decoder"
        fpn_key: str = "fcosfpn"
        grouper_key: GroupUsage = GroupUsage(grouper_key="grouper", masking=MaskingStrategy.fullattention)
        mixer_key: str = "mixer"

        dense_features: list[Literal["decoder", "fpn", "encoder"]] = Field(
            "decoder",
            description="Decoder for segmentation, fpn for detection (clf & regression), encoder uses first feature map",
        )

    @classmethod
    def load_from_disk(cls: Type[M3Model], file_path: DistributedPath, to_device: str) -> M3Model:
        return cls(file_path, to_device)

    def __init__(
        self, modules_path: DistributedPath | str, device_identifier: str = "cuda", cfg: Config | None = None
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = M3Model.Config()
        self.cfg = cfg

        if isinstance(modules_path, str):
            modules_path = DistributedPath(uri=modules_path)
        if modules_path.uri.startswith("http"):
            # Download file to cache
            cache_folder = get_default_cachepath("models")
            # If the model name is not in the MMM_MODELS, use that name
            model_name = {v: k for k, v in M3_MODELS.items()}.get(
                modules_path.uri, modules_path.get_filestem_suggestion()
            )
            if not model_name.endswith(".pt.zip"):
                model_name = f"{model_name}.pt.zip"
            if not (target_path := cache_folder / model_name).exists():
                with logfire.span("Model does not exist at {target_path}, downloading...", target_path=target_path):
                    torch.hub.download_url_to_file(modules_path.uri, target_path)
            else:
                logfire.warning("Model already exists at {target_path}, not redownloading", target_path=target_path)
            modules_path = DistributedPath(uri=str(target_path), options=modules_path.options)

        self.device = device_identifier

        assert modules_path.uri.endswith(".pt.zip"), "Model file must be a .pt.zip file"
        if getattr(self, "unique_identifier", None) is None:
            self.unique_identifier = modules_path.upath().stem.split(".")[0]

        # Extract the zip file to a temporary directory
        with modules_path.upath().open("rb") as f:
            with zipfile.ZipFile(f, "r") as zip_ref:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    zip_ref.extractall(tmpdirname)
                    modules_path = DistributedPath(uri=tmpdirname, options=modules_path.options)
                    # Load the modules from the temporary directory
                    self.torch_modules: nn.ModuleDict = nn.ModuleDict()
                    for file in modules_path.upath().glob("*.pt"):
                        try:
                            with file.open("rb") as module_f:
                                self.torch_modules[file.stem] = torch.load(module_f, weights_only=False).to(
                                    device_identifier
                                )
                        except Exception as e:
                            logfire.warning("Did not load {file} due to {e}", file=file, e=e)

    def get_identifier(self) -> str:
        if getattr(self, "unique_identifier", None) is None:
            return "M3Model"
        return self.unique_identifier

    def get_sharedblock_keys(self) -> list[str]:
        return [k for k, v in self.torch_modules.items() if isinstance(v, SharedBlock)]

    def get_task_keys(self) -> list[str]:
        return [k for k, v in self.torch_modules.items() if isinstance(v, MTLTask)]

    def get_device(self) -> str:
        return self.device

    def __getitem__(self, key):
        return self.torch_modules[key]

    def keys(self):
        return self.torch_modules.keys()

    @staticmethod
    def save_to_disk(file_path: DistributedPath, neural_modules: nn.ModuleDict):
        """
        Stores the modules in a Zip with one file per module.

        The file names of neural modules are the keys of the nn.ModuleDict.
        """
        assert file_path.uri.endswith(".zip")
        import tempfile
        import zipfile

        with zipfile.ZipFile(file_path.upath(), "w") as zip_ref:
            for k, v in neural_modules.items():
                with tempfile.NamedTemporaryFile() as tmpfile:
                    torch.save(v, tmpfile.name)
                    zip_ref.write(tmpfile.name, f"{k}.pt")
        return file_path

    @torch.inference_mode()
    def compute_dense_features(self, feature_pyramid):
        """
        All feature maps are [batch_size, channels, height, width].
        """
        pixel_tokens = []

        if "decoder" in self.cfg.dense_features:
            pixel_tokens.append(self.torch_modules[self.cfg.decoder_key].forward(feature_pyramid))

        if "fpn" in self.cfg.dense_features:
            cls_features, reg_features = self.torch_modules[self.cfg.fpn_key].forward(feature_pyramid)
            pixel_tokens.extend(cls_features[:1])
            # pixel_tokens.extend(reg_features[:1])

        for i, feature_map in enumerate(pixel_tokens[1:]):
            if feature_map.shape[-2:] != pixel_tokens[0].shape[-2:]:
                print(f"Resizing feature map {i + 1} from {feature_map.shape[-2:]} to {pixel_tokens[0].shape[-2:]}")
                pixel_tokens[i + 1] = F.resize(
                    feature_map, size=pixel_tokens[0].shape[-2:], interpolation=F.InterpolationMode.NEAREST_EXACT
                )

        return torch.cat(pixel_tokens, dim=1)

    @torch.inference_mode()
    def compress_instances(
        self,
        instances: dict[str, Any],
        for_representation: list[CompressType],
    ) -> dict[CompressType, Repr]:
        """
        Instances are a dictionary of input data for the model.

        An image key would correspond to a batch of images.

        The meta key is special as it is unbatched (still a list).
        """
        model_input = instances["image"]
        res = {}

        if CompressType.rgbimage in for_representation:
            res[CompressType.rgbimage] = model_input
            if len(for_representation) == 1:
                return res  # No need to compute a neural representation

        # Also replaces at least the last feature map with the squeezer computation
        feature_pyramid, squeezed = SemSegTask.compute_squeezed_features(
            self.torch_modules[self.cfg.encoder_key],
            self.torch_modules[self.cfg.squeezer_key],
            model_input.to(self.get_device()),
        )

        if CompressType.token in for_representation:
            res[CompressType.token] = squeezed

        if CompressType.ctxtoken in for_representation:
            raise NotImplementedError("Needs to respect positional information")
            from mmm.mtl_modules.shared_blocks.Grouper import Grouper

            grouper: Grouper = self.torch_modules[self.cfg.grouper_key]
            supercase_indices: torch.Tensor = grouper.extract_ids_from_batch(
                [x["group_id"] for x in instances["meta"]]
            ).to(grouper.torch_device)
            grouptokens, grouper_weights = grouper(squeezed, supercase_indices)
            res[CompressType.ctxtoken] = grouptokens

        if CompressType.pixeltoken in for_representation:
            res[CompressType.pixeltoken] = self.compute_dense_features(feature_pyramid)
        return res

    # @torch.inference_mode()
    # def _group_for_classification(self, compressed_label: CompressedLabel, batch_size=768, keep_n=50):
    #     from mmm.mtl_modules.shared_blocks.Grouper import Grouper
    #     from torch.utils.data import DataLoader

    #     grouper: Grouper = self.torch_modules[self.cfg.grouper_key]
    #     instance_contexts = list(compressed_label.instances.keys())
    #     if "single" in instance_contexts:
    #         logging.warning(f"Context 'single' is already present in {instance_contexts=}, and will be overwritten.")
    #     all_instances = [compressed_label.instances[context] for context in instance_contexts]

    #     # Keep the keep_n instances with the highest relevance
    #     if len(all_instances) > keep_n:
    #         # Rate the relevance of all instances and filter by it
    #         instance_weights = []
    #         for batch in DataLoader(
    #             [i.tensor.squeeze() for i in all_instances], batch_size=batch_size, shuffle=False, num_workers=0
    #         ):
    #             weights = grouper.rate_instance_relevance(
    #                 batch.to(grouper.torch_device), [0 for _ in range(len(batch))]
    #             )
    #             instance_weights.append(weights.cpu())
    #         instance_weights: torch.Tensor = torch.cat(instance_weights).squeeze(0)

    #         keep_indices = instance_weights.argsort(descending=True)[:keep_n].tolist()
    #         instance_contexts = [instance_contexts[i] for i in keep_indices]
    #         compressed_label.instances = {context: compressed_label.instances[context] for context in instance_contexts}
    #     # print(len(compressed_label.instances))
    #     # Group the remaining instances
    #     supercase_indices: torch.Tensor = grouper.extract_ids_from_batch(
    #         [0 for _ in range(len(compressed_label.instances))]
    #     ).to(grouper.torch_device)
    #     case_batch = torch.cat([compressed_label.instances[context].tensor for context in instance_contexts])
    #     squeezed, grouper_weights = grouper(case_batch.to(grouper.torch_device), supercase_indices)

    #     # Replace the instances with the grouped instance
    #     groupinstance = Repr(
    #         tensor=squeezed.to(compressed_label.instances[instance_contexts[0]].ground_truth.device),
    #         ground_truth=compressed_label.instances[instance_contexts[0]].ground_truth,
    #         meta=compressed_label.instances[instance_contexts[0]].meta,
    #     )
    #     compressed_label.instances = {"single": groupinstance}
    #     return compressed_label

    # def group(self, for_label: str, compressed_label: CompressedLabel) -> CompressedLabel:
    #     output_cfg = compressed_label.subject.labeling_config.get_parsed()[for_label]
    #     for_inputs: list[str] = compressed_label.subject.labeling_config.get_parsed()[for_label]["to_name"]
    #     assert len(for_inputs) == 1, "Grouping for multiple inputs is not implemented"

    #     input_element = compressed_label.subject.labeling_config.get_input_element_by_key(for_inputs[0])
    #     images_as_list = input_element["type"] == "Image" and isinstance(
    #         compressed_label.subject.data[for_inputs[0]], list
    #     )
    #     require_grouping = len(compressed_label.instances) > 1
    #     require_grouping = require_grouping or images_as_list
    #     if require_grouping:
    #         labeltype: LabelType = compressed_label.subject.labeling_config.determine_type_of_label(output_cfg)

    #         if labeltype in ["Choices", "Survival"]:
    #             return self._group_for_classification(compressed_label)
    #         elif labeltype in ["BrushLabels"]:
    #             # BrushLabels are not grouped
    #             pass
    #         else:
    #             raise ValueError(f"Grouping for label type {labeltype} is not implemented")
    #     else:
    #         return compressed_label
