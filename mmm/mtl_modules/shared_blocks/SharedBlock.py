from abc import abstractmethod
from pathlib import Path
from typing import List, Literal, Mapping, Sequence, Tuple
import io

try:
    import onnx
except ImportError:
    pass
import torch
import torch.nn as nn

from mmm.mtl_modules.MTLModule import MTLModule
from mmm.torch_ext import replace_childen_recursive
from mmm.neural.module_conversions import (
    convert_2d_to_3d,
    build_instancenorm_like,
)

from mmm.mtl_modules.MTLModule import MTLModule
from .TaskSpecificLayer import TaskSpecificLayer

ModelInput = torch.Tensor | List[torch.Tensor]


class SharedBlock(MTLModule):
    """
    A shared block is intended to contain mainly parameters shared among multiple tasks.
    A shared block can have task-specific layers, new tasks can be introduced by `prepare_for_task(...)`

    A classic example is an encoder which encodes an image into a disentangled latent space.
    """

    class Config(MTLModule.Config):
        norm_layer: Literal[
            "nonorm",
            "instancenorm",
            "keepnorm",
            "layernorm",
            "taskspecific-batchnorm",
            "affinelayernorm",
        ] = "affinelayernorm"

    def __init__(self, args: Config) -> None:
        super().__init__(args)
        self.args = args
        self.next_active_task: str = ""
        self.task_specific_modules: List[TaskSpecificLayer] = []
        self._made_mtl_compatible = False

    def make_mtl_compatible(self):
        # Set normalization layer
        if self.args.norm_layer == "keepnorm":
            # shared blocks should use batchnorm as default
            pass
        elif self.args.norm_layer == "layernorm" or self.args.norm_layer == "affinelayernorm":
            if self.args.norm_layer == "layernorm":
                # Transformer models come with Layernorms by default, those only need to lose their learnable params
                replace_childen_recursive(
                    self,
                    nn.LayerNorm,
                    lambda oldnorm: nn.LayerNorm(
                        normalized_shape=oldnorm.normalized_shape,
                        eps=oldnorm.eps,
                        elementwise_affine=False,
                    ),
                )
            # CNNs come with Batchnorms by default, those can be replaced by Groupnorm with one group
            replace_childen_recursive(
                self,
                nn.modules.batchnorm._NormBase,
                lambda oldnorm: nn.GroupNorm(
                    1,
                    oldnorm.num_features,
                    affine=self.args.norm_layer == "affinelayernorm",
                ),
            )
        elif self.args.norm_layer == "instancenorm":
            replace_childen_recursive(self, nn.modules.batchnorm._NormBase, build_instancenorm_like)
        elif self.args.norm_layer == "nonorm":
            replace_childen_recursive(self, nn.modules.batchnorm._NormBase, lambda _: nn.Identity())
        elif self.args.norm_layer == "taskspecific-batchnorm":
            # Batchnorm supertype taken from the PyTorch implementation of SyncBatchNorm.convert_sync_batchnorm
            self._make_layertype_task_specific(nn.modules.batchnorm._BatchNorm)
        else:
            raise Exception(f"Unknown normlayer {self.args.norm_layer}")

        self._made_mtl_compatible = True

    def convert_to_3d(self):
        raise DeprecationWarning("Converting to 3D encoder is currently untested")
        for k, layer in self.named_children():
            setattr(self, k, convert_2d_to_3d(layer))

    def _make_layertype_task_specific(self, layertype_to_replace):
        """
        Replaces all layers of a certain type by a wrapper module which copies that layer for each task.
        """

        def task_specific_constructor(oldlayer) -> TaskSpecificLayer:
            newlayer = TaskSpecificLayer(oldlayer)
            self.task_specific_modules.append(newlayer)
            return newlayer

        replace_childen_recursive(self, layertype_to_replace, task_specific_constructor)

    def prepare_for_task(self, task_id: str) -> None:
        """
        Needs to be called once before training for each task if task-specific layers are used.
        """
        for l in self.task_specific_modules:
            l.create_layer_for_task(task_id)

    def set_active_task(self, task_id: str):
        """
        Needs to be called once before each forward pass if task-specific layers are used.
        """
        for l in self.task_specific_modules:
            l.set_active_task(task_id)

    def get_example_input(self) -> ModelInput | Tuple[ModelInput, ...]:
        """
        Used for example in tracing for ONNX export.
        """
        return torch.rand(1, 3, 224, 224).to(self.torch_device)

    def get_input_names(self) -> Sequence[str]:
        return ["input"]

    def get_output_names(self) -> Sequence[str]:
        return ["output"]

    def get_dynamic_axes(self) -> Mapping[str, Mapping[int, str]]:
        return {}

    def export_to_onnx(self, path: Path, for_task: str = "") -> None:
        if onnx is None:
            raise ImportError("ONNX not installed, cannot export model")

        self.set_active_task(for_task)

        example_input = self.get_example_input()
        example_output_shapes = [
            output_tensor.shape
            for example_output in (self(*example_input) if isinstance(example_input, tuple) else self(example_input))
            for output_tensor in (example_output if isinstance(example_output, list) else (example_output,))
        ]
        output_names = self.get_output_names()
        dynamic_axes = self.get_dynamic_axes()

        with io.BytesIO() as buffer:
            torch.onnx.export(
                self.eval(),  # model being run
                example_input,  # model input (or a tuple for multiple inputs)
                buffer,  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=13,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=self.get_input_names(),  # the model's input names
                output_names=output_names,  # the model's output names
                training=torch.onnx.TrainingMode.EVAL,
                dynamic_axes=self.get_dynamic_axes(),
            )
            buffer.seek(0)
            onnx_model = onnx.load(buffer)

        # Set the correct model output shapes since shape inference fails for some of our models
        for i, (name, shape) in enumerate(zip(output_names, example_output_shapes)):
            for d, s in enumerate(shape):
                if (name not in dynamic_axes or d not in dynamic_axes[name]) and onnx_model.graph.output[
                    i
                ].type.tensor_type.shape.dim[d].dim_param:
                    onnx_model.graph.output[i].type.tensor_type.shape.dim[d].dim_param = ""
                    onnx_model.graph.output[i].type.tensor_type.shape.dim[d].dim_value = s

        onnx.save_model(onnx_model, str(path))

    @abstractmethod
    def forward(self, input):  # type: ignore (weird override thing from PyTorch)
        raise NotImplementedError
