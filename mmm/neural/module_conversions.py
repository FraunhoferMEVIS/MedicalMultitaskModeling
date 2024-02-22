import torch.nn as nn
from typing import List, Any, get_type_hints, Dict, Callable, Type, Optional, TypeVar
from inspect import signature


def base_norm_replacement(oldnorm: nn.Module, replacements: Dict):
    for old_type, new_type in replacements.items():
        if isinstance(oldnorm, old_type):
            return new_type(oldnorm.num_features)
    raise Exception(f"I do not know how to convert {type(oldnorm)} to instance norm")


def build_instancenorm_like(
    oldnorm: nn.Module,
) -> nn.modules.instancenorm._InstanceNorm:
    return base_norm_replacement(oldnorm, {nn.BatchNorm2d: nn.InstanceNorm2d, nn.BatchNorm3d: nn.InstanceNorm3d})


def replace_childen_recursive(m: nn.Module, layertype_to_replace, newlayer_constructor, suppress_inheritance=False):
    """
    newlayer_constructor gets an instance of the layer to be replaced.

    Use cases might be replacing all 2D layers by their respective 3D versions.
    """
    for k, layer in m.named_children():
        by_typecheck = suppress_inheritance and type(layer) is layertype_to_replace
        by_isinstancecheck = (not suppress_inheritance) and isinstance(layer, layertype_to_replace)
        # Replace the object in question
        if by_typecheck or by_isinstancecheck:
            if newlayer_constructor is None:
                raise Exception(f"Constructor of replacement of {layertype_to_replace} is None :(")
            newlayer = newlayer_constructor(layer)
            # Does not work for children's children, which is why this is a recursive function
            setattr(m, k, newlayer)

        replace_childen_recursive(
            layer,
            layertype_to_replace,
            newlayer_constructor,
            suppress_inheritance=suppress_inheritance,
        )


def derive_args_from_layer(old_layer: nn.Module, new_layer_type, transformator=None):
    args: Dict[str, Any] = {}
    for p_name, param in signature(new_layer_type).parameters.items():
        # if param.default is param.empty:
        # First see if the old layer has the same attribute and take it
        # Then try to use the default
        p_old = getattr(old_layer, p_name, param.default)
        if True in [isinstance(p_old, t) for t in [str, int, tuple, bool]]:
            if transformator is not None:
                p_old = transformator(p_name, p_old)
            args[p_name] = p_old
        # except AttributeError as e:
        #     args.append(param.default)
    return args


def replace_tuples(layer_2d: nn.Module, target_layer_type) -> nn.Module:
    def two_tuple_to_three_tuple(p_name: str, p_old: Any) -> Any:
        if isinstance(p_old, tuple) and len(p_old) == 2:
            # print(f"REPLACING {p_name} with value {p_old}")
            return p_old[0], p_old[1], (p_old[0] + p_old[1]) // 2
        return p_old

    res = target_layer_type(**derive_args_from_layer(layer_2d, target_layer_type, two_tuple_to_three_tuple))

    # res.kernel_size = (
    #     conv2d_layer.kernel_size[0],
    #     conv2d_layer.kernel_size[1],
    #     conv2d_layer.kernel_size[0] + conv2d_layer.kernel_size[1] // 2
    # )
    # res.stride = (
    #     conv2d_layer.stride[0],
    #     conv2d_layer.stride[1],
    #     conv2d_layer.stride[0] + conv2d_layer.stride[1] // 2
    # )
    return res
    # return nn.Conv3d(
    #     conv2d_layer.in_channels,
    #     conv2d_layer.out_channels,
    #     # conv2d has a 2-tuple as kernel size which will result in an error
    #     kernel_size=conv2d_layer.kernel_size[0]
    # )


M = TypeVar("M", bound=nn.Module)


def convert_2d_to_3d(m: M) -> M:
    d: Dict[Type[nn.Module], Optional[Callable[[nn.Module], nn.Module]]] = {
        nn.Conv2d: lambda l: replace_tuples(l, nn.Conv3d),
        nn.AdaptiveAvgPool2d: lambda l: replace_tuples(l, nn.AdaptiveAvgPool3d),
        nn.Dropout2d: None,
        nn.InstanceNorm2d: None,
        nn.BatchNorm2d: lambda l: replace_tuples(l, nn.BatchNorm3d),
        nn.ConvTranspose2d: None,
        nn.MaxPool2d: lambda l: replace_tuples(l, nn.MaxPool3d),
        nn.AdaptiveMaxPool2d: None,
        nn.AvgPool2d: None,
        nn.ReplicationPad2d: None,
        nn.ConstantPad2d: None,
        # Batch norm does not need to be adapted when going to 3D?
        # StateLessBatchnorm: None
    }

    for old_layer, newlayer_constructor in d.items():
        if type(m) is old_layer:
            if newlayer_constructor is None:
                raise Exception(f"Layertype of {old_layer} is missing a way to convert to 3d!")
            return newlayer_constructor(m)
        replace_childen_recursive(m, old_layer, newlayer_constructor, suppress_inheritance=True)
    return m
