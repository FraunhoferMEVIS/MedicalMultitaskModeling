import torch
from mmm.neural.activations import ActivationFn, ActivationFunctionConfig


def test_fn_from_config():
    test_tensor = torch.rand(10)
    multidim_tensor = torch.rand(5, 10)
    for v in ActivationFn:
        f = ActivationFunctionConfig(fn_type=v).build_instance()
        f(test_tensor)

    # Test some values for correctness
    assert torch.equal(
        multidim_tensor,
        ActivationFunctionConfig(fn_type=ActivationFn.Identity).build_instance()(multidim_tensor),
    )
