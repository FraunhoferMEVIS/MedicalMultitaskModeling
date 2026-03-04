from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from mmm.api.M3Model import M3_MODELS, M3Model
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder
from tests.test_shared_blocks import default_encoder_factory

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    if TYPE_CHECKING:
        onnx, ort = None, None


@pytest.fixture
def skip_if_export_not_installed() -> bool:
    try:
        import onnx

        return True
    except ImportError:
        pytest.skip("onnx not available")


@pytest.fixture(params=["cpu", "cuda"])
def torch_devices(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda not available")
    return request.param


@pytest.fixture(params=list(M3_MODELS.keys()))
def model_weights(request):
    return request.param


def test_loading_weights(model_weights, torch_devices):
    native_blocks = M3Model(M3_MODELS[model_weights], device_identifier=torch_devices)
    assert len(native_blocks.get_sharedblock_keys()) > 0
    assert len(native_blocks.get_task_keys()) >= 0


def test_onnx_export_encoder(tmp_path: Path, default_encoder_factory, torch_devices, skip_if_export_not_installed):
    pytest.skip("ONNX export got some new options but with the current implementation they get stuck")
    GPU_GPU_DECIMALS, CPU_GPU_DECIMALS = 3, 2

    enc: PyramidEncoder = default_encoder_factory().set_device(torch_devices)
    enc.eval()

    # Obtain the target value using torch
    with torch.no_grad():
        test_input: torch.Tensor = torch.rand(1, 3, 256, 256).to(enc.torch_device)
        y_shouldbe = enc(test_input)[-1].cpu().numpy()
        y_shouldbe2 = enc(test_input)[-1].cpu().numpy()
        enc.set_device("cpu")
        enc.eval()
        y_shouldbe_cpu = enc(test_input.cpu())[-1].cpu().numpy()

    # Ensure that the torch model works
    np.testing.assert_almost_equal(y_shouldbe2, y_shouldbe, decimal=GPU_GPU_DECIMALS)
    np.testing.assert_almost_equal(y_shouldbe_cpu, y_shouldbe, CPU_GPU_DECIMALS)

    onnx_file_path = Path(tmp_path) / "onnx_net.onnx"

    enc.export_to_onnx(onnx_file_path)
    onnx_model = onnx.load(str(onnx_file_path))
    ort_sess = ort.InferenceSession(str(onnx_file_path))
    outputs = ort_sess.run(
        None,
        {
            input_arg.name: input_value
            for input_arg, input_value in zip(ort_sess.get_inputs(), (test_input.numpy(force=True),))
        },
    )
    onnx.checker.check_model(onnx_model)  # type: ignore
    np.testing.assert_almost_equal(outputs[-1], y_shouldbe, CPU_GPU_DECIMALS)
    np.testing.assert_almost_equal(outputs[-1], y_shouldbe2, CPU_GPU_DECIMALS)
    np.testing.assert_almost_equal(outputs[-1], y_shouldbe_cpu, CPU_GPU_DECIMALS)


def test_export_encoder(tmp_path: Path, default_encoder_factory, torch_devices):
    """
    PyTorch team recommends it like this:
    https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html#compare-the-pytorch-results-with-the-ones-from-the-onnx-runtime
    """
    GPU_GPU_DECIMALS, CPU_GPU_DECIMALS = 3, 2

    enc: PyramidEncoder = default_encoder_factory().set_device(torch_devices)
    if enc.args.model.architecture in ["swinformer"]:
        pytest.skip("Swinformer does not support export")
    enc.eval()

    # Obtain the target value using torch
    with torch.no_grad():
        test_input: torch.Tensor = torch.rand(1, 3, 256, 256).to(enc.torch_device)
        y_shouldbe = enc(test_input)[-1].cpu().numpy()
        y_shouldbe2 = enc(test_input)[-1].cpu().numpy()
        enc.set_device("cpu")
        enc.eval()
        y_shouldbe_cpu = enc(test_input.cpu())[-1].cpu().numpy()

    # Ensure that the torch model works
    np.testing.assert_almost_equal(y_shouldbe2, y_shouldbe, decimal=GPU_GPU_DECIMALS)
    np.testing.assert_almost_equal(y_shouldbe_cpu, y_shouldbe, CPU_GPU_DECIMALS)

    ep_file_path = Path(tmp_path) / "onnx_net.onnx"

    enc.export(ep_file_path)
    ep = torch.export.load(ep_file_path)
    with torch.no_grad():
        outputs = ep.module().to(torch_devices)(test_input)
        out = outputs[-1].cpu().numpy()
    np.testing.assert_almost_equal(out, y_shouldbe, CPU_GPU_DECIMALS)
    np.testing.assert_almost_equal(out, y_shouldbe2, CPU_GPU_DECIMALS)
    np.testing.assert_almost_equal(out, y_shouldbe_cpu, CPU_GPU_DECIMALS)
