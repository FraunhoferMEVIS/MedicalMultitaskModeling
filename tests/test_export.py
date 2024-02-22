import pytest
import numpy as np
import torch
from pathlib import Path
from mmm.data_loading.ClassificationDataset import ClassificationDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder

from mmm.data_loading.synthetic.mockup import ClassificationMockupDataset
from mmm.trainer.MTLTrainer import MTLTrainer
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder
from mmm.mtl_modules.shared_blocks.Squeezer import Squeezer
from mmm.neural.modules.simple_cnn import MiniConvNet
from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.mtl_modules.tasks.ClassificationTask import ClassificationTask

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    pass


@pytest.fixture(params=["cpu", "cuda"])
def torch_devices(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda not available")
    return request.param


def test_onnx_export_encoder(tmp_path: Path, default_encoder_factory, torch_devices):
    GPU_GPU_DECIMALS, CPU_GPU_DECIMALS = 3, 2

    enc: PyramidEncoder = default_encoder_factory().set_device(torch_devices)
    enc.eval()

    # Obtain the target value using torch
    with torch.no_grad():
        test_input: torch.Tensor = torch.rand(1, 3, 64, 64).to(enc.torch_device)
        y_shouldbe = enc(test_input)[-1].cpu().numpy()
        y_shouldbe2 = enc(test_input)[-1].cpu().numpy()
        enc.set_device("cpu")
        enc.eval()
        y_shouldbe_cpu = enc(test_input.cpu())[-1].cpu().numpy()

    # Ensure that the torch model works
    np.testing.assert_almost_equal(y_shouldbe2, y_shouldbe, decimal=GPU_GPU_DECIMALS)
    np.testing.assert_almost_equal(y_shouldbe_cpu, y_shouldbe, CPU_GPU_DECIMALS)

    onnx_file_path = Path(tmp_path) / "onnx_net.onnx"

    if enc.args.model.architecture == "swinformer":
        with pytest.raises(NotImplementedError):
            enc.export_to_onnx(onnx_file_path)
        return
    else:
        enc.export_to_onnx(onnx_file_path)
    onnx_model = onnx.load(str(onnx_file_path))
    ort_sess = ort.InferenceSession(str(onnx_file_path))
    outputs = ort_sess.run(None, {"input": test_input.detach().cpu().numpy()})
    onnx.checker.check_model(onnx_model)  # type: ignore
    np.testing.assert_almost_equal(outputs[-1], y_shouldbe, CPU_GPU_DECIMALS)
    np.testing.assert_almost_equal(outputs[-1], y_shouldbe2, CPU_GPU_DECIMALS)
    np.testing.assert_almost_equal(outputs[-1], y_shouldbe_cpu, CPU_GPU_DECIMALS)
