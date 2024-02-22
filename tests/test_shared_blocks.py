import pytest
import torch
import torch.nn as nn

from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder
from mmm.neural.modules.simple_cnn import MiniConvNet
from mmm.mtl_modules.shared_blocks.PyramidDecoder import PyramidDecoder
from mmm.neural.modules.swinformer import TorchVisionSwinformer
from mmm.neural.modules.TorchVisionCNN import TorchVisionCNN
from mmm.neural.modules.TimmEncoder import TimmEncoder, valid_variants as timm_variants

# Used to make readable names appear in test report
default_encoders = {
    "MiniConvNet": MiniConvNet.Config(),
    "Swinformer": TorchVisionSwinformer.Config(pretrained=False, variant="tiny"),
    "ResNet18": TorchVisionCNN.Config(pretrained=False, variant="resnet18"),
    "Efficientnet_v2": TorchVisionCNN.Config(pretrained=False, variant="efficientnet_v2_s"),
    "Densenet": TorchVisionCNN.Config(pretrained=False, variant="densenet121"),
    "Convnext": TimmEncoder.Config(pretrained=False, variant="convnext_femto"),
}


@pytest.fixture(params=timm_variants)
def timm_variants_fixture(request):
    return request.param


@pytest.fixture(params=list(default_encoders.keys()))
def default_encoder_factory(request):
    """
    Does not include Timm encoders because their hidden dim is not configurable
    """

    def build_encoder(**kwargs):
        enc_config = default_encoders[request.param]
        return PyramidEncoder(args=PyramidEncoder.Config(model=enc_config, **kwargs))

    return build_encoder


@pytest.fixture
def default_decoder_factory():
    def build_decoder(enc: PyramidEncoder):
        return PyramidDecoder(PyramidDecoder.Config(), enc.get_feature_pyramid_channels(), 32)

    return build_decoder


def test_encoder_checkpointing(tmp_path, default_encoder_factory):
    encoder, encoder2 = default_encoder_factory(), default_encoder_factory()

    # Batchnorms are initialized to exact values, so only check that at least one parameter is different before
    assert False in [
        torch.equal(p1, p2) for (_, p1), (_, p2) in zip(encoder.named_parameters(), encoder2.named_parameters())
    ], "No parameter is different before the checkpointing test, so the test makes no sense"

    p = tmp_path
    encoder.save_checkpoint(p)
    encoder2.load_checkpoint(p)

    for (p1_name, p1), (p2_name, p2) in zip(encoder.named_parameters(), encoder2.named_parameters()):
        assert p1_name == p2_name, "The test assumes that parameters are loaded in the same order"
        assert p1 is not p2, "Both blocks seem to hold the exact same parameter instance"
        assert torch.equal(p1, p2), f"Checkpointing forgot {p1_name} parameter"


def assert_dims_of_encoder(enc, device):
    B, C, H, W = 4, 3, 64, 96
    enc.set_active_task("original")  # task-specific layers not relevant for this test

    testinput = torch.randn((B, C, H, W)).to(device)
    pyr = enc.forward(testinput)
    for i, (stride, dim) in enumerate(zip(enc.get_strides(), enc.get_feature_pyramid_channels())):
        assert pyr[i].size() == (B, dim, H // stride, W // stride)
        assert pyr[0].size() == (B, 3, 64, 96)


def test_default_encoder_dims(default_encoder_factory, torch_device):
    enc: PyramidEncoder = default_encoder_factory().to(torch_device)
    assert_dims_of_encoder(enc, torch_device)


def test_timm_variants(torch_device, timm_variants_fixture):
    args = TimmEncoder.Config(pretrained=False, variant=timm_variants_fixture)
    enc = PyramidEncoder(PyramidEncoder.Config(model=args)).to(torch_device)
    assert_dims_of_encoder(enc, torch_device)


def test_num_task_specific_layers(default_encoder_factory):
    enc: PyramidEncoder = default_encoder_factory(norm_layer="taskspecific-batchnorm")
    assert enc._made_mtl_compatible, f"{enc} missing `self.make_mtl_compatible()` call"

    batchnorms = [m for m in enc.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    # After instantiation, each task-specific module should contain exactly one batchnorm
    assert len(enc.task_specific_modules) == len(batchnorms)
