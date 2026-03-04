from mmm.mtl_modules.shared_blocks.Embedder import SharedEmbedder
from mmm.mtl_modules.shared_blocks.FeatureMixer import FeatureMixer
from mmm.mtl_modules.shared_blocks.Grouper import Grouper
from mmm.mtl_modules.shared_blocks.MTLDecoder import MTLDecoder
from mmm.mtl_modules.shared_blocks.PyramidDecoder import PyramidDecoder
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.mtl_modules.shared_blocks.Squeezer import Squeezer
from mmm.neural.modules.simple_cnn import MiniConvNet, MiniDecoder
from mmm.neural.modules.smp_modules import SMPUnetDecoder
from mmm.neural.modules.swinformer import TorchVisionSwinformer
from mmm.neural.modules.TorchVisionCNN import TorchVisionCNN

# Optional dependency group for detection
try:
    from mmm.mtl_modules.shared_blocks.FCOSDecoder import FCOSDecoder
except ImportError:
    from typing import TYPE_CHECKING

    if not TYPE_CHECKING:
        FCOSDecoder = None
