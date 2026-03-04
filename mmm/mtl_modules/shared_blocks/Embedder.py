from typing import Annotated, Mapping, Tuple, Union

import torch
import torch.nn as nn
from pydantic import Field

from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.neural.modules.tabular_embedding import CategoricalEmbedder


class SharedEmbedder(SharedBlock):
    """
    Takes a feature pyramid and squeezes it into a single feature map.
    """

    class Config(SharedBlock.Config):
        module_name: str = "shared_embedder"
        embedding: CategoricalEmbedder.Config

    def __init__(self, args: Config, embedding_dim: int):
        super().__init__(args)
        self.args: SharedEmbedder.Config

        self.embedder = CategoricalEmbedder(self.args.embedding, embedding_dim)

        self.make_mtl_compatible()

    def forward(self, tokens: torch.Tensor, categories: list[str]) -> torch.Tensor:
        return self.embedder(tokens, categories)
