from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field

from mmm.BaseModel import BaseModel


class CategoricalEmbedder(nn.Module):
    """
    Convert a categorical feature into an embedding vector.

    >>> import torch
    >>> from mmm.neural.modules.tabular_embedding import CategoricalEmbedder
    >>> tokens = torch.zeros(batchdim := 4, embedding_dim := 2)
    >>> embedder = CategoricalEmbedder(
    ...     CategoricalEmbedder.Config(categories=["red", "blue", "yellow"]),
    ...     embedding_dim,
    ... )
    >>> # Do nothing if category is empty string
    >>> out = embedder(tokens, ["", "yellow", "blue", "blue"])
    >>> assert torch.allclose(tokens[0], out[0])
    >>> out.shape, out[2] == out[3], out[1] != out[2]
    (torch.Size([4, 2]), tensor([True, True]), tensor([True, True]))
    """

    class Config(BaseModel):
        tabular_type: Literal["categorical"] = "categorical"
        categories: list[str] = Field(
            description="List of options to choose from. Order matters. Empty string will be ignored in forward."
        )

    def __init__(self, cfg: Config, embedding_dim: int):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(len(self.cfg.categories), embedding_dim)

        self.category_to_index = {category: i for i, category in enumerate(self.cfg.categories)}
        self.category_to_index[""] = 0  # will be ignored in forward

    def forward(self, tokens: torch.Tensor, categories: list[str]) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): A tensor of shape (batchdim, embedding_dim) representing the input tokens.
            categories (list[str]): A list of strings where each string represents a category for each batch item.

        Returns:
            torch.Tensor: A tensor of shape (batchdim, embedding_dim) representing the output after applying the embedding and dropout.
        """
        # If the string is not in the categories, use a zero vector
        category_ids = torch.tensor([self.category_to_index[category] for category in categories], device=tokens.device)
        factor = torch.tensor(
            [1.0 if category != "" else 0.0 for category in categories], device=tokens.device
        ).unsqueeze(1)
        return tokens + self.embedding(category_ids) * factor


class NumericEmbedder(nn.Module):
    """
    Convert a tabular feature into an embedding vector.

    Enables to define notable intervals for the feature, and adds a representation of the value itself.

    >>> import torch; from mmm.neural.modules.tabular_embedding import NumericEmbedder
    >>> test_numerics = torch.tensor([[-10000.0], [2.0], [3.0]])
    >>> NumericEmbedder(NumericEmbedder.Config(), embedding_dim:=3)(test_numerics).shape
    torch.Size([3, 3, 3])
    """

    class Config(BaseModel):
        tabular_type: Literal["numeric"] = "numeric"
        intervals: tuple[float, float] = Field(
            default=(-torch.inf, torch.inf), description="Tuple of min and max values for the feature."
        )
        dropout: float = 0.2

    def __init__(self, cfg: Config, embedding_dim: int):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = embedding_dim
        self.buckets = torch.tensor(self.cfg.intervals)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.categorical = nn.Embedding(len(self.cfg.intervals) - 1, embedding_dim)
        self.numeric = nn.Linear(1, embedding_dim)

    def forward(self, x):
        assert x.shape[1] == 1
        interval_indices = torch.bucketize(x, self.buckets) - 1
        embedded = self.categorical(interval_indices)

        # Use tanh to avoid exploding values and make the layer require multiple channels
        scaled = F.tanh(self.numeric(x))
        return self.dropout(embedded + scaled)
