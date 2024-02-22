import torch
import math


def posemb(t: torch.Tensor, device: str, dim: int = 16) -> torch.Tensor:
    """
    A tensor such as `Tensor([1, 5])` is transformed to an embedding of shape (2, dim).
    As a result, it can be stacked or added to a semantic embedding.

    `dim` should be divisable by 2.

    You can inspect the result using a seaborn heatmap:

    ```python
    import seaborn as sns
    sns.heatmap(posemb(torch.Tensor(list(range(20000))), device="cpu", dim=16))
    ```
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb
