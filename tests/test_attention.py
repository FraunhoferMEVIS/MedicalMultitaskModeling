import torch
from mmm.neural.modules.attention import DropRandomMHAMask, build_mha_alibi_mask, get_alibi_slope


def test_never_drop_itself():
    B = 2  # Batch size
    S = 5  # Sequence length
    H = 3  # Number of heads
    mask = DropRandomMHAMask(drop_prob=0.1).build_mask(b=B, num_heads=H, seqlen=S, device=torch.device("cpu"))
    assert torch.cat([mask[..., i, i] for i in range(S)]).sum().item() == 0.0


def test_drop_random_mha_mask():
    B, S, H = 2, 5, 3  # Batch size, Sequence length, Number of heads
    mask = DropRandomMHAMask(drop_prob=0.5).build_mask(b=B, num_heads=H, seqlen=S, device=torch.device("cpu"))
    assert float("-inf") in torch.unique(mask).tolist()
    assert 0.0 in torch.unique(mask).tolist()
    assert mask.shape == (B, H, S, S)


def test_alibi_mask():
    mask = build_mha_alibi_mask(get_alibi_slope(H := 3), S := torch.tensor([0, 0, 1, 3, 4, 0]), True)
    assert True not in (mask > 0.0), "ALiBi decays longer distances"
