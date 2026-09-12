import pytest
import torch
from tritonforge.models.transformer_block import TritonForgeTransformerBlock


def test_transformer_block_forward():
    torch.manual_seed(42)
    block = TritonForgeTransformerBlock(d_model=64, n_heads=4, ffn_dim=128)
    x = torch.randn(2, 16, 64, dtype=torch.float32)

    out = block(x)
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()
