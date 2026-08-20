import pytest
import torch
from tritonforge.models.transformer_block import TransformerBlock


def test_transformer_block_forward():
    torch.manual_seed(42)
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=128)
    x = torch.randn(2, 16, 64, dtype=torch.float32)

    out = block(x)
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()


def test_transformer_block_causal():
    torch.manual_seed(42)
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=128)
    x = torch.randn(2, 16, 64, dtype=torch.float32)

    out = block(x, causal=True)
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()
