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


def test_transformer_block_causal_invariance():
    """
    Verifies causal masking invariant:
    Mutating future tokens at step t > 0 MUST NOT affect the model outputs at step 0.
    """
    torch.manual_seed(42)
    block = TritonForgeTransformerBlock(d_model=64, n_heads=4, ffn_dim=128, causal=True)
    block.eval()

    # Input tensor 1
    x1 = torch.randn(1, 8, 64, dtype=torch.float32)
    # Input tensor 2: Identical at token 0, completely different at tokens 1..7
    x2 = x1.clone()
    x2[:, 1:, :] = torch.randn(1, 7, 64, dtype=torch.float32)

    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)

    # Token 0 representations must match exactly under causal attention
    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-4, rtol=1e-4), (
        "Causal masking invariant failed: output at position 0 changed when future tokens were mutated!"
    )
