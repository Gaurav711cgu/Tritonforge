import math
import pytest
import torch
import torch.nn.functional as F

from tritonforge.kernels.attention import fused_attention, pytorch_flash_attention
from tritonforge.kernels.norm import fused_rmsnorm, pytorch_rmsnorm
from tritonforge.kernels.activation import fused_swiglu, pytorch_swiglu
from tritonforge.kernels.fused_norm_linear import FusedRMSNormLinear


def test_attention_numerical_correctness():
    torch.manual_seed(42)
    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)
    v = torch.randn(B, H, S, D, dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(D)

    out_py = pytorch_flash_attention(q, k, v, sm_scale, causal=False)
    out_fused = fused_attention(q, k, v, sm_scale, causal=False)

    assert torch.allclose(out_py, out_fused, atol=1e-4, rtol=1e-4)


def test_causal_attention_numerical_correctness():
    torch.manual_seed(42)
    B, H, S, D = 2, 2, 64, 32
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)
    v = torch.randn(B, H, S, D, dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(D)

    out_py = pytorch_flash_attention(q, k, v, sm_scale, causal=True)
    out_fused = fused_attention(q, k, v, sm_scale, causal=True)

    assert torch.allclose(out_py, out_fused, atol=1e-4, rtol=1e-4)


def test_rmsnorm_numerical_correctness():
    torch.manual_seed(42)
    x = torch.randn(8, 256, dtype=torch.float32)
    weight = torch.ones(256, dtype=torch.float32)

    out_py = pytorch_rmsnorm(x, weight, eps=1e-6)
    out_fused = fused_rmsnorm(x, weight, eps=1e-6)

    assert torch.allclose(out_py, out_fused, atol=1e-4, rtol=1e-4)


def test_swiglu_numerical_correctness():
    torch.manual_seed(42)
    x = torch.randn(8, 512, dtype=torch.float32)

    out_py = pytorch_swiglu(x)
    out_fused = fused_swiglu(x)

    assert torch.allclose(out_py, out_fused, atol=1e-4, rtol=1e-4)


def test_fused_rmsnorm_linear_layer():
    torch.manual_seed(42)
    layer = FusedRMSNormLinear(d_model=128, out_features=256)
    x = torch.randn(4, 32, 128, dtype=torch.float32)

    out = layer(x)
    assert out.shape == (4, 32, 256)
    assert not torch.isnan(out).any()
    assert layer.hbm_bytes_saved_per_forward == 2 * 128 * 2
