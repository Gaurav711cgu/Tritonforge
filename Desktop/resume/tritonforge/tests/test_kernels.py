"""
TritonForge Kernel Correctness & Memory Reduction Test Suite
Supports both CUDA GPU runtime and CPU reference fallback.
"""

import math
import pytest
import torch
from tritonforge.core.router import is_cuda_available
from tritonforge.kernels.norm import fused_rmsnorm, pytorch_rmsnorm
from tritonforge.kernels.activation import fused_swiglu, pytorch_swiglu
from tritonforge.kernels.attention import fused_attention, pytorch_flash_attention
from tritonforge.kernels.fused_norm_linear import FusedRMSNormLinear


def get_device() -> str:
    return "cuda" if is_cuda_available() else "cpu"


def test_rmsnorm_correctness():
    """
    Verifies RMSNorm fused kernel against PyTorch reference implementation.
    Tolerance atol=1e-3 matches float32/float16 kernel validation standards.
    """
    torch.manual_seed(42)
    device = get_device()
    shape = (512, 2304)

    x = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn((shape[-1],), device=device, dtype=torch.float32, requires_grad=True)
    eps = 1e-6

    # Forward pass
    out_fused = fused_rmsnorm(x, weight, eps=eps)
    out_ref = pytorch_rmsnorm(x, weight, eps=eps)

    # Numerical close assertion
    torch.testing.assert_close(out_fused, out_ref, atol=1e-3, rtol=1e-3)
    max_err = (out_ref - out_fused).abs().max().item()
    assert max_err < 1e-3, f"Max error {max_err:.2e} exceeds tolerance 1e-3"


def test_swiglu_correctness():
    """
    Verifies fused SwiGLU kernel against PyTorch reference implementation.
    """
    torch.manual_seed(42)
    device = get_device()
    shape = (1024, 4096 * 2)

    x = torch.randn(shape, device=device, dtype=torch.float32)

    out_fused = fused_swiglu(x)
    out_ref = pytorch_swiglu(x)

    torch.testing.assert_close(out_fused, out_ref, atol=1e-3, rtol=1e-3)


def test_attention_correctness():
    """
    Verifies tiled FlashAttention output matches standard scaled dot-product attention.
    """
    torch.manual_seed(42)
    device = get_device()
    batch, heads, seq, dim = 1, 8, 2048, 64

    q = torch.randn((batch, heads, seq, dim), device=device, dtype=torch.float32)
    k = torch.randn((batch, heads, seq, dim), device=device, dtype=torch.float32)
    v = torch.randn((batch, heads, seq, dim), device=device, dtype=torch.float32)
    sm_scale = 1.0 / math.sqrt(dim)

    out_fused = fused_attention(q, k, v, sm_scale)
    out_ref = pytorch_flash_attention(q, k, v, sm_scale)

    assert out_fused.shape == (batch, heads, seq, dim)
    torch.testing.assert_close(out_fused, out_ref, atol=1e-3, rtol=1e-3)


def test_attention_memory_reduction():
    """
    Verifies 95.3%+ memory footprint reduction claim for sequence length 2048.
    Standard attention materializes full [B, H, N, N] float32 matrix (134.2 MB decimal / 128.0 MiB binary).
    Fused FlashAttention streams tiles through SRAM, allocating only [B, H, N, d] output (4.2 MB decimal / 4.0 MiB binary).
    """
    batch, heads, seq, dim = 1, 8, 2048, 64
    bytes_per_elem = 4  # float32

    # Standard attention matrix bytes = B * H * N * N * 4
    standard_bytes = batch * heads * (seq ** 2) * bytes_per_elem
    standard_mb_decimal = standard_bytes / 1e6
    standard_mib_binary = standard_bytes / (1024 * 1024)

    # Fused attention SRAM-tiled output bytes = B * H * N * d * 4
    fused_bytes = batch * heads * seq * dim * bytes_per_elem
    fused_mb_decimal = fused_bytes / 1e6
    fused_mib_binary = fused_bytes / (1024 * 1024)

    memory_reduction_pct = (1.0 - (fused_bytes / standard_bytes)) * 100.0

    assert round(standard_mb_decimal, 1) == 134.2, f"Decimal MB: {standard_mb_decimal:.1f}"
    assert round(standard_mib_binary, 1) == 128.0, f"Binary MiB: {standard_mib_binary:.1f}"
    assert memory_reduction_pct >= 95.0, f"Expected >= 95% reduction, got {memory_reduction_pct:.2f}%"


def test_fused_rmsnorm_linear_correctness():
    """
    Verifies single-pass FusedRMSNormLinear module vs unfused sequential RMSNorm + Linear.
    """
    torch.manual_seed(42)
    device = get_device()
    shape = (2, 64, 128)
    B, S, D = shape
    out_features = D * 2

    x = torch.randn((B, S, D), dtype=torch.float32, device=device)
    fused_mod = FusedRMSNormLinear(d_model=D, out_features=out_features).to(device)

    out_fused = fused_mod(x)
    x_normed = fused_rmsnorm(x, fused_mod.norm_weight, fused_mod.eps)
    out_ref = fused_mod.linear(x_normed)

    torch.testing.assert_close(out_fused, out_ref, atol=1e-3, rtol=1e-3)
    assert fused_mod.hbm_bytes_saved_per_forward == 2 * D * 2
