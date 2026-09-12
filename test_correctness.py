import pytest
import torch
import math
from tritonforge.core.router import is_cuda_available
from tritonforge.kernels.norm import fused_rmsnorm, pytorch_rmsnorm
from tritonforge.kernels.norm_autograd import fused_rmsnorm_autograd
from tritonforge.kernels.rmsnorm_cuda_ext import rmsnorm_cuda_cpp
from tritonforge.kernels.activation import fused_swiglu, pytorch_swiglu
from tritonforge.kernels.attention import fused_attention, pytorch_flash_attention
from tritonforge.kernels.fused_norm_linear import FusedRMSNormLinear

def get_test_device() -> str:
    return "cuda" if is_cuda_available() else "cpu"

def get_tolerance(dtype=torch.float32) -> float:
    return 1e-3 if dtype in [torch.float16, torch.bfloat16] else 1e-4

@pytest.mark.parametrize("shape", [(8, 128), (4, 1024), (2, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_correctness(shape, dtype):
    """
    Verifies that fused_rmsnorm matches pytorch_rmsnorm numerically,
    supporting float32, float16, and bfloat16 dtypes.
    """
    device = get_test_device()

    # Skip fp16/bf16 on CPU as PyTorch has limited native support for CPU fp16 math
    if device == "cpu" and dtype in [torch.float16, torch.bfloat16]:
        pytest.skip("Skipping FP16/BF16 testing on CPU.")

    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn((shape[-1],), dtype=dtype, device=device, requires_grad=True)
    eps = 1e-6

    out_fused = fused_rmsnorm(x, weight, eps=eps)
    out_ref = pytorch_rmsnorm(x, weight, eps=eps)

    tol = get_tolerance(dtype)
    torch.testing.assert_close(out_fused, out_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("shape", [(4, 256), (2, 1024)])
def test_rmsnorm_autograd_correctness(shape):
    """
    Verifies FusedRMSNormAutograd forward and backward passes against PyTorch reference.
    """
    device = get_test_device()
    x = torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True)
    weight = torch.randn((shape[-1],), dtype=torch.float32, device=device, requires_grad=True)
    eps = 1e-6

    out_auto = fused_rmsnorm_autograd(x, weight, eps=eps)
    out_ref = pytorch_rmsnorm(x, weight, eps=eps)

    torch.testing.assert_close(out_auto, out_ref, atol=1e-4, rtol=1e-4)

    dy = torch.randn_like(out_auto)
    out_auto.backward(dy, retain_graph=True)
    grad_x_auto = x.grad.clone()
    grad_w_auto = weight.grad.clone()

    x.grad.zero_()
    weight.grad.zero_()

    out_ref.backward(dy)
    grad_x_ref = x.grad.clone()
    grad_w_ref = weight.grad.clone()

    torch.testing.assert_close(grad_x_auto, grad_x_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(grad_w_auto, grad_w_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("shape", [(4, 256), (2, 512)])
def test_rmsnorm_cuda_cpp_correctness(shape):
    """
    Verifies raw CUDA C++ / CPU fallback RMSNorm implementation.
    """
    device = get_test_device()
    x = torch.randn(shape, dtype=torch.float32, device=device)
    weight = torch.randn((shape[-1],), dtype=torch.float32, device=device)
    eps = 1e-6

    out_cpp = rmsnorm_cuda_cpp(x, weight, eps=eps)
    out_ref = pytorch_rmsnorm(x, weight, eps=eps)

    torch.testing.assert_close(out_cpp, out_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256), (8, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_swiglu_correctness(shape, dtype):
    device = get_test_device()
    x = torch.randn(shape, dtype=dtype, device=device)

    out_fused = fused_swiglu(x)
    out_ref = pytorch_swiglu(x)

    tol = get_tolerance(dtype)
    torch.testing.assert_close(out_fused, out_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("batch_heads_seq_dim", [
    (1, 2, 64, 64),
    (2, 4, 128, 128),
    (1, 2, 64, 96)
])
def test_attention_correctness(batch_heads_seq_dim):
    device = get_test_device()
    B, H, N, d = batch_heads_seq_dim

    q = torch.randn((B, H, N, d), dtype=torch.float32, device=device)
    k = torch.randn((B, H, N, d), dtype=torch.float32, device=device)
    v = torch.randn((B, H, N, d), dtype=torch.float32, device=device)

    sm_scale = 1.0 / math.sqrt(d)

    out_fused = fused_attention(q, k, v, sm_scale)
    out_ref = pytorch_flash_attention(q, k, v, sm_scale)

    torch.testing.assert_close(out_fused, out_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", [(2, 64, 128), (1, 128, 256)])
def test_fused_rmsnorm_linear_correctness(shape):
    device = get_test_device()
    B, S, D = shape
    out_features = D * 2

    x = torch.randn((B, S, D), dtype=torch.float32, device=device)

    fused_mod = FusedRMSNormLinear(d_model=D, out_features=out_features).to(device)

    out_fused = fused_mod(x)

    x_normed = fused_rmsnorm(x, fused_mod.norm_weight, fused_mod.eps)
    out_ref = fused_mod.linear(x_normed)

    torch.testing.assert_close(out_fused, out_ref, atol=1e-5, rtol=1e-5)
    assert fused_mod.hbm_bytes_saved_per_forward == 2 * D * 2
