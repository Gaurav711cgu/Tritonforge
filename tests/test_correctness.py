import pytest
import torch
import math
from tritonforge.core.router import is_cuda_available
from tritonforge.kernels.norm import fused_rmsnorm, pytorch_rmsnorm
from tritonforge.kernels.activation import fused_swiglu, pytorch_swiglu
from tritonforge.kernels.attention import fused_attention, pytorch_flash_attention

def get_test_device() -> str:
    return "cuda" if is_cuda_available() else "cpu"

def get_tolerance(dtype=torch.float32) -> float:
    return 1e-4 if dtype == torch.float16 else 1e-5

@pytest.mark.parametrize("shape", [(8, 128), (4, 1024), (2, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rmsnorm_correctness(shape, dtype):
    """
    Verifies that fused_rmsnorm matches pytorch_rmsnorm numerically,
    supporting both standard GPU and fallback CPU runtimes.
    """
    device = get_test_device()
    
    # Skip fp16 on CPU as PyTorch has limited native support for CPU fp16 math
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("Skipping FP16 testing on CPU.")

    # 1. Initialize input tensors
    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn((shape[-1],), dtype=dtype, device=device, requires_grad=True)
    eps = 1e-6
    
    # 2. Run Forward pass
    out_fused = fused_rmsnorm(x, weight, eps=eps)
    out_ref = pytorch_rmsnorm(x, weight, eps=eps)
    
    # 3. Assert Forward correctness
    tol = get_tolerance(dtype)
    torch.testing.assert_close(out_fused, out_ref, atol=tol, rtol=tol)
    
    # 4. Backward Pass Correctness
    # Generate identical incoming gradients
    dy = torch.randn_like(out_fused)
    
    # Run backpropagation on Triton/Fallback path
    out_fused.backward(dy, retain_graph=True)
    grad_x_fused = x.grad.clone()
    grad_w_fused = weight.grad.clone()
    
    # Reset gradients
    x.grad.zero_()
    weight.grad.zero_()
    
    # Run backpropagation on PyTorch Reference path
    out_ref.backward(dy)
    grad_x_ref = x.grad.clone()
    grad_w_ref = weight.grad.clone()
    
    # Assert Backward correctness
    torch.testing.assert_close(grad_x_fused, grad_x_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(grad_w_fused, grad_w_ref, atol=tol, rtol=tol)

@pytest.mark.parametrize("shape", [(16, 256), (8, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_swiglu_correctness(shape, dtype):
    """Verifies fused_swiglu matches reference split implementations."""
    device = get_test_device()
    
    # SwiGLU inputs are (M, 2*N)
    x = torch.randn(shape, dtype=dtype, device=device)
    
    out_fused = fused_swiglu(x)
    out_ref = pytorch_swiglu(x)
    
    tol = get_tolerance(dtype)
    torch.testing.assert_close(out_fused, out_ref, atol=tol, rtol=tol)

@pytest.mark.parametrize("batch_heads_seq_dim", [
    (1, 2, 64, 64),
    (2, 4, 128, 128),
    # Test fallback shape trigger (d = 96 is unsupported by block-tiled Triton FlashAttention)
    (1, 2, 64, 96)
])
def test_attention_correctness(batch_heads_seq_dim):
    """Verifies tiled FlashAttention matches standard reference calculations."""
    device = get_test_device()
    B, H, N, d = batch_heads_seq_dim
    
    # Initialize Query, Key, and Value matrices
    q = torch.randn((B, H, N, d), dtype=torch.float32, device=device)
    k = torch.randn((B, H, N, d), dtype=torch.float32, device=device)
    v = torch.randn((B, H, N, d), dtype=torch.float32, device=device)
    
    sm_scale = 1.0 / math.sqrt(d)
    
    out_fused = fused_attention(q, k, v, sm_scale)
    out_ref = pytorch_flash_attention(q, k, v, sm_scale)
    
    # Note: Attention calculations can accumulate slight numeric differences 
    # due to local online softmax tiling updates. We verify with appropriate bounds.
    torch.testing.assert_close(out_fused, out_ref, atol=1e-4, rtol=1e-4)
