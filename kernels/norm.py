import torch
import torch.nn as nn
from tritonforge.core.router import is_cuda_available

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard PyTorch eager implementation of Root Mean Square Normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


if HAS_TRITON:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X, Y, W, Rrms,
        stride_x_row, stride_y_row,
        N, eps,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        X += row_idx * stride_x_row
        Y += row_idx * stride_y_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rrms = tl.math.rsqrt(var + eps)

        if Rrms is not None:
            tl.store(Rrms + row_idx, rrms)

        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        y = x * rrms * w
        tl.store(Y + cols, y, mask=mask)


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Fused RMSNorm forward pass. Uses Triton kernel on GPU if available, else PyTorch fallback.
    """
    if not is_cuda_available() or not HAS_TRITON or not x.is_cuda:
        return pytorch_rmsnorm(x, weight, eps)

    M = x.numel() // x.shape[-1]
    N = x.shape[-1]

    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)

    _rmsnorm_fwd_kernel[(M,)](
        x, y, weight, None,
        x.stride(-2) if x.ndim > 1 else N,
        y.stride(-2) if y.ndim > 1 else N,
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y
