import torch
import torch.nn as nn
from tritonforge.core.router import is_cuda_available, triton_route

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

MAX_BLOCK_SIZE = 4096


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
        row_x = X + row_idx * stride_x_row
        row_y = Y + row_idx * stride_y_row

        # Pass 1: Accumulate sum of squares across tiles along N
        var_acc = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(row_x + cols, mask=mask, other=0.0).to(tl.float32)
            var_acc += tl.sum(x * x, axis=0)

        var = var_acc / N
        rrms = tl.math.rsqrt(var + eps)

        if Rrms is not None:
            tl.store(Rrms + row_idx, rrms)

        # Pass 2: Scale and write back to HBM across tiles along N
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(row_x + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
            y = x * rrms * w
            tl.store(row_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton_route(fallback_fn=pytorch_rmsnorm)
def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Fused RMSNorm forward pass. Uses Triton kernel on GPU if available, else PyTorch fallback.
    Guarantees contiguous row strides for 2D, 3D, or non-contiguous sliced tensors.
    Tiles hidden dimension N when N > 4096 to prevent hardware block size overflow.
    """
    if not is_cuda_available() or not HAS_TRITON or not x.is_cuda:
        return pytorch_rmsnorm(x, weight, eps)

    orig_shape = x.shape
    N = orig_shape[-1]

    # Guarantee contiguous memory layout and flatten batch/seq dims into 2D rows
    x_contig = x.contiguous().view(-1, N)
    weight_contig = weight.contiguous()
    M = x_contig.shape[0]

    y = torch.empty_like(x_contig)
    BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)

    _rmsnorm_fwd_kernel[(M,)](
        x_contig, y, weight_contig, None,
        x_contig.stride(0),
        y.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y.view(orig_shape)
