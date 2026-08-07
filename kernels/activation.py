import torch
import torch.nn.functional as F
from tritonforge.core.router import is_cuda_available

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def pytorch_swiglu(x: torch.Tensor) -> torch.Tensor:
    """Standard PyTorch SwiGLU: swish(gate) * value."""
    gate, value = x.chunk(2, dim=-1)
    return F.silu(gate) * value


if HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(
        X, Y,
        stride_x_row, stride_y_row,
        N,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        X += row_idx * stride_x_row
        Y += row_idx * stride_y_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        gate = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        val = tl.load(X + N + cols, mask=mask, other=0.0).to(tl.float32)

        # SiLU(gate) * val
        silu_gate = gate * tl.sigmoid(gate)
        y = silu_gate * val

        tl.store(Y + cols, y, mask=mask)


def fused_swiglu(x: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU forward pass. Uses Triton GPU kernel if available, else PyTorch fallback.
    Input shape: (..., 2 * N) -> Output shape: (..., N)
    """
    if not is_cuda_available() or not HAS_TRITON or not x.is_cuda:
        return pytorch_swiglu(x)

    N = x.shape[-1] // 2
    M = x.numel() // x.shape[-1]

    out_shape = list(x.shape[:-1]) + [N]
    y = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(N)

    _swiglu_fwd_kernel[(M,)](
        x, y,
        x.stride(-2) if x.ndim > 1 else x.shape[-1],
        y.stride(-2) if y.ndim > 1 else N,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y
