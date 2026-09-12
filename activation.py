import torch
import torch.nn.functional as F
from tritonforge.core.router import is_cuda_available, triton_route

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

MAX_BLOCK_SIZE = 4096


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
        row_x = X + row_idx * stride_x_row
        row_y = Y + row_idx * stride_y_row

        # Tile across hidden dimension N
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N

            gate = tl.load(row_x + cols, mask=mask, other=0.0).to(tl.float32)
            val = tl.load(row_x + N + cols, mask=mask, other=0.0).to(tl.float32)

            # SiLU(gate) * val
            silu_gate = gate * tl.sigmoid(gate)
            y = silu_gate * val

            tl.store(row_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton_route(fallback_fn=pytorch_swiglu)
def fused_swiglu(x: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU forward pass. Uses Triton GPU kernel if available, else PyTorch fallback.
    Input shape: (..., 2 * N) -> Output shape: (..., N)
    Guarantees contiguous row strides for 2D, 3D, or non-contiguous sliced tensors.
    Tiles hidden dimension N when N > 4096 to prevent hardware block size overflow.
    """
    if not is_cuda_available() or not HAS_TRITON or not x.is_cuda:
        return pytorch_swiglu(x)

    orig_shape = x.shape
    total_hidden = orig_shape[-1]
    N = total_hidden // 2
    out_shape = (*orig_shape[:-1], N)

    # Guarantee contiguous memory layout and flatten batch/seq dims into 2D rows
    x_contig = x.contiguous().view(-1, total_hidden)
    M = x_contig.shape[0]

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)

    _swiglu_fwd_kernel[(M,)](
        x_contig, y,
        x_contig.stride(0),
        y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y.view(out_shape)
