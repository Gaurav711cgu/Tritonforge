import torch
from tritonforge.core.router import triton_route, HAS_TRITON

# Try importing Triton packages conditionally
if HAS_TRITON:
    import triton
    import triton.language as tl
else:
    triton = None
    tl = None

# =====================================================================
# PYTORCH REFERENCE & FALLBACK PATHS
# =====================================================================

def pytorch_swiglu(x: torch.Tensor) -> torch.Tensor:
    """Standard PyTorch implementation of SwiGLU activation (fallback path)."""
    a, b = torch.chunk(x, chunks=2, dim=-1)
    return (a * torch.sigmoid(a)) * b

# =====================================================================
# TRITON GPU KERNEL WITH AUTOTUNING (Compiled only if Triton library is present)
# =====================================================================

if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        ],
        key=['N_cols']
    )
    @triton.jit
    def _swiglu_fwd_kernel(
        X_ptr,          # Pointer to input tensor X (shape: M x 2*N)
        Y_ptr,          # Pointer to output tensor Y (shape: M x N)
        stride_x_row,   # Stride between rows in X
        stride_y_row,   # Stride between rows in Y
        N_cols,         # Number of columns in output Y (half of X)
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        col_block_idx = tl.program_id(1)
        
        col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N_cols
        
        offset_a = row_idx * stride_x_row + col_offsets
        offset_b = row_idx * stride_x_row + N_cols + col_offsets
        
        a = tl.load(X_ptr + offset_a, mask=mask, other=0.0)
        b = tl.load(X_ptr + offset_b, mask=mask, other=0.0)
        
        silu_a = a / (1.0 + tl.exp(-a))
        y = silu_a * b
        
        offset_y = row_idx * stride_y_row + col_offsets
        tl.store(Y_ptr + offset_y, y, mask=mask)
else:
    _swiglu_fwd_kernel = None

# =====================================================================
# SYSTEM GATEWAY & SHAPE VALIDATOR
# =====================================================================

def swiglu_shape_validator(x: torch.Tensor) -> bool:
    """Verifies input shape is even and fits shared memory allocations."""
    N = x.shape[-1]
    return N % 2 == 0 and N <= 16384

@triton_route(fallback_fn=pytorch_swiglu, shape_validator=swiglu_shape_validator)
def fused_swiglu(x: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU Gated Activation.
    Accepts tensor of shape (..., 2*N) and returns (..., N).
    Operates on GPU using Triton; falls back to PyTorch on CPU.
    """
    assert HAS_TRITON, "Cannot invoke fused_swiglu without Triton compiler installed."
    assert x.is_contiguous(), "Input tensor must be contiguous."
    
    orig_shape = x.shape
    x_flat = x.view(-1, orig_shape[-1])
    
    M, double_N = x_flat.shape
    N = double_N // 2
    
    y_flat = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    grid = lambda meta: (M, triton.cdiv(N, meta['BLOCK_SIZE']))
    
    _swiglu_fwd_kernel[grid](
        x_flat, y_flat,
        x_flat.stride(0), y_flat.stride(0),
        N
    )
    
    new_shape = orig_shape[:-1] + (N,)
    return y_flat.view(*new_shape)
