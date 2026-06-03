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
# PYTORCH REFERENCE & FALLBACK PATHS (Eager execution on CPU or GPU fallback)
# =====================================================================

def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard PyTorch implementation of RMSNorm (fallback path)."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# =====================================================================
# TRITON GPU KERNELS (Compiled only if Triton library is present)
# =====================================================================

if HAS_TRITON:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr,          # Pointer to input tensor X
        Y_ptr,          # Pointer to output tensor Y
        W_ptr,          # Pointer to weight tensor Gamma
        R_ptr,          # Pointer to save-rsqrt tensor (for backward pass)
        stride_x_row,   # Stride between rows in X
        stride_y_row,   # Stride between rows in Y
        N_cols,         # Number of columns per row
        eps,            # Epsilon constant
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N_cols
        
        x = tl.load(X_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0)
        
        x_sq = x * x
        sum_squares = tl.sum(x_sq, axis=0)
        mean_squares = sum_squares / N_cols
        
        rsqrt = 1.0 / tl.sqrt(mean_squares + eps)
        
        if R_ptr is not None:
            tl.store(R_ptr + row_idx, rsqrt)
            
        w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        y = x * rsqrt * w
        tl.store(Y_ptr + row_idx * stride_y_row + col_offsets, y, mask=mask)

    @triton.jit
    def _rmsnorm_bwd_kernel(
        DY_ptr,         # Pointer to incoming gradient DY
        X_ptr,          # Pointer to forward input tensor X
        W_ptr,          # Pointer to weight tensor Gamma
        R_ptr,          # Pointer to saved rsqrt tensor
        DX_ptr,         # Pointer to output gradient DX
        DW_ptr,         # Pointer to row-wise weight gradient accumulation
        stride_dy_row,
        stride_x_row,
        stride_dx_row,
        N_cols,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N_cols
        
        dy = tl.load(DY_ptr + row_idx * stride_dy_row + col_offsets, mask=mask, other=0.0)
        x = tl.load(X_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0)
        w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        rsqrt = tl.load(R_ptr + row_idx)
        
        u = x * rsqrt
        dy_w = dy * w
        dy_w_x = dy_w * x
        sum_dy_w_x = tl.sum(dy_w_x, axis=0)
        
        dx = (dy_w * rsqrt) - (x * (rsqrt * rsqrt * rsqrt / N_cols) * sum_dy_w_x)
        dw_row = dy * u
        
        tl.store(DX_ptr + row_idx * stride_dx_row + col_offsets, dx, mask=mask)
        tl.store(DW_ptr + row_idx * N_cols + col_offsets, dw_row, mask=mask)

else:
    _rmsnorm_fwd_kernel = None
    _rmsnorm_bwd_kernel = None

# =====================================================================
# PYTORCH AUTOGRAD WRAPPER
# =====================================================================

class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        assert HAS_TRITON, "Cannot invoke FusedRMSNormFunction without Triton compiler installed."
        assert x.is_contiguous(), "Input tensor X must be contiguous."
        assert weight.is_contiguous(), "Weight tensor must be contiguous."
        
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        M, N = x_flat.shape
        
        y_flat = torch.empty_like(x_flat)
        rsqrt = torch.empty((M,), dtype=torch.float32, device=x.device)
        
        BLOCK_SIZE = triton.next_power_of_2(N)
        grid = (M,)
        
        _rmsnorm_fwd_kernel[grid](
            x_flat, y_flat, weight, rsqrt,
            x_flat.stride(0), y_flat.stride(0),
            N, eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        ctx.save_for_backward(x_flat, weight, rsqrt)
        ctx.orig_shape = orig_shape
        
        return y_flat.view(*orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple:
        x_flat, weight, rsqrt = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        
        dy_flat = dy.reshape(-1, orig_shape[-1])
        M, N = dy_flat.shape
        
        dx_flat = torch.empty_like(x_flat)
        dw_rows = torch.empty((M, N), dtype=weight.dtype, device=weight.device)
        
        BLOCK_SIZE = triton.next_power_of_2(N)
        grid = (M,)
        
        _rmsnorm_bwd_kernel[grid](
            dy_flat, x_flat, weight, rsqrt,
            dx_flat, dw_rows,
            dy_flat.stride(0), x_flat.stride(0), dx_flat.stride(0),
            N, BLOCK_SIZE=BLOCK_SIZE
        )
        
        dw = dw_rows.sum(dim=0)
        return dx_flat.view(*orig_shape), dw, None

# =====================================================================
# SYSTEM GATEWAY & SHAPE VALIDATOR
# =====================================================================

def norm_shape_validator(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> bool:
    """Verifies that columns dimension satisfies GPU shared memory limits."""
    N = x.shape[-1]
    return N <= 8192

@triton_route(fallback_fn=pytorch_rmsnorm, shape_validator=norm_shape_validator)
def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Main interface. Executes Fused RMSNorm on GPU using Triton,
    otherwise automatically degrades to standard PyTorch implementation on CPU.
    """
    return FusedRMSNormFunction.apply(x, weight, eps)
