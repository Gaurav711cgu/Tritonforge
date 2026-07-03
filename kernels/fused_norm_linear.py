"""
Fused RMSNorm + Linear Projection kernel.

Upgraded to production-grade:
  1. Dynamic routing: GEMV kernel for M=1 (generation), GEMM kernel for M>1 (prefills/training).
  2. Memory-efficient Fused Backward: Bypasses PyTorch autograd graph tracking by manually
     computing dW/dX gradients using cuBLAS matmul and Triton _rmsnorm_bwd_kernel directly.
  3. Dynamic Autotuning: Sweeps block sizing and stages to avoid register pressure limits.
"""
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# =====================================================================
# TRITON FUSED NORM + LINEAR KERNELS
# =====================================================================

if HAS_TRITON:
    # --- GEMM Kernel (For M > 1) ---
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        ],
        key=['M', 'N', 'K']
    )
    @triton.jit
    def _rmsnorm_linear_gemm_kernel(
        X_ptr, W_ptr, norm_w_ptr, Y_ptr,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        M, N, K,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # 1. Compute RMSNorm scale factors (rsqrt) for the rows
        sum_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            off_k = k_start + tl.arange(0, BLOCK_K)
            x_ptr = X_ptr + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
            mask_x = (off_m[:, None] < M) & (off_k[None, :] < K)
            x_val = tl.load(x_ptr, mask=mask_x, other=0.0).to(tl.float32)
            sum_sq += tl.sum(x_val * x_val, axis=1)
            
        rsqrt = 1.0 / tl.sqrt(sum_sq / K + eps)  # size [BLOCK_M]
        
        # 2. Compute GEMM: Y_block = X_normed_block @ W_block.T
        accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        
        for k_start in range(0, K, BLOCK_K):
            off_k = k_start + tl.arange(0, BLOCK_K)
            
            # Load X block
            x_ptr = X_ptr + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
            mask_x = (off_m[:, None] < M) & (off_k[None, :] < K)
            x_val = tl.load(x_ptr, mask=mask_x, other=0.0)
            
            # Load norm weight
            norm_w_val = tl.load(norm_w_ptr + off_k, mask=off_k < K, other=0.0)
            
            # Normalize X block in registers
            x_norm = x_val * rsqrt[:, None] * norm_w_val[None, :]
            
            # Load W block (W is [N, K], so block is [BLOCK_N, BLOCK_K])
            w_ptr = W_ptr + off_n[:, None] * stride_wn + off_k[None, :] * stride_wk
            mask_w = (off_n[:, None] < N) & (off_k[None, :] < K)
            w_val = tl.load(w_ptr, mask=mask_w, other=0.0)
            
            # Compute block GEMM
            accumulator += tl.dot(x_norm, tl.trans(w_val))
            
        y_ptr = Y_ptr + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn
        mask_y = (off_m[:, None] < M) & (off_n[None, :] < N)
        tl.store(y_ptr, accumulator.to(tl.float16), mask=mask_y)

    # --- GEMV Kernel (Optimized for M = 1 vector generation) ---
    @triton.jit
    def _rmsnorm_linear_gemv_kernel(
        X_ptr, W_ptr, norm_w_ptr, Y_ptr,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        M, N, K,
        eps,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        if pid_n >= N:
            return
            
        # 1. Compute RMSNorm of the single row (row 0)
        sum_sq = 0.0
        for k_start in range(0, K, BLOCK_K):
            off_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = off_k < K
            x_ptr = X_ptr + off_k * stride_xk
            x_val = tl.load(x_ptr, mask=mask_k, other=0.0).to(tl.float32)
            sum_sq += tl.sum(x_val * x_val, axis=0)
            
        rsqrt = 1.0 / tl.sqrt(sum_sq / K + eps)
        
        # 2. Compute dot product of normalized row with row pid_n of W
        accumulator = 0.0
        for k_start in range(0, K, BLOCK_K):
            off_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = off_k < K
            
            # Load X
            x_ptr = X_ptr + off_k * stride_xk
            x_val = tl.load(x_ptr, mask=mask_k, other=0.0)
            
            # Load norm weight
            norm_w_val = tl.load(norm_w_ptr + off_k, mask=mask_k, other=0.0)
            
            # Normalize X
            x_norm = x_val * rsqrt * norm_w_val
            
            # Load W row
            w_ptr = W_ptr + pid_n * stride_wn + off_k * stride_wk
            w_val = tl.load(w_ptr, mask=mask_k, other=0.0)
            
            accumulator += tl.sum(x_norm * w_val, axis=0)
            
        y_ptr = Y_ptr + pid_n * stride_yn
        tl.store(y_ptr, accumulator.to(tl.float16))
else:
    _rmsnorm_linear_gemm_kernel = None
    _rmsnorm_linear_gemv_kernel = None


# =====================================================================
# PYTORCH AUTOGRAD FUNCTION
# =====================================================================

if HAS_TRITON:
    class FusedRMSNormLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, norm_weight, W, eps=1e-6):
            assert x.is_cuda and norm_weight.is_cuda and W.is_cuda
            assert x.is_contiguous() and W.is_contiguous()
            
            M, K = x.shape
            N, _ = W.shape
            
            Y = torch.empty((M, N), device=x.device, dtype=torch.float16)
            
            if M == 1:
                # Launch specialized GEMV path
                BLOCK_K = 256
                grid = (N,)
                _rmsnorm_linear_gemv_kernel[grid](
                    x, W, norm_weight, Y,
                    x.stride(0), x.stride(1),
                    W.stride(0), W.stride(1),
                    Y.stride(0), Y.stride(1),
                    M, N, K,
                    eps,
                    BLOCK_K=BLOCK_K
                )
            else:
                # Launch autotuned GEMM path
                # Autotuning handles grid configurations automatically
                grid = lambda meta: (
                    triton.cdiv(M, meta['BLOCK_M']),
                    triton.cdiv(N, meta['BLOCK_N'])
                )
                _rmsnorm_linear_gemm_kernel[grid](
                    x, W, norm_weight, Y,
                    x.stride(0), x.stride(1),
                    W.stride(0), W.stride(1),
                    Y.stride(0), Y.stride(1),
                    M, N, K,
                    eps
                )
            
            ctx.save_for_backward(x, norm_weight, W)
            ctx.eps = eps
            return Y

        @staticmethod
        def backward(ctx, dy):
            x, norm_weight, W = ctx.saved_tensors
            eps = ctx.eps
            
            # Recompute RMSNorm scale factors (rsqrt) in memory-efficient way
            variance = x.pow(2).mean(-1, keepdim=True)
            rsqrt = torch.rsqrt(variance + eps)
            x_normed = x * rsqrt * norm_weight
            
            # Compute gradients:
            # 1. dW = dy.T @ x_normed (Uses standard contiguous PyTorch matrix mult)
            dW = torch.matmul(dy.t(), x_normed)
            
            # 2. dx_normed = dy @ W
            dx_normed = torch.matmul(dy, W)
            
            # 3. Propagate dx_normed through RMSNorm backward using custom Triton bwd kernel
            from tritonforge.kernels.norm import _rmsnorm_bwd_kernel
            
            M, K = x.shape
            dx = torch.empty_like(x)
            dw_row = torch.empty((M, K), device=x.device, dtype=x.dtype)
            
            BLOCK_SIZE = triton.next_power_of_2(K)
            grid = (M,)
            
            _rmsnorm_bwd_kernel[grid](
                dx_normed, x, norm_weight, rsqrt,
                dx, dw_row,
                dx_normed.stride(0), x.stride(0), dx.stride(0), K,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            d_norm_weight = dw_row.sum(dim=0)
            
            return dx, d_norm_weight, dW, None
else:
    FusedRMSNormLinearFunction = None


# =====================================================================
# PUBLIC INTERFACE & MODULE
# =====================================================================

def fused_rmsnorm_linear(x: torch.Tensor, norm_weight: torch.Tensor, W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Fused RMSNorm + Linear (QKV Projection) forward pass.
    Automatically handles CPU/GPU fallback.
    """
    if not HAS_TRITON or not x.is_cuda:
        # Fallback path
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps) * norm_weight
        return x_normed @ W.t()
        
    orig_shape = x.shape
    if len(orig_shape) > 2:
        x_2d = x.view(-1, orig_shape[-1]).contiguous()
    else:
        x_2d = x.contiguous()
        
    res_2d = FusedRMSNormLinearFunction.apply(x_2d, norm_weight, W, eps)
    
    if len(orig_shape) > 2:
        return res_2d.view(*orig_shape[:-1], W.shape[0])
    return res_2d


class FusedRMSNormLinear(nn.Module):
    def __init__(self, d_model: int, out_features: int, eps: float = 1e-6):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(d_model))
        self.linear = nn.Linear(d_model, out_features, bias=False)
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm_linear(x, self.norm_weight, self.linear.weight, self.eps)
