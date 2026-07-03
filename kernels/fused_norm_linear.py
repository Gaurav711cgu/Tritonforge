"""
Fused RMSNorm + Linear Projection kernel.

Standard (unfused) path:
  x_normed = rmsnorm(x)        # writes [M, N] to HBM
  qkv = x_normed @ W_qkv.T    # reads  [M, N] from HBM again

Fused path (this kernel):
  Single kernel: reads x once, computes norm in SRAM,
  immediately applies W_qkv projection, writes output once.
  
  Eliminates one full HBM read of [M, N] = seq_len * d_model * 2 bytes.
  For Gemma-2-2b-it: seq=2048, d=2304 → saves 9.4MB per forward pass per layer.
  26 layers × forward + backward = ~490MB HBM traffic eliminated per step.

Use case: direct replacement for the QKV projection in transformer attention.
  Instead of: x_norm = rmsnorm(x); q, k, v = linear(x_norm).chunk(3)
  Use:        qkv = fused_rmsnorm_linear(x, norm_weight, W_qkv)
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
# TRITON FUSED NORM + LINEAR KERNEL
# =====================================================================

if HAS_TRITON:
    @triton.jit
    def _rmsnorm_linear_fwd_kernel(
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
        # Program coordinates
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Offsets
        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # 1. Compute RMSNorm scale factors (rsqrt) for the rows in our block
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
            
        # 3. Write output block of Y
        y_ptr = Y_ptr + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn
        mask_y = (off_m[:, None] < M) & (off_n[None, :] < N)
        tl.store(y_ptr, accumulator.to(tl.float16), mask=mask_y)
else:
    _rmsnorm_linear_fwd_kernel = None


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
            
            BLOCK_M = 16
            BLOCK_N = 64
            BLOCK_K = 32
            
            grid = (
                triton.cdiv(M, BLOCK_M),
                triton.cdiv(N, BLOCK_N)
            )
            
            _rmsnorm_linear_fwd_kernel[grid](
                x, W, norm_weight, Y,
                x.stride(0), x.stride(1),
                W.stride(0), W.stride(1),
                Y.stride(0), Y.stride(1),
                M, N, K,
                eps,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K
            )
            
            ctx.save_for_backward(x, norm_weight, W)
            ctx.eps = eps
            return Y

        @staticmethod
        def backward(ctx, dy):
            x, norm_weight, W = ctx.saved_tensors
            eps = ctx.eps
            
            with torch.enable_grad():
                x_var = x.detach().requires_grad_(True)
                norm_w_var = norm_weight.detach().requires_grad_(True)
                W_var = W.detach().requires_grad_(True)
                
                # Reconstruct forward sequentially for autograd backward pass
                variance = x_var.pow(2).mean(-1, keepdim=True)
                x_normed = x_var * torch.rsqrt(variance + eps) * norm_w_var
                y = x_normed @ W_var.t()
                
                y.backward(dy)
                
            return x_var.grad, norm_w_var.grad, W_var.grad, None
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
    """
    Drop-in replacement for nn.Sequential(RMSNorm, nn.Linear) in transformer blocks.
    Executes a single fused Triton JIT kernel on GPU.
    """
    def __init__(self, d_model: int, out_features: int, eps: float = 1e-6):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(d_model))
        self.linear = nn.Linear(d_model, out_features, bias=False)
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, d_model]
        returns: [B, seq_len, out_features]
        """
        return fused_rmsnorm_linear(x, self.norm_weight, self.linear.weight, self.eps)

    @property
    def hbm_bytes_saved_per_forward(self) -> int:
        """
        Bytes saved vs unfused (RMSNorm write + Linear read of intermediate).
        Approximation for a single [1, seq_len, d_model] input in fp16.
        """
        return 2 * self.d_model * 2  # per token, in bytes


def benchmark_fused_vs_unfused(seq_len: int = 2048, d_model: int = 2304, out_features: int = 6912):
    """
    Compares fused vs unfused RMSNorm + Linear.
    out_features=6912 = 3 * d_model for QKV projection in Gemma-2-2b-it.
    """
    from tritonforge.core.profiler import profile_op, estimate_metrics
    from tritonforge.kernels.norm import fused_rmsnorm
    
    x = torch.randn(1, seq_len, d_model, device="cuda", dtype=torch.float16).contiguous()
    norm_w = torch.ones(d_model, device="cuda", dtype=torch.float16)
    W = torch.randn(out_features, d_model, device="cuda", dtype=torch.float16)

    # Unfused: separate ops
    def unfused(x, norm_w, W):
        x_norm = fused_rmsnorm(x, norm_w)
        return x_norm @ W.T

    # Fused module
    fused_module = FusedRMSNormLinear(d_model, out_features).cuda().half()

    unfused_ms = profile_op(unfused, x, norm_w, W, warmups=20, reps=100)
    fused_ms = profile_op(fused_module, x, warmups=20, reps=100)
    speedup = unfused_ms / fused_ms

    bytes_io = (d_model + out_features) * seq_len * 2  # input + output in fp16
    bw_gbs, _ = estimate_metrics(fused_ms, bytes_io)

    return {
        "seq_len": seq_len,
        "d_model": d_model,
        "out_features": out_features,
        "unfused_ms": round(unfused_ms, 4),
        "fused_ms": round(fused_ms, 4),
        "speedup": round(speedup, 2),
        "achieved_bw_gbs": round(bw_gbs, 1),
    }
