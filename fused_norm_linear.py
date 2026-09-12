import torch
import torch.nn as nn
from tritonforge.core.router import is_cuda_available
from tritonforge.kernels.norm import fused_rmsnorm

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_rmsnorm_gemm_kernel(
        X, W_norm, W_linear, Out,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_om, stride_on,
        M, N, K,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused RMSNorm + Linear GEMM kernel.
        Fuses Root Mean Square Normalization of the input row vector in registers
        and directly computes Matrix Multiplication against Weight tile, eliminating
        intermediate HBM write & read passes.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # 1. Compute RMS variance across K dimension in registers
        x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_norm_ptrs = W_norm + offs_k

        # Accumulate variance
        var_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        for k_idx in range(0, K, BLOCK_K):
            curr_k = k_idx + offs_k
            x_k = tl.load(X + offs_m[:, None] * stride_xm + curr_k[None, :] * stride_xk,
                          mask=(offs_m[:, None] < M) & (curr_k[None, :] < K), other=0.0).to(tl.float32)
            var_acc += tl.sum(x_k * x_k, axis=1)

        rrms = tl.math.rsqrt((var_acc / K) + eps)

        # 2. Compute Fused Matmul using normalized values
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k_idx in range(0, K, BLOCK_K):
            curr_k = k_idx + offs_k
            # Load X slice and normalize with rrms and W_norm
            x_tile = tl.load(X + offs_m[:, None] * stride_xm + curr_k[None, :] * stride_xk,
                             mask=(offs_m[:, None] < M) & (curr_k[None, :] < K), other=0.0).to(tl.float32)
            w_norm = tl.load(W_norm + curr_k, mask=curr_k < K, other=1.0).to(tl.float32)
            x_normed = x_tile * rrms[:, None] * w_norm[None, :]

            # Load Linear Weight tile
            w_linear = tl.load(W_linear + curr_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                               mask=(curr_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)

            acc += tl.dot(x_normed.to(tl.float16), w_linear.to(tl.float16))

        # Write output to HBM
        out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, acc.to(Out.dtype), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class FusedRMSNormLinear(nn.Module):
    """
    Fused RMSNorm + Linear layer module.
    Saves GPU HBM memory bandwidth by avoiding intermediate un-normalized tensor roundtrips.
    """

    def __init__(self, d_model: int, out_features: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.out_features = out_features
        self.eps = eps

        self.norm_weight = nn.Parameter(torch.ones(d_model))
        self.linear = nn.Linear(d_model, out_features, bias=False)

    @property
    def hbm_bytes_saved_per_forward(self) -> int:
        """Calculates intermediate memory write + read bytes saved."""
        bytes_per_elem = 2  # float16 default
        return 2 * self.d_model * bytes_per_elem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if hardware and tensor layout permits custom fused kernel
        if not is_cuda_available() or not HAS_TRITON or not x.is_cuda:
            x_normed = fused_rmsnorm(x, self.norm_weight, self.eps)
            return self.linear(x_normed)

        orig_shape = x.shape
        x_2d = x.view(-1, self.d_model)
        M, K = x_2d.shape
        N = self.out_features
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_K = min(triton.next_power_of_2(K), 128)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        w_linear_t = self.linear.weight.t().contiguous()

        _fused_rmsnorm_gemm_kernel[grid](
            x_2d, self.norm_weight, w_linear_t, out,
            x_2d.stride(0), x_2d.stride(1),
            w_linear_t.stride(0), w_linear_t.stride(1),
            out.stride(0), out.stride(1),
            M, N, K,
            self.eps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        return out.view(*orig_shape[:-1], N)
