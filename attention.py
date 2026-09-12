import math
import torch
import torch.nn.functional as F
from tritonforge.core.router import is_cuda_available, triton_route

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def pytorch_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, causal: bool = False) -> torch.Tensor:
    """Standard PyTorch reference scaled dot-product attention."""
    scores = torch.matmul(q, k.transpose(-1, -2)) * sm_scale
    if causal:
        seq_len_q = q.shape[-2]
        seq_len_k = k.shape[-2]
        mask = torch.triu(torch.full((seq_len_q, seq_len_k), float("-inf"), device=q.device, dtype=q.dtype), diagonal=1)
        scores = scores + mask
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)


if HAS_TRITON:
    @triton.jit
    def _attn_fwd_kernel(
        Q, K, V, sm_scale,
        L, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        # Offset pointers for current batch and head
        q_offset = off_z * stride_qz + off_h * stride_qh
        k_offset = off_z * stride_kz + off_h * stride_kh
        v_offset = off_z * stride_vz + off_h * stride_vh
        o_offset = off_z * stride_oz + off_h * stride_oh

        # Block offsets for Q (sequence chunk M) and head dim D
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_n = tl.arange(0, BLOCK_N)

        # Initialize pointers to Q
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        # Initialize pointers to K and V
        k_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        # Output pointers
        o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

        # Initialize online softmax statistics in SRAM
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        # Load Q block with boundary mask
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        # Loop over K and V blocks (along sequence dimension N)
        end_n = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
        for start_n in range(0, end_n, BLOCK_N):
            current_offs_n = start_n + offs_n
            # Load K tile
            k = tl.load(k_ptrs, mask=current_offs_n[None, :] < N_CTX, other=0.0)
            
            # Compute QK^T in SRAM
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale

            # Apply causal mask if requested
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= current_offs_n[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            # Online softmax update (Milakov-Gimelshein algorithm)
            m_curr = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_curr)
            alpha = tl.math.exp(m_i - m_new)
            p = tl.math.exp(qk - m_new[:, None])
            
            # Scale previous accumulator by alpha
            acc *= alpha[:, None]
            
            # Load V tile and accumulate
            v = tl.load(v_ptrs, mask=current_offs_n[:, None] < N_CTX, other=0.0)
            acc += tl.dot(p.to(v.dtype), v)
            
            # Update denominator
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

            # Advance K and V pointers along sequence length
            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn

        # Normalize accumulator by l_i and write output to HBM
        acc = acc / l_i[:, None]
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


@triton_route(fallback_fn=pytorch_flash_attention)
def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float = None, causal: bool = False) -> torch.Tensor:
    """
    Fused FlashAttention forward pass with SRAM online softmax streaming.
    Utilizes real @triton.jit block-tiled kernel on CUDA; falls back seamlessly to PyTorch reference if unavailable.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    d = q.shape[-1]
    # Check eligibility for Triton FlashAttention kernel
    if not is_cuda_available() or not HAS_TRITON or not q.is_cuda or d not in [32, 64, 128]:
        return pytorch_flash_attention(q, k, v, sm_scale, causal=causal)

    # Tensor shapes: [Batch, Heads, Seq_len, Head_dim] or [Batch, Seq_len, Head_dim]
    is_3d = False
    if q.ndim == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        is_3d = True

    Z, H, N_CTX, D = q.shape
    out = torch.empty_like(q)
    L = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _attn_fwd_kernel[grid](
        q, k, v, sm_scale,
        L, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=D,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=2
    )

    if is_3d:
        out = out.squeeze(1)

    return out
