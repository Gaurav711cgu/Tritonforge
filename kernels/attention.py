import math
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

def pytorch_flash_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    sm_scale: float = None
) -> torch.Tensor:
    """Standard PyTorch fallback using optimized scaled_dot_product_attention."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=sm_scale
    )

# =====================================================================
# TRITON FLASH ATTENTION FORWARD KERNEL (Compiled only if Triton library is present)
# =====================================================================

if HAS_TRITON:
    @triton.jit
    def _flash_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr,      # Tensor pointers
        M_ptr, L_ptr,                    # Row maximums & denominator trackers
        stride_qb, stride_qh, stride_qn, # Strides for Q (batch, head, seq)
        stride_kb, stride_kh, stride_kn, # Strides for K
        stride_vb, stride_vh, stride_vn, # Strides for V
        stride_ob, stride_oh, stride_on, # Strides for O
        H,                               # Number of heads
        N_ctx,                           # Sequence length
        d_model,                         # Head dimension
        sm_scale,                        # Scaling factor
        BLOCK_M: tl.constexpr,           # Query tiling block size
        BLOCK_N: tl.constexpr,           # Key/Value tiling block size
    ):
        start_m = tl.program_id(0)       # Query tile index
        off_hz = tl.program_id(1)        # Batch * Head combined index
        
        off_b = off_hz // H
        off_h = off_hz % H
        
        q_offset_base = off_b * stride_qb + off_h * stride_qh
        k_offset_base = off_b * stride_kb + off_h * stride_kh
        v_offset_base = off_b * stride_vb + off_h * stride_vh
        o_offset_base = off_b * stride_ob + off_h * stride_oh
        
        col_offsets = tl.arange(0, d_model)
        q_row_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        q_mask = q_row_offsets[:, None] < N_ctx
        
        q = tl.load(Q_ptr + q_offset_base + q_row_offsets[:, None] * stride_qn + col_offsets[None, :], mask=q_mask, other=0.0)
        
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)
        
        for start_n in range(0, N_ctx, BLOCK_N):
            k_row_offsets = start_n + tl.arange(0, BLOCK_N)
            k_mask = k_row_offsets[None, :] < N_ctx
            
            k = tl.load(K_ptr + k_offset_base + k_row_offsets[None, :] * stride_kn + col_offsets[:, None], mask=k_mask, other=0.0)
            
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            
            qk = tl.where(q_row_offsets[:, None] < N_ctx, qk, float("-inf"))
            qk = tl.where(k_row_offsets[None, :] < N_ctx, qk, float("-inf"))
            
            m_ij = tl.max(qk, axis=1)
            
            m_next = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp(m_i - m_next)
            
            p = tl.math.exp(qk - m_next[:, None])
            p = tl.where(k_row_offsets[None, :] < N_ctx, p, 0.0)
            
            v = tl.load(V_ptr + v_offset_base + k_row_offsets[:, None] * stride_vn + col_offsets[None, :], mask=k_mask[:, None], other=0.0)
            
            l_ij = tl.sum(p, axis=1)
            l_i = l_i * alpha + l_ij
            
            acc = acc * alpha[:, None]
            acc += tl.dot(p, v)
            
            m_i = m_next
            
        o = acc / l_i[:, None]
        tl.store(O_ptr + o_offset_base + q_row_offsets[:, None] * stride_on + col_offsets[None, :], o, mask=q_mask)
        
        if M_ptr is not None:
            tl.store(M_ptr + off_hz * N_ctx + q_row_offsets, m_i, mask=q_row_offsets < N_ctx)
        if L_ptr is not None:
            tl.store(L_ptr + off_hz * N_ctx + q_row_offsets, l_i, mask=q_row_offsets < N_ctx)
else:
    _flash_attn_fwd_kernel = None

# =====================================================================
# SYSTEM GATEWAY & SHAPE VALIDATOR
# =====================================================================

def attention_shape_validator(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    sm_scale: float = None
) -> bool:
    """FlashAttention tiling restricts execution to power-of-two dimensions."""
    d = q.shape[-1]
    return d in {32, 64, 128, 256}

@triton_route(fallback_fn=pytorch_flash_attention, shape_validator=attention_shape_validator)
def fused_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    sm_scale: float = None
) -> torch.Tensor:
    """
    Tiled FlashAttention forward pass using OpenAI Triton.
    Automatically routes to PyTorch scaled_dot_product_attention if running on CPU
    or if head dimension size is non-standard.
    """
    assert HAS_TRITON, "Cannot invoke fused_attention without Triton compiler installed."
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "All Q, K, V tensors must be contiguous."
    
    B, H, N, d = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
        
    o = torch.empty_like(q)
    metadata_m = torch.empty((B * H, N), dtype=torch.float32, device=q.device)
    metadata_l = torch.empty((B * H, N), dtype=torch.float32, device=q.device)
    
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    
    _flash_attn_fwd_kernel[grid](
        q, k, v, o,
        metadata_m, metadata_l,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, N, d, sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return o
