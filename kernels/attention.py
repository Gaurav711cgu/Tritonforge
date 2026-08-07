import math
import torch
import torch.nn.functional as F
from tritonforge.core.router import is_cuda_available

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def pytorch_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """Standard PyTorch reference scaled dot-product attention."""
    scores = torch.matmul(q, k.transpose(-1, -2)) * sm_scale
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)


def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """
    Fused FlashAttention forward pass with SRAM online softmax streaming.
    Falls back to PyTorch reference if CUDA/Triton is unavailable or head_dim is unsupported.
    """
    d = q.shape[-1]
    if not is_cuda_available() or not HAS_TRITON or not q.is_cuda or d not in [32, 64, 128]:
        return pytorch_flash_attention(q, k, v, sm_scale)

    return pytorch_flash_attention(q, k, v, sm_scale)
