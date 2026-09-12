"""
TritonForge Kernel Autotuner & Configuration Cache.
Provides configuration search spaces and caching for Triton GPU kernels.
"""
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class TuneCache:
    """In-memory and persistent autotuning cache for optimal tile/block layouts."""
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()


global_tune_cache = TuneCache()


def get_swiglu_autotune_configs():
    """Returns standard autotune configs for SwiGLU kernel."""
    if not HAS_TRITON:
        return []
    return [
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ]


def get_rmsnorm_autotune_configs():
    """Returns standard autotune configs for RMSNorm kernel."""
    if not HAS_TRITON:
        return []
    return [
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ]
