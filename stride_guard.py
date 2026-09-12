"""
TritonForge: GPU Tensor Stride Safety Guard.
Fixes the critical stride bug where non-contiguous tensors passed to Triton
kernels cause silent data corruption (incorrect results, no error raised).
This is a correctness-critical fix — production ML systems at Google/Meta
enforce contiguity checks at every kernel boundary.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def ensure_contiguous(tensor, name: str = "tensor"):
    """
    Guarantees the input tensor is contiguous in memory before dispatching
    to any Triton kernel.

    WHY: Triton kernels compute pointer offsets using stride metadata.
    If a tensor is a non-contiguous view (e.g., after a .permute(), .t(),
    or slice), the strides no longer reflect a packed memory layout.
    This causes the kernel to read from incorrect memory addresses,
    producing silently wrong results — the hardest class of bug to debug.

    INVARIANT: The returned tensor is always row-major (C-contiguous).
    """
    if not HAS_TORCH:
        return tensor

    if not tensor.is_contiguous():
        logger.debug(
            f"Tensor '{name}' is non-contiguous (strides={tensor.stride()}). "
            f"Calling .contiguous() before kernel dispatch."
        )
        return tensor.contiguous()
    return tensor


def safe_triton_dispatch(kernel_fn, *tensors, tensor_names: Optional[list] = None, **kwargs):
    """
    Wrapper for all Triton kernel calls that automatically enforces contiguity.

    Usage:
        output = safe_triton_dispatch(my_triton_kernel, input_a, input_b, grid=grid)
    """
    names = tensor_names or [f"arg_{i}" for i in range(len(tensors))]
    safe_tensors = [ensure_contiguous(t, n) for t, n in zip(tensors, names)]
    return kernel_fn(*safe_tensors, **kwargs)
