"""
TritonForge Hardware-Aware Dynamic Fallback Router.

Implements a 4-layer hardware-resilient routing decorator:
1. Environment Check: HAS_TRITON availability.
2. Hardware Check: CUDA runtime availability and device tensor placement.
3. Shape Validation: Optional custom dimension / alignment guard.
4. JIT Exception Trapping: Catches triton.compiler.CompilationError, PTX errors,
   CUDA OOM or driver faults, and transparently executes the PyTorch reference fallback.
"""

import functools
import logging
from typing import Callable, Optional, Any
import torch

logger = logging.getLogger(__name__)

# Detect Triton compiler availability
try:
    import triton
    import triton.language as tl
    try:
        from triton.compiler.errors import CompilationError
    except ImportError:
        try:
            from triton.compiler import CompilationError
        except ImportError:
            CompilationError = Exception
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    CompilationError = Exception


def is_cuda_available() -> bool:
    """Checks if CUDA is available and functional on the host device."""
    return torch.cuda.is_available()


def triton_route(
    fallback_fn: Callable,
    shape_validator: Optional[Callable[..., bool]] = None,
    log_fallback: bool = False,
):
    """
    4-Layer Fallback Routing Decorator for Triton GPU Kernels.

    Parameters:
    - fallback_fn: Callable PyTorch eager reference implementation.
    - shape_validator: Optional predicate (args, kwargs) -> bool to enforce tensor boundary constraints.
    - log_fallback: Boolean flag to log routing actions at warning/debug levels.
    """
    def decorator(kernel_fn: Callable):
        @functools.wraps(kernel_fn)
        def wrapper(*args, **kwargs):
            # Layer 1: Triton compiler installation check
            if not HAS_TRITON:
                if log_fallback:
                    logger.debug("[triton_route] Triton not installed. Routing to PyTorch fallback.")
                return fallback_fn(*args, **kwargs)

            # Layer 2: CUDA device availability & Tensor placement
            if not is_cuda_available():
                if log_fallback:
                    logger.debug("[triton_route] CUDA not available on host. Routing to PyTorch fallback.")
                return fallback_fn(*args, **kwargs)

            # Verify all input tensors are allocated on a CUDA device
            for arg in args:
                if isinstance(arg, torch.Tensor) and not arg.is_cuda:
                    if log_fallback:
                        logger.debug("[triton_route] Host CPU tensor detected. Routing to PyTorch fallback.")
                    return fallback_fn(*args, **kwargs)
            for v in kwargs.values():
                if isinstance(v, torch.Tensor) and not v.is_cuda:
                    if log_fallback:
                        logger.debug("[triton_route] Host CPU tensor detected in kwargs. Routing to PyTorch fallback.")
                    return fallback_fn(*args, **kwargs)

            # Layer 3: Shape and dimension boundary validation
            if shape_validator is not None:
                try:
                    if not shape_validator(*args, **kwargs):
                        if log_fallback:
                            logger.info("[triton_route] Shape boundary check failed. Routing to PyTorch fallback.")
                        return fallback_fn(*args, **kwargs)
                except Exception as shape_err:
                    logger.warning(f"[triton_route] Shape validator exception: {shape_err}. Routing to fallback.")
                    return fallback_fn(*args, **kwargs)

            # Layer 4: Execution & JIT Compilation error trapping
            try:
                return kernel_fn(*args, **kwargs)
            except (CompilationError, RuntimeError, Exception) as err:
                err_msg = str(err).lower()
                # Check for JIT compilation errors, PTX syntax errors, device limits, or CUDA failures
                is_compilation_or_hardware_error = (
                    isinstance(err, CompilationError)
                    or "compilation" in err_msg
                    or "ptx" in err_msg
                    or "cuda" in err_msg
                    or "out of memory" in err_msg
                    or "not supported" in err_msg
                    or "illegal memory" in err_msg
                    or "device-side assert" in err_msg
                )
                if is_compilation_or_hardware_error:
                    logger.warning(
                        f"[triton_route] Triton kernel execution failed ({type(err).__name__}: {err}). "
                        f"Transparently routing to PyTorch fallback: {fallback_fn.__name__}"
                    )
                    return fallback_fn(*args, **kwargs)
                # Re-raise standard algorithmic/programming bugs
                raise err

        return wrapper
    return decorator
