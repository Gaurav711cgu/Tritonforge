"""
TritonForge Eager Fallback Wrapper.
Provides hardware-aware execution fallback if Triton compilation fails.
"""
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def triton_fallback_guard(triton_kernel: Callable, pytorch_fallback: Callable):
    """
    Decorator/Wrapper to catch Triton compilation errors (e.g., incompatible 
    CUDA drivers, PTX errors) and transparently route the tensor operation 
    to standard PyTorch eager mode.
    """
    def wrapper(*args, **kwargs):
        if not HAS_TRITON:
            logger.warning("Triton not installed. Falling back to PyTorch eager execution.")
            return pytorch_fallback(*args, **kwargs)
            
        try:
            # Attempt to execute the Triton kernel
            return triton_kernel(*args, **kwargs)
        except Exception as e:
            # Catching generic Exception since triton.compiler.CompilationError 
            # might not be available if Triton is missing or fundamentally broken
            error_str = str(e).lower()
            if "compilation" in error_str or "ptx" in error_str or "cuda" in error_str:
                logger.error(f"Triton compilation failed: {e}. Falling back to PyTorch eager.")
                return pytorch_fallback(*args, **kwargs)
            # Re-raise if it's a legitimate runtime/logic error, not a hardware/compiler failure
            raise e
            
    return wrapper
