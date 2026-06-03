import torch
import logging
from typing import Callable, Any, Optional

# Setup lightweight logger
logger = logging.getLogger("TritonForge.Router")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Check Triton compiler presence on host
try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def is_cuda_available() -> bool:
    """Verifies if an NVIDIA GPU, CUDA driver, and Triton compiler are available to PyTorch."""
    return torch.cuda.is_available() and HAS_TRITON

def triton_route(fallback_fn: Callable[..., Any], shape_validator: Optional[Callable[..., bool]] = None) -> Callable[..., Any]:
    """
    Decorator that intercepts kernel execution calls.
    Routes to the fallback function if:
      1. CUDA/Triton is not available on the current machine.
      2. Any input tensor is not on CUDA (and cannot be automatically moved).
      3. The custom shape validator returns False.
    """
    def decorator(triton_fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 1. Device and Compiler detection gate
            if not is_cuda_available():
                logger.warning(f"CUDA/Triton environment not detected. Routing {triton_fn.__name__} to fallback {fallback_fn.__name__}.")
                return fallback_fn(*args, **kwargs)
            
            # 2. Shape validation gate (custom routing criteria)
            if shape_validator is not None:
                if not shape_validator(*args, **kwargs):
                    logger.warning(f"Input shape check failed for {triton_fn.__name__}. Routing to fallback {fallback_fn.__name__}.")
                    return fallback_fn(*args, **kwargs)
            
            # 3. Verify target tensor placement
            has_cpu_tensors = False
            for idx, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and not arg.is_cuda:
                    has_cpu_tensors = True
                    break
            
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and not v.is_cuda:
                    has_cpu_tensors = True
                    break
            
            if has_cpu_tensors:
                logger.warning(f"Tensors found on CPU. Routing {triton_fn.__name__} to fallback {fallback_fn.__name__}.")
                return fallback_fn(*args, **kwargs)
            
            # 4. Success: Execute optimized Triton path
            try:
                return triton_fn(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error during Triton kernel execution: {e}. Attempting recovery via fallback.")
                return fallback_fn(*args, **kwargs)
                
        return wrapper
    return decorator
