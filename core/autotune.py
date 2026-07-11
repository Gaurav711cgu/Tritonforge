import json
import logging

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

logger = logging.getLogger("TritonForge.Autotune")

class TritonCacheManager:
    """
    Centralized Autotuning Cache State Management.
    Provides a registry for all Triton kernels to register themselves,
    and a unified loader to apply offline AOT (Ahead-of-Time) tuning
    configurations, preventing cold-start compile latency.
    """
    def __init__(self):
        self._registry = {}
        
    def register_kernel(self, name: str, kernel_func):
        """Register a Triton JIT kernel for AOT cache injection."""
        self._registry[name] = kernel_func
        
    def load_cache(self, cache_path: str):
        """Loads a global cache JSON and injects triton.Config into kernel caches."""
        if not HAS_TRITON:
            return
            
        try:
            with open(cache_path, "r") as f:
                global_cache = json.load(f)
        except FileNotFoundError:
            logger.warning(f"No cache found at '{cache_path}'.")
            return
            
        pinned_total = 0
        
        # Detect legacy format (if keys are shapes)
        if any("," in k for k in global_cache.keys()):
            logger.info("Detected legacy tune cache format. Upgrading to centralized schema.")
            global_cache = {"_rmsnorm_linear_gemm_kernel": global_cache}
            
        for kernel_name, shapes_dict in global_cache.items():
            kernel_fn = self._registry.get(kernel_name)
            if not kernel_fn:
                logger.warning(f"Kernel '{kernel_name}' in cache but not registered.")
                continue
                
            pinned = 0
            for shape_key, cfg in shapes_dict.items():
                if cfg.get("path") == "gemv":
                    continue
                M, N, K = map(int, shape_key.split(","))
                key = (M, N, K)
                
                kwargs = {
                    "BLOCK_M": cfg.get("BLOCK_M", 32),
                    "BLOCK_N": cfg.get("BLOCK_N", 64),
                    "BLOCK_K": cfg.get("BLOCK_K", 32),
                }
                
                # Dynamic shape validation logic injection
                if kernel_name == "_rmsnorm_linear_gemm_kernel":
                    kwargs["IS_ALIGNED"] = (M % cfg.get("BLOCK_M", 32) == 0 and
                                            K % cfg.get("BLOCK_K", 32) == 0 and
                                            N % cfg.get("BLOCK_N", 64) == 0)
                
                kernel_fn.cache[key] = triton.Config(
                    kwargs=kwargs,
                    num_warps=cfg.get("num_warps", 4),
                    num_stages=cfg.get("num_stages", 3),
                )
                pinned += 1
                
            pinned_total += pinned
            
        logger.info(f"Loaded {pinned_total} pre-tuned configs across {len(global_cache)} kernels.")

cache_manager = TritonCacheManager()
