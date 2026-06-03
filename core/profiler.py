import time
import torch
from typing import Callable, Any, Tuple, Dict

def profile_op(op: Callable[..., Any], *args: Any, warmups: int = 10, reps: int = 100, **kwargs: Any) -> float:
    """
    Measures the execution time of a callable operation.
    If execution is on GPU (CUDA), uses CUDA events for precise microsecond timings.
    Otherwise, defaults to high-resolution system performance counter timings.
    
    Returns the average execution duration in milliseconds.
    """
    # Verify execution device context
    is_cuda = False
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            is_cuda = True
            break
    for v in kwargs.values():
        if isinstance(v, torch.Tensor) and v.is_cuda:
            is_cuda = True
            break

    # Warmup runs
    for _ in range(warmups):
        op(*args, **kwargs)
        
    if is_cuda:
        # CUDA Timing
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(reps):
            op(*args, **kwargs)
        end_event.record()
        
        torch.cuda.synchronize()
        # Returns duration in milliseconds
        return start_event.elapsed_time(end_event) / reps
    else:
        # CPU Timing
        start_time = time.perf_counter()
        for _ in range(reps):
            op(*args, **kwargs)
        end_time = time.perf_counter()
        # Returns duration in milliseconds
        return ((end_time - start_time) / reps) * 1000.0

def estimate_metrics(duration_ms: float, bytes_transferred: int, flops_count: int = 0) -> Tuple[float, float]:
    """
    Calculates operational metrics based on duration and bytes/flops.
    
    Returns a Tuple containing:
      - Memory Bandwidth in GB/s
      - Compute Throughput in TFLOPs
    """
    duration_sec = duration_ms / 1000.0
    if duration_sec <= 0:
        return 0.0, 0.0
        
    # Bandwidth = Bytes / Time / 10^9
    bandwidth_gbs = bytes_transferred / duration_sec / 1e9
    
    # TFLOPs = FLOPs / Time / 10^12
    tflops = flops_count / duration_sec / 1e12 if flops_count > 0 else 0.0
    
    return bandwidth_gbs, tflops
