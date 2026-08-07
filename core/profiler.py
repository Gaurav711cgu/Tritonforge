import time
import torch
from typing import Callable, Tuple, Any


def profile_op(op_func: Callable, *args: Any, warmups: int = 10, reps: int = 100, **kwargs: Any) -> float:
    """
    Profiles average execution latency of op_func(*args, **kwargs) in milliseconds.
    Supports CUDA synchronization if available.
    """
    for _ in range(warmups):
        _ = op_func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(reps):
        _ = op_func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    return ((end_time - start_time) / reps) * 1000.0


def estimate_metrics(latency_ms: float, bytes_transferred: int, flops_count: int = 0) -> Tuple[float, float]:
    """
    Estimates throughput in GB/s and TFLOPS given latency in ms.
    """
    latency_sec = latency_ms / 1000.0
    gb_per_sec = (bytes_transferred / 1e9) / latency_sec if latency_sec > 0 else 0.0
    tflops = (flops_count / 1e12) / latency_sec if latency_sec > 0 else 0.0
    return gb_per_sec, tflops
