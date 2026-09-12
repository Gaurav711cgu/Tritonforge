"""
profile_runner.py — NVTX-annotated execution runner for NVIDIA Nsight Systems capture.
Usage:
    nsys profile -o profile_output python benchmarks/profile_runner.py
"""

import torch
from tritonforge.kernels.norm import fused_rmsnorm, pytorch_rmsnorm
from tritonforge.kernels.norm_autograd import fused_rmsnorm_autograd
from tritonforge.kernels.activation import fused_swiglu
from tritonforge.kernels.attention import fused_attention


def run_profiled_workload():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running NVTX profiled workload on device: {device}")

    shape = (4096, 2304)
    x = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn((shape[-1],), device=device, dtype=torch.float32, requires_grad=True)
    eps = 1e-6

    # Warmup
    for _ in range(5):
        _ = fused_rmsnorm(x, weight, eps)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # NVTX Range 1: Fused RMSNorm Forward Pass
    try:
        torch.cuda.nvtx.range_push("TritonForge::RMSNorm_Forward")
    except Exception:
        pass

    for _ in range(20):
        y = fused_rmsnorm(x, weight, eps)

    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass

    # NVTX Range 2: RMSNorm Autograd Backward Pass
    try:
        torch.cuda.nvtx.range_push("TritonForge::RMSNorm_Backward")
    except Exception:
        pass

    for _ in range(20):
        y = fused_rmsnorm_autograd(x, weight, eps)
        loss = y.sum()
        loss.backward()

    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass

    print("NVTX profiled workload completed successfully.")


if __name__ == "__main__":
    run_profiled_workload()
