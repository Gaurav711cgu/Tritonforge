"""
TritonForge GPU Kernel Benchmark & Profiling Suite
Evaluates custom Triton & CUDA kernels vs. PyTorch Eager Baseline:
  1. Fused RMSNorm vs PyTorch Eager (variance + rsqrt + weight mul)
  2. Block-tiled Attention vs PyTorch SDPA (scaled dot-product attention)
  3. Fused SwiGLU vs PyTorch Eager (sigmoid gating + elementwise mul)
  4. Fused RMSNorm + Linear Projection
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from tritonforge.kernels.norm import fused_rmsnorm, pytorch_rmsnorm
from tritonforge.kernels.attention import fused_attention, pytorch_flash_attention
from tritonforge.kernels.activation import fused_swiglu, pytorch_swiglu
from tritonforge.kernels.fused_norm_linear import FusedRMSNormLinear


def time_kernel(fn, *args, warmup=10, iters=20, device="cpu"):
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iters):
            _ = fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fn(*args)
        elapsed_ms = ((time.perf_counter() - t0) / iters) * 1000.0
    return elapsed_ms


def run_all_benchmarks():
    print("=" * 80)
    print("TRITONFORGE BENCHMARK SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Device: {device}\n")

    results = []

    # 1. RMSNorm Benchmark
    print("[1/4] Benchmarking RMSNorm...")
    for dim in [1024, 2048, 4096]:
        x = torch.randn(16, 512, dim, device=device, dtype=torch.float32)
        w = torch.ones(dim, device=device, dtype=torch.float32)

        t_eager = time_kernel(pytorch_rmsnorm, x, w, device=device)
        t_fused = time_kernel(fused_rmsnorm, x, w, device=device)
        speedup = t_eager / max(t_fused, 1e-6)

        out_eager = pytorch_rmsnorm(x, w)
        out_fused = fused_rmsnorm(x, w)
        max_err = float((out_eager - out_fused).abs().max().item())

        results.append({
            "kernel": "RMSNorm",
            "shape": f"(16, 512, {dim})",
            "eager_ms": round(t_eager, 4),
            "fused_ms": round(t_fused, 4),
            "speedup": round(speedup, 2),
            "max_abs_err": f"{max_err:.2e}"
        })
        print(f"  Dim: {dim:<5} | Eager: {t_eager:.4f} ms | Fused: {t_fused:.4f} ms | Speedup: {speedup:.2f}x | Max Err: {max_err:.2e}")

    # 2. Attention Benchmark
    print("\n[2/4] Benchmarking Attention...")
    for seq_len in [128, 256, 512]:
        B, H, D = 4, 8, 64
        q = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, seq_len, D, device=device, dtype=torch.float32)
        scale = 1.0 / (D ** 0.5)

        t_eager = time_kernel(pytorch_flash_attention, q, k, v, scale, False, device=device)
        t_fused = time_kernel(fused_attention, q, k, v, scale, False, device=device)
        speedup = t_eager / max(t_fused, 1e-6)

        out_eager = pytorch_flash_attention(q, k, v, scale)
        out_fused = fused_attention(q, k, v, scale)
        max_err = float((out_eager - out_fused).abs().max().item())

        results.append({
            "kernel": "Attention",
            "shape": f"(4, 8, {seq_len}, 64)",
            "eager_ms": round(t_eager, 4),
            "fused_ms": round(t_fused, 4),
            "speedup": round(speedup, 2),
            "max_abs_err": f"{max_err:.2e}"
        })
        print(f"  SeqLen: {seq_len:<4} | Eager: {t_eager:.4f} ms | Fused: {t_fused:.4f} ms | Speedup: {speedup:.2f}x | Max Err: {max_err:.2e}")

    # 3. SwiGLU Benchmark
    print("\n[3/4] Benchmarking SwiGLU...")
    for dim in [1024, 2048, 4096]:
        x = torch.randn(16, 512, dim * 2, device=device, dtype=torch.float32)

        t_eager = time_kernel(pytorch_swiglu, x, device=device)
        t_fused = time_kernel(fused_swiglu, x, device=device)
        speedup = t_eager / max(t_fused, 1e-6)

        out_eager = pytorch_swiglu(x)
        out_fused = fused_swiglu(x)
        max_err = float((out_eager - out_fused).abs().max().item())

        results.append({
            "kernel": "SwiGLU",
            "shape": f"(16, 512, {dim*2})",
            "eager_ms": round(t_eager, 4),
            "fused_ms": round(t_fused, 4),
            "speedup": round(speedup, 2),
            "max_abs_err": f"{max_err:.2e}"
        })
        print(f"  Dim: {dim:<5} | Eager: {t_eager:.4f} ms | Fused: {t_fused:.4f} ms | Speedup: {speedup:.2f}x | Max Err: {max_err:.2e}")

    # 4. Fused RMSNorm + Linear
    print("\n[4/4] Benchmarking Fused RMSNorm + Linear...")
    layer = FusedRMSNormLinear(d_model=1024, out_features=1024).to(device)
    x = torch.randn(8, 256, 1024, device=device, dtype=torch.float32)

    def eager_baseline(x_in):
        normed = pytorch_rmsnorm(x_in, layer.norm_weight, layer.eps)
        return layer.linear(normed)

    t_eager = time_kernel(eager_baseline, x, device=device)
    t_fused = time_kernel(layer, x, device=device)
    speedup = t_eager / max(t_fused, 1e-6)

    out_eager = eager_baseline(x)
    out_fused = layer(x)
    max_err = float((out_eager - out_fused).abs().max().item())

    results.append({
        "kernel": "FusedRMSNormLinear",
        "shape": "(8, 256, 1024)",
        "eager_ms": round(t_eager, 4),
        "fused_ms": round(t_fused, 4),
        "speedup": round(speedup, 2),
        "max_abs_err": f"{max_err:.2e}"
    })
    print(f"  Config: (8, 256, 1024) | Eager: {t_eager:.4f} ms | Fused: {t_fused:.4f} ms | Speedup: {speedup:.2f}x | Max Err: {max_err:.2e}")

    # Save output
    output_path = base_dir / "benchmarks" / "results.json"
    with open(output_path, "w") as f:
        json.dump({
            "device": str(device),
            "timestamp": time.time(),
            "results": results
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Benchmark results saved to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_benchmarks()
