import torch
import math
from tritonforge.core.router import is_cuda_available
from tritonforge.core.profiler import profile_op, estimate_metrics
from tritonforge.core.plotting import plot_speedup_curves, plot_roofline
from tritonforge.kernels.norm import fused_rmsnorm, pytorch_rmsnorm
from tritonforge.kernels.activation import fused_swiglu, pytorch_swiglu
from tritonforge.kernels.attention import fused_attention, pytorch_flash_attention

def run_benchmarks():
    device = "cuda" if is_cuda_available() else "cpu"
    print("=" * 70)
    print(f"TRITONFORGE BENCHMARKING ENGINE | Device: {device.upper()}")
    print("=" * 70)
    
    # -----------------------------------------------------------------
    # Benchmark 1: RMSNorm Profiling
    # -----------------------------------------------------------------
    print("\n[1/3] Benchmarking Fused RMSNorm...")
    sizes = [512, 1024, 2048, 4096, 8192]
    py_times = []
    tr_times = []
    
    # Roofline plotting vectors
    intensities = []
    tflops_achieved = []
    labels = []
    
    print(f"{'Size (Cols)':<12} | {'PyTorch (ms)':<12} | {'Triton/Fallback (ms)':<20} | {'Speedup':<8} | {'Bandwidth (GB/s)':<16}")
    print("-" * 75)
    
    for N in sizes:
        M = 4096  # Fixed rows batch size
        x = torch.randn((M, N), dtype=torch.float32, device=device)
        weight = torch.randn((N,), dtype=torch.float32, device=device)
        
        # 1. Measure Latencies
        py_ms = profile_op(pytorch_rmsnorm, x, weight, warmups=5, reps=50)
        tr_ms = profile_op(fused_rmsnorm, x, weight, warmups=5, reps=50)
        
        py_times.append(py_ms)
        tr_times.append(tr_ms)
        
        # 2. Compute Bandwidth Metrics
        # FP32 = 4 bytes. Reads: X (M*N*4) + W (N*4). Writes: Y (M*N*4).
        bytes_transferred = M * N * 4 * 2 + N * 4
        bandwidth_gbs, _ = estimate_metrics(tr_ms, bytes_transferred, flops_count=0)
        
        # Operational intensity for RMSNorm (Memory Bound):
        # 4 flops per input element (square, add reduction, sqrt, multiply)
        # Intensity = FLOPs / Bytes = (4 * M * N) / (8 * M * N) = 0.5 FLOPs/Byte
        norm_flops = 4 * M * N
        _, achieved_tflops = estimate_metrics(tr_ms, bytes_transferred, flops_count=norm_flops)
        
        speedup = py_ms / tr_ms
        print(f"{N:<12} | {py_ms:<12.4f} | {tr_ms:<20.4f} | {speedup:<8.2f}x | {bandwidth_gbs:<16.2f}")
        
        # Save points for Roofline plot if GPU is active
        if device == "cuda":
            intensities.append(norm_flops / bytes_transferred)
            tflops_achieved.append(achieved_tflops)
            labels.append(f"RMSNorm-{N}")
            
    # Draw latency charts
    plot_speedup_curves(sizes, py_times, tr_times, xlabel="Column Dimension Size", title="RMSNorm Speedup Performance", output_path="rmsnorm_speedup_curves.png")

    # -----------------------------------------------------------------
    # Benchmark 2: SwiGLU Gated Activation Profiling
    # -----------------------------------------------------------------
    print("\n[2/3] Benchmarking Fused SwiGLU Gated Activation...")
    sizes_swiglu = [256, 512, 1024, 2048, 4096]
    py_times_swi = []
    tr_times_swi = []
    
    print(f"{'Size (Cols)':<12} | {'PyTorch (ms)':<12} | {'Triton/Fallback (ms)':<20} | {'Speedup':<8} | {'Bandwidth (GB/s)':<16}")
    print("-" * 75)
    
    for N in sizes_swiglu:
        M = 4096
        # Input tensor has shape (M, 2*N)
        x = torch.randn((M, 2 * N), dtype=torch.float32, device=device)
        
        # 1. Measure Latencies
        py_ms = profile_op(pytorch_swiglu, x, warmups=5, reps=50)
        tr_ms = profile_op(fused_swiglu, x, warmups=5, reps=50)
        
        py_times_swi.append(py_ms)
        tr_times_swi.append(tr_ms)
        
        # 2. Compute Bandwidth Metrics
        # Reads: X (M * 2*N * 4). Writes: Y (M * N * 4). Total = M * N * 12 bytes.
        bytes_transferred = M * N * 12
        bandwidth_gbs, _ = estimate_metrics(tr_ms, bytes_transferred, flops_count=0)
        
        # SwiGLU FLOPS: sigmoid calculation (~10 FLOPS) + multiplies.
        swiglu_flops = M * N * 15
        _, achieved_tflops = estimate_metrics(tr_ms, bytes_transferred, flops_count=swiglu_flops)
        
        speedup = py_ms / tr_ms
        print(f"{N:<12} | {py_ms:<12.4f} | {tr_ms:<20.4f} | {speedup:<8.2f}x | {bandwidth_gbs:<16.2f}")
        
        if device == "cuda":
            intensities.append(swiglu_flops / bytes_transferred)
            tflops_achieved.append(achieved_tflops)
            labels.append(f"SwiGLU-{N}")
            
    plot_speedup_curves(sizes_swiglu, py_times_swi, tr_times_swi, xlabel="Output Column Dimension Size", title="SwiGLU Speedup Performance", output_path="swiglu_speedup_curves.png")

    # -----------------------------------------------------------------
    # Benchmark 3: FlashAttention Profiling (Compute Bound)
    # -----------------------------------------------------------------
    print("\n[3/3] Benchmarking Tiled Attention...")
    seq_lengths = [256, 512, 1024, 2048]
    print(f"{'Seq Length':<12} | {'PyTorch (ms)':<12} | {'Triton/Fallback (ms)':<20} | {'Speedup':<8} | {'Throughput (TFLOPs)':<18}")
    print("-" * 78)
    
    for seq_len in seq_lengths:
        B, H, d = 4, 8, 64
        q = torch.randn((B, H, seq_len, d), dtype=torch.float32, device=device)
        k = torch.randn((B, H, seq_len, d), dtype=torch.float32, device=device)
        v = torch.randn((B, H, seq_len, d), dtype=torch.float32, device=device)
        
        py_ms = profile_op(pytorch_flash_attention, q, k, v, warmups=5, reps=20)
        tr_ms = profile_op(fused_attention, q, k, v, warmups=5, reps=20)
        
        # Attention Forward FLOPS: 4 * B * H * N^2 * d
        flops = 4 * B * H * (seq_len ** 2) * d
        # Memory bytes: Q, K, V reads + O write. Total = 4 * B * H * N * d * 4 bytes
        bytes_transferred = 4 * B * H * seq_len * d * 4
        
        _, achieved_tflops = estimate_metrics(tr_ms, bytes_transferred, flops_count=flops)
        speedup = py_ms / tr_ms
        print(f"{seq_len:<12} | {py_ms:<12.4f} | {tr_ms:<20.4f} | {speedup:<8.2f}x | {achieved_tflops:<18.4f}")
        
        if device == "cuda":
            intensities.append(flops / bytes_transferred)
            tflops_achieved.append(achieved_tflops)
            labels.append(f"Attention-{seq_len}")
            
    # -----------------------------------------------------------------
    # Roofline Analysis Visualization (only on GPU machines)
    # -----------------------------------------------------------------
    if device == "cuda" and len(intensities) > 0:
        print("\nPlotting Roofline Analysis curves...")
        # Get target device properties
        prop = torch.cuda.get_device_properties(0)
        # Approximate values for roofline limits (e.g. RTX 4090 ~80 TFLOPs FP32, A100 ~19.5 TFLOPs FP32)
        # TritonForge dynamically queries memory bandwidth
        # Memory Clock Rate (kHz) * Bus Width (bits) / 8 / 10^6 = Bandwidth in GB/s
        mem_bandwidth_gbs = (prop.memoryBandwidth) / 1e9 if hasattr(prop, 'memoryBandwidth') else 900.0
        # If property is missing, use default A100/H100 profile bounds
        if mem_bandwidth_gbs == 0:
            mem_bandwidth_gbs = 1500.0 # Standard A100 profile
            
        # Estimating TFLOPs based on multiprocessor count and warp speeds
        # Peak FLOPs = Cores * Clock * 2 operations per clock
        # For simplicity, we establish a robust peak:
        peak_tflops = 19.5 # A100 peak single-precision compute limit
        
        plot_roofline(
            intensities=intensities,
            tflops_achieved=tflops_achieved,
            labels=labels,
            peak_bandwidth_gbs=mem_bandwidth_gbs,
            peak_tflops=peak_tflops,
            output_path="roofline_analysis.png"
        )
        print("Success! Generated roofline_analysis.png and speedup charts.")
        
if __name__ == "__main__":
    run_benchmarks()
