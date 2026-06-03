import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_speedup_curves(
    sizes: List[int],
    pytorch_times: List[float],
    triton_times: List[float],
    xlabel: str = "Dimension Size",
    title: str = "Triton vs. PyTorch Latency Comparison",
    output_path: str = "speedup_curves.png"
):
    """
    Plots benchmark latency curves and speedup ratios.
    - Top plot: Latency (ms) comparison.
    - Bottom plot: Speedup ratio (PyTorch time / Triton time).
    """
    sizes_arr = np.array(sizes)
    py_arr = np.array(pytorch_times)
    tr_arr = np.array(triton_times)
    speedups = py_arr / tr_arr
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Latency Curves
    ax1.plot(sizes_arr, py_arr, marker='o', color='#E11D48', label='PyTorch Eager', linewidth=2)
    ax1.plot(sizes_arr, tr_arr, marker='s', color='#2563EB', label='Triton Optimized', linewidth=2)
    ax1.set_ylabel("Execution Time (ms)")
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # 2. Speedup Curve
    ax2.plot(sizes_arr, speedups, marker='^', color='#10B981', linewidth=2.5, label='Speedup Factor')
    ax2.axhline(y=1.0, color='#6B7280', linestyle=':', label='Break-Even')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Speedup Ratio (x)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_roofline(
    intensities: List[float],
    tflops_achieved: List[float],
    labels: List[str],
    peak_bandwidth_gbs: float,
    peak_tflops: float,
    output_path: str = "roofline_analysis.png"
):
    """
    Plots a standard Roofline Model comparison chart.
    X-axis: Arithmetic Intensity (FLOPs / Byte) in log scale.
    Y-axis: Attained Performance (TFLOPs) in log scale.
    """
    # Create evaluation intensity space (logarithmic)
    x_intensity = np.logspace(-3, 3, 1000)
    
    # Roofline performance bounds
    # Limit = min(Peak TFLOPs, Bandwidth * Intensity)
    # Bandwidth in GB/s is equivalent to TFLOPs/s per FLOP/Byte (since GB = 10^9, TB = 10^12)
    # Bandwidth * Intensity = (Bandwidth / 1000) * Intensity (TFLOPs)
    bandwidth_limit = (peak_bandwidth_gbs / 1000.0) * x_intensity
    roofline_limit = np.minimum(peak_tflops, bandwidth_limit)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Roofline bounds
    plt.loglog(x_intensity, roofline_limit, color='#374151', label='Hardware Roofline Limit', linewidth=2.5)
    plt.axhline(y=peak_tflops, color='#EF4444', linestyle='--', label=f'Peak Compute limit ({peak_tflops} TFLOPs)', alpha=0.7)
    
    # Plot measured operators
    colors = plt.cm.get_cmap('viridis', len(intensities))
    for idx, (intensity, perf, label) in enumerate(zip(intensities, tflops_achieved, labels)):
        plt.scatter(intensity, perf, color=colors(idx), s=120, zorder=5, label=label, edgecolor='black')
        
    plt.xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=11)
    plt.ylabel("Attained Performance (TFLOPs)", fontsize=11)
    plt.title("TritonForge Roofline Performance Model", fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, which="both", linestyle=':', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', loc='lower right')
    
    # Set display limits
    plt.xlim(min(min(intensities) * 0.5, 0.01), max(max(intensities) * 2.0, 100.0))
    plt.ylim(min(min(tflops_achieved) * 0.5, 0.001), peak_tflops * 1.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
