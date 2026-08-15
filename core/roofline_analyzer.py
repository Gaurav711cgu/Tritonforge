from typing import Dict, Any

class RooflineAnalyzer:
    """Roofline Performance Model & Operational Intensity Analyzer.
    Calculates operational intensity (FLOPs / DRAM Byte transferred) and determines
    whether a kernel execution is memory-bandwidth bound or compute-bound.
    """
    def __init__(self, peak_tflops_fp32: float = 8.1, peak_hbm_gbps: float = 320.0):
        self.peak_tflops_fp32 = peak_tflops_fp32
        self.peak_hbm_gbps = peak_hbm_gbps
        # Ridge point = Peak TFLOPS / Peak HBM Bandwidth (FLOPs / Byte)
        self.ridge_point = (peak_tflops_fp32 * 1e12) / (peak_hbm_gbps * 1e9)

    def analyze_kernel(self, total_flops: float, total_bytes: float, execution_time_ms: float) -> Dict[str, Any]:
        operational_intensity = total_flops / max(total_bytes, 1.0)
        achieved_tflops = (total_flops / (execution_time_ms * 1e-3)) / 1e12
        achieved_gbps = (total_bytes / (execution_time_ms * 1e-3)) / 1e9
        
        is_memory_bound = operational_intensity < self.ridge_point
        roofline_ceiling = (operational_intensity * self.peak_hbm_gbps * 1e9) / 1e12 if is_memory_bound else self.peak_tflops_fp32
        efficiency_pct = (achieved_tflops / roofline_ceiling) * 100.0 if roofline_ceiling > 0 else 0.0

        return {
            "operational_intensity_flops_per_byte": float(operational_intensity),
            "ridge_point_flops_per_byte": float(self.ridge_point),
            "is_memory_bound": is_memory_bound,
            "achieved_tflops": float(achieved_tflops),
            "achieved_hbm_gbps": float(achieved_gbps),
            "roofline_efficiency_pct": float(min(efficiency_pct, 100.0))
        }
