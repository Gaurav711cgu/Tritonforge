from typing import List, Dict, Any


def plot_speedup_curves(
    sizes: List[int],
    py_times: List[float],
    tr_times: List[float],
    xlabel: str = "Dimension Size",
    title: str = "Speedup Performance",
    output_path: str = "speedup_curves.png"
) -> None:
    """Mock/stub plotting speedup curves."""
    pass


def plot_roofline(
    intensities: List[float],
    tflops_achieved: List[float],
    labels: List[str],
    peak_bandwidth_gbs: float,
    peak_tflops: float,
    output_path: str = "roofline_analysis.png"
) -> None:
    """Mock/stub plotting roofline analysis."""
    pass


def plot_benchmark_results(results: List[Dict[str, Any]], title: str, output_file: str) -> None:
    """Mock/stub plotting benchmark results."""
    pass
