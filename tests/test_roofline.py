import torch
from core.kernel_factory import KernelFactory, ExecutionProvider
from core.roofline_analyzer import RooflineAnalyzer

def test_kernel_factory_provider_selection():
    factory = KernelFactory()
    cpu_tensor = torch.randn(10, 10)
    info = factory.get_provider_info(cpu_tensor)
    
    assert info["selected_provider"] == "PYTORCH_EAGER"
    assert info["is_cuda"] is False

    factory_forced = KernelFactory(force_provider=ExecutionProvider.CUDA_CPP)
    assert factory_forced.select_provider(cpu_tensor) == ExecutionProvider.CUDA_CPP

def test_roofline_analyzer_memory_bound_detection():
    analyzer = RooflineAnalyzer(peak_tflops_fp32=8.1, peak_hbm_gbps=320.0)
    res = analyzer.analyze_kernel(total_flops=4096 * 2, total_bytes=4096 * 4, execution_time_ms=0.380)
    
    assert res["is_memory_bound"] is True
    assert res["operational_intensity_flops_per_byte"] < res["ridge_point_flops_per_byte"]
    assert res["achieved_hbm_gbps"] > 0.0
