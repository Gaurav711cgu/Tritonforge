import os
import torch
from tritonforge.core.router import is_cuda_available
from tritonforge.kernels.norm import pytorch_rmsnorm

_cuda_ext = None


def load_cuda_ext():
    global _cuda_ext
    if _cuda_ext is not None:
        return _cuda_ext

    if not is_cuda_available():
        return None

    try:
        from torch.utils.cpp_extension import load_inline

        cu_file = os.path.join(os.path.dirname(__file__), "rmsnorm_cuda.cu")
        if os.path.exists(cu_file):
            with open(cu_file, "r") as f:
                cuda_source = f.read()

            cpp_source = "torch::Tensor rmsnorm_cuda_forward(torch::Tensor x, torch::Tensor weight, float eps);"

            _cuda_ext = load_inline(
                name="rmsnorm_cuda_ext",
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                functions=["forward"],
                verbose=False
            )
            return _cuda_ext
    except Exception as e:
        print(f"[Warning] CUDA C++ Extension Compilation skipped: {e}")
        return None


def rmsnorm_cuda_cpp(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Executes raw CUDA C++ warp-reduced RMSNorm kernel if CUDA compiler is present,
    else falls back cleanly to PyTorch reference implementation.
    """
    ext = load_cuda_ext()
    if ext is not None and x.is_cuda:
        return ext.forward(x, weight, eps)
    return pytorch_rmsnorm(x, weight, eps)
