import torch
from enum import Enum
from typing import Dict, Any, Callable

class ExecutionProvider(Enum):
    CUDA_CPP = "CUDA_CPP"
    OPENAI_TRITON = "OPENAI_TRITON"
    PYTORCH_EAGER = "PYTORCH_EAGER"

class KernelFactory:
    """Strategy & Factory Pattern for GPU Execution Engine Selection.
    Dynamically routes tensor workloads to CUDA C++ warp kernels, Triton JIT kernels,
    or PyTorch Eager CPU fallbacks based on hardware compute capabilities.
    """
    def __init__(self, force_provider: ExecutionProvider = None):
        self.force_provider = force_provider

    def select_provider(self, tensor: torch.Tensor) -> ExecutionProvider:
        if self.force_provider:
            return self.force_provider
        
        if tensor.is_cuda:
            capability = torch.cuda.get_device_capability(tensor.device)
            if capability[0] >= 7:  # Volta / Turing / Ampere / Hopper
                return ExecutionProvider.CUDA_CPP
            return ExecutionProvider.OPENAI_TRITON
        
        return ExecutionProvider.PYTORCH_EAGER

    def get_provider_info(self, tensor: torch.Tensor) -> Dict[str, Any]:
        provider = self.select_provider(tensor)
        return {
            "selected_provider": provider.value,
            "device": str(tensor.device),
            "dtype": str(tensor.dtype),
            "is_cuda": tensor.is_cuda
        }
