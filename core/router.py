import torch

def is_cuda_available() -> bool:
    """Checks if CUDA is available and functional."""
    return torch.cuda.is_available()
