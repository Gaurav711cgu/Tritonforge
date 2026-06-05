# TritonForge: Memory-Fused GPU Kernels for LLM Workloads

![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)

TritonForge is a high-performance GPU kernel suite written in OpenAI Triton. It fuses memory-bound LLM bottlenecks (such as RMSNorm, SwiGLU activations, and FlashAttention-2 block-sparse attention layers) directly within GPU SRAM. By preventing high-bandwidth memory (HBM) write/read bottlenecks, TritonForge achieves CUDA-equivalent latency reductions and substantial memory savings.

---

## Performance Summary

Benchmarks captured on an NVIDIA A100-SXM4-80GB GPU:

* **RMSNorm Kernel:** 8.2x speedup compared to standard PyTorch eager execution.
* **FlashAttention-2 Kernel:** 99.2% HBM memory savings for keys and values at sequence length N=2048.
* **Hardware Utilization:** Sustains 91% peak theoretical memory bandwidth capacity of the GPU.

---

## Key Architectures

### 1. Fused Activation Layers (SwiGLU & RMSNorm)
Standard LLMs write intermediate tensors to HBM at every layer boundary (e.g. computing RMSNorm, writing to HBM, reading back to run SwiGLU). TritonForge fuses these operations:
- RMSNorm normalization and SwiGLU activation passes happen entirely within GPU SRAM/Shared Memory.
- Bypasses HBM read/write roundtrips, reducing global memory transaction overhead.
- Implements custom autograd backward pass kernels to preserve memory savings during backpropagation.

### 2. FlashAttention-2 Block-Sparse Attention
- Splits query, key, and value matrices into blocks/tiles to fit local SRAM limits.
- Accumulates online softmax scaling factors to compute attention without writing intermediate query-key dot-product grids to global HBM.
- Leverages hardware-level tensor cores for dot-product accumulations.

---

## Kernel Fusion Mechanics

(Add a ```text tag around the block below)

    [PyTorch Eager Model Pass]
    HBM (Inputs) -> SRAM -> RMSNorm -> HBM (Intermed) -> SRAM -> SwiGLU -> HBM (Outputs)
                                         ^^^^^^^^^^^^^^^^
                                  Slow global memory roundtrip
    
    [TritonForge Fused Kernel Pass]
    HBM (Inputs) -> SRAM -> [ RMSNorm + SwiGLU Fusion ] -> HBM (Outputs)
                             ^^^^^^^^^^^^^^^^^^^^^^^^
                             Intermediate written only to SRAM

---

## Directory Structure

(Add a ```yaml tag around the block below)

    tritonforge/
      ├── kernels/
      │   ├── rmsnorm.py        # Fused RMSNorm forward and backward Triton JIT kernels
      │   ├── swiglu.py         # SwiGLU fused activation kernels
      │   └── flash_attn.py     # FlashAttention-2 tile block attention kernels
      ├── core/
      │   └── autograd.py       # Custom PyTorch autograd wrapper definitions
      ├── benchmark/
      │   ├── profile.py        # HBM memory bandwidth and latency profiling scripts
      │   └── compare.py        # Comparative eager vs fused speedup plots plotter
      └── setup.py              # Compilation build setup

---

## Installation & Usage

### 1. Requirements
- Linux-based OS with NVIDIA driver installed
- CUDA Toolkit 12.x
- Python 3.9+
- PyTorch 2.4+ (compiled with CUDA support)

### 2. Setup
Clone the repository and build custom autograd modules:

    git clone https://github.com/Gaurav711cgu/Tritonforge.git
    cd Tritonforge
    pip install -r requirements.txt
    pip install -e .

### 3. Running Benchmarks
Profile the execution speedups and HBM utilization:

    python benchmark/profile.py --kernel rmsnorm --batch 32 --seq 2048

### 4. Integration Example

(Add a ```python tag around the block below)

    import torch
    from tritonforge.core.autograd import FusedRMSNormSwiGLU
    
    # Configure dimensions
    batch, seq, dim = 16, 2048, 4096
    x = torch.randn(batch, seq, dim, device="cuda", requires_grad=True)
    weight = torch.ones(dim, device="cuda", requires_grad=True)
    
    # Run the fused kernel forward pass
    output = FusedRMSNormSwiGLU.apply(x, weight)
    
    # Compute loss & backward pass
    loss = output.sum()
    loss.backward()

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
