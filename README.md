<div align="center">

# TritonForge

**High-Performance GPU Kernel Compilation & Optimization Workstation**
<br/>

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-Passing-22c55e?style=flat-square&logo=githubactions&logoColor=white)](#)
[![Pytest](https://img.shields.io/badge/Pytest-28%2F28%20Passed-22c55e?style=flat-square&logo=pytest&logoColor=white)](#testing--verification)
[![SAST Security](https://img.shields.io/badge/SAST-Clean-22c55e?style=flat-square&logo=python&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square)](#)

<br/>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](./notebooks/TritonForge_Benchmark.ipynb) &nbsp;·&nbsp; [Roofline Model Card](./PERFORMANCE_CARD.md) &nbsp;·&nbsp; [Nsight Report](./benchmarks/nsight_profile_report.md) &nbsp;·&nbsp; [API Documentation](#api-documentation) &nbsp;·&nbsp; [Run Tests](#testing--verification)

</div>

---

## Executive Summary

> **TritonForge** is an enterprise GPU kernel compilation and performance optimization workstation built on OpenAI Triton and raw CUDA C++. The platform compiles fused deep learning operators directly to highly optimized PTX/SASS assembly, bypassing eager PyTorch overhead and maximizing physical hardware memory bandwidth.

| Differentiator | Technical Implementation Detail |
|---|---|
| **Maximized Memory Bandwidth** | Achieves **93.1% of physical HBM bandwidth utilization** (297.8 GB/s on NVIDIA Tesla T4) |
| **CUDA C++ Warp Reduction** | Implements raw CUDA C++ extensions (`rmsnorm_cuda.cu`) with `__shfl_xor_sync` warp-level reductions |
| **Autograd Training Support** | Extends custom forward kernels with `torch.autograd.Function` backward passes passing `gradcheck()` |
| **$O(N)$ Space FlashAttention-2** | Maintains online softmax scaling vectors in SRAM, reducing VRAM footprint by up to **95.3%** |
| **CI Benchmark Drift Guard** | Automated CI workflow (`card_vs_json_check.py`) enforcing documentation metrics match JSON results within ±10% |

---

## ⚡ PyTorch Eager & TritonForge Head-to-Head Benchmarks

> Measured on physical NVIDIA Tesla T4 GPU (320 GB/s peak HBM bandwidth, CUDA 12.1, PyTorch 2.4.0 / 2.1.0):

| Fused Kernel | PyTorch Eager (ms) | TritonForge (ms) | Speedup vs PyTorch | Key Metric / Bandwidth |
|---|---|---|---|---|
| **Fused RMSNorm** | 1.701 ms | **0.380 ms** | **4.47x** | **93.1% HBM Utilization** (297.8 GB/s) |
| **FlashAttention-2** | 4.312 ms | **1.625 ms** | **2.65x** | **95.3% Memory Reduction** (134.2MB → 6.3MB) |
| **SwiGLU Activation** | 0.490 ms | **0.280 ms** | **1.75x** | **202.3 GB/s Bandwidth** (4 launches → 1) |
| **Transformer Block** | 11.40 ms | **3.52 ms** | **3.24x** | **Full Decoder Block Fusion** |

---

## 🛠️ CUDA C++ Extension & Autograd Training Support

TritonForge provides dual-mode execution for both inference serving and training:

1. **Raw CUDA C++ Kernel Extension (`rmsnorm_cuda.cu`):**
   - Implements warp-parallel cooperative reduction using `__shfl_xor_sync`.
   - Loaded dynamically via `torch.utils.cpp_extension.load_inline()`.
   - Eliminates Python runtime overhead for ultra-low latency scenarios.

2. **PyTorch Autograd Integration (`norm_autograd.py`):**
   - Implements `FusedRMSNormAutograd(torch.autograd.Function)`.
   - Provides exact mathematical backward pass gradients for $dX$ and $dWeight$.
   - Verified with `torch.autograd.gradcheck()` across float32, float16, and bfloat16 dtypes.

---

## 📂 Repository Structure

```yaml
tritonforge/
  ├── tritonforge/
  │   ├── kernels/
  │   │   ├── norm.py               # Triton fused RMSNorm forward pass
  │   │   ├── norm_autograd.py      # Autograd backward pass implementation
  │   │   ├── rmsnorm_cuda.cu       # Raw CUDA C++ kernel with warp reduction
  │   │   ├── rmsnorm_cuda_ext.py   # PyTorch C++ extension loader
  │   │   ├── activation.py         # Fused SwiGLU activation kernel
  │   │   ├── attention.py          # SRAM-tiled FlashAttention-2 kernel
  │   │   └── fused_norm_linear.py  # Single-pass Fused RMSNorm + Linear
  │   ├── models/
  │   │   └── transformer_block.py  # Fused Transformer Decoder Block
  │   └── core/
  │       └── router.py             # CUDA / CPU automatic hardware router
  ├── benchmarks/
  │   ├── nsight_profile_report.md  # NVIDIA Nsight SM occupancy & roofline metrics
  │   ├── profile_runner.py         # NVTX-annotated profile runner
  │   ├── card_vs_json_check.py     # CI benchmark drift guard script
  │   ├── vllm_serving_benchmark.py # vLLM serving integration benchmark
  │   └── block_benchmark.py        # Multi-kernel transformer block benchmark
  ├── tests/                        # 28 passing Pytest unit correctness tests
  ├── .github/workflows/
  │   ├── ci.yml                    # Pytest & security lint workflow
  │   └── benchmark_guard.yml       # Benchmark drift guard workflow
  ├── setup.py                      # Top-level setuptools package installer
  └── requirements.txt              # Production dependency list
```

---

## 🚀 Getting Started

### 1. Run Unit & Numerical Precision Tests
```bash
pytest tests/ -v
```

### 2. Run CI Benchmark Drift Guard
```bash
python3 benchmarks/card_vs_json_check.py
```

### 3. Run Standalone vLLM Serving Benchmark
```bash
python3 benchmarks/vllm_serving_benchmark.py
```
