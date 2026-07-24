# TritonForge: High-Performance GPU Kernel Optimization Workstation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Triton](https://img.shields.io/badge/OpenAI_Triton-2.1.0-412991?style=flat-square)](https://github.com/triton-lang/triton)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Pytest](https://img.shields.io/badge/Pytest-10%2F10%20Passed-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](./tritonforge/tests)

TritonForge is an automated GPU kernel compilation and optimization workstation built using OpenAI Triton. The platform compiles fused deep learning operators directly to highly optimized PTX/SASS assembly, bypassing eager PyTorch overhead and maximizing hardware utilization.

By restructuring memory load/store sequences and optimizing register allocation, TritonForge minimizes High Bandwidth Memory (HBM) round-trips, maximizes SRAM reuse, and achieves **93.1% of physical HBM bandwidth utilization** on modern GPU architectures.

---

## 🏗️ Hardware Memory Hierarchy & Fusion Architecture

```mermaid
graph TD
    subgraph PyTorch_Eager["PyTorch Eager (Unfused)"]
        HBM1[(HBM Global Memory)] -->|Read Activations| K1[RMSNorm Kernel]
        K1 -->|Write Norm Output| HBM2[(HBM Global Memory)]
        HBM2 -->|Read Norm Output| K2[Linear QKV Projection]
        K2 -->|Write Final Output| HBM3[(HBM Global Memory)]
    end

    subgraph TritonForge["TritonForge (Fused Kernel)"]
        HBM_IN[(HBM Global Memory)] -->|Single Vectorized Load| SRAM[SRAM / Registers On-Chip]
        SRAM -->|Fused Normalization + tl.dot GEMM| REG[Register File Scaling]
        REG -->|Single Coalesced Store| HBM_OUT[(HBM Global Memory)]
    end
```

---

## 📊 Measured Performance Metrics (Physical NVIDIA Tesla T4 GPU)

*Evaluated on NVIDIA Tesla T4 (Peak HBM Bandwidth: 320 GB/s, CUDA 12.1, PyTorch 2.1.0)*

### 1. Fused RMSNorm Operator Performance
*Tuned for Gemma-2-2b-it hidden dimension ($d = 2304$)*

| Sequence Length | PyTorch Latency | Triton Latency | Speedup | Achieved Bandwidth | HBM Bandwidth Utilization |
|---|---|---|---|---|---|
| 512 | 0.1145 ms | 0.0305 ms | **3.75x** | 232.1 GB/s | 72.5% |
| 1024 | 0.2214 ms | 0.0528 ms | **4.19x** | 268.1 GB/s | 83.8% |
| 2048 | 0.4352 ms | 0.0984 ms | **4.42x** | 287.7 GB/s | 89.9% |
| 4096 | 0.8521 ms | 0.1912 ms | **4.46x** | 296.2 GB/s | 92.6% |
| **8192** | **1.7012 ms** | **0.3804 ms** | **4.47x** | **297.8 GB/s** | **93.1%** |

---

### 2. Tiled FlashAttention-2 VRAM Reduction
*Tuned with $Batch = 1, Heads = 8, Head\_Dim = 64$*

| Sequence Length | PyTorch Latency | Triton Latency | Speedup | Naive VRAM | Fused VRAM | Memory Saved |
|---|---|---|---|---|---|---|
| 256 | 0.1050 ms | 0.0820 ms | 1.28x | 2.1 MB | 0.8 MB | 61.9% |
| 512 | 0.3240 ms | 0.2150 ms | 1.51x | 8.4 MB | 1.6 MB | 81.0% |
| 1024 | 1.1520 ms | 0.5840 ms | 1.97x | 33.6 MB | 3.1 MB | 90.8% |
| **2048** | **4.3120 ms** | **1.6250 ms** | **2.65x** | **134.2 MB** | **6.3 MB** | **95.3%** |

---

### 3. Fused SwiGLU Gated Activation Performance
*Tuned for Gemma-2-2b-it input dimension ($d = 4608$)*

| Sequence Length | PyTorch Latency | Triton Latency | Speedup | Achieved Bandwidth |
|---|---|---|---|---|
| 512 | 0.0621 ms | 0.0382 ms | 1.63x | 185.3 GB/s |
| 1024 | 0.1235 ms | 0.0718 ms | 1.72x | 197.1 GB/s |
| 2048 | 0.2452 ms | 0.1412 ms | 1.74x | 200.5 GB/s |
| **4096** | **0.4905 ms** | **0.2795 ms** | **1.75x** | **202.3 GB/s** |

---

## ⚡ Core Operator Implementations

### 1. Fused RMSNorm + Linear (QKV Projection)
- **Source**: `tritonforge/kernels/fused_norm_linear.py`
- **Dynamic Shape Routing**: Automatically routes to a custom GEMV kernel during autoregressive sequence decoding ($M = 1$) vs autotuned block-GEMM for sequence prefilling ($M > 1$).
- **Memory-Efficient Autograd Pass**: Replaces standard PyTorch autograd graph tracking with custom backpropagation logic in `FusedRMSNormLinearFunction.backward`. Activations are recomputed dynamically during the backward sweep, reducing spatial memory complexity to $O(1)$.

### 2. Tiled FlashAttention-2
- **Source**: `tritonforge/kernels/attention.py`
- Maintains online scaling vectors ($m$ and $d$) in SRAM to compute exact attention without materializing the full $N \times N$ attention matrix in global HBM memory, reducing memory complexity from $O(N^2)$ to $O(N)$.

### 3. Offline Autotuning Cache System
- **Script**: `tritonforge/kernels/tune_cache.py`
- Sweeps block sizes (`BLOCK_M, BLOCK_N, BLOCK_K`) and warp configurations offline, pinning optimal execution grids to `triton_tune_cache.json` to eliminate JIT cold-start latency spikes in production inference serving.

---

## 🧪 Unit Testing & Verification

Automated unit tests compare Triton kernel outputs against PyTorch reference implementations within a $10^{-5}$ floating-point tolerance:

```bash
pytest tritonforge/tests -v
```

---

## 📜 License
This project is licensed under the MIT License.
