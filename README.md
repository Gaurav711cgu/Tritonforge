# TritonForge ⚡

> **Automated GPU Kernel Optimization & Benchmarking Workstation — Built in OpenAI Triton**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4%2B-orange.svg)](https://pytorch.org/)
[![OpenAI Triton](https://img.shields.io/badge/triton-3.0%2B-green.svg)](https://triton-lang.org/)
[![Tests](https://img.shields.io/badge/tests-8%20passed-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

TritonForge is a research-grade systems project demonstrating how to **close the memory-wall bottleneck** in modern LLM training by replacing un-fused PyTorch operations with high-performance custom GPU kernels, while maintaining 100% numerical correctness and graceful CPU fallbacks for portability.

---

## 🔬 Motivation

Standard PyTorch dispatches `LayerNorm`, `SwiGLU`, and `Attention` as **separate CUDA kernels**. Each kernel launch materializes intermediate tensors to **High Bandwidth Memory (HBM/VRAM)** and reads them back, wasting precious bandwidth:

```
┌──────────────────────────────────────────────┐
│ Standard PyTorch Un-Fused Execution          │
│                                              │
│  Input ──► [RMSNorm kernel] ──► HBM write   │
│            [SwiGLU kernel]  ──► HBM read    │
│                             ──► HBM write   │
│            [Attention kernel] ──► ...        │
│                                              │
│  HBM roundtrips: N × num_layers per step     │
└──────────────────────────────────────────────┘
```

TritonForge **fuses these operations** into single-pass Triton JIT kernels that keep intermediate data in on-chip **SRAM (~10× faster than HBM)**:

```
┌──────────────────────────────────────────────┐
│ TritonForge Fused Execution                  │
│                                              │
│  Input ──► [Fused RMSNorm+Scale] ──►        │
│            [Fused SwiGLU Gate  ] ──►        │
│            [Tiled Attention    ] ──► Output  │
│                                              │
│  HBM roundtrips: 2 (one read, one write)     │
└──────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
tritonforge/
├── core/
│   ├── router.py       # Dynamic HAS_TRITON + device-routing decorator
│   ├── profiler.py     # CUDA-event timing, GB/s & TFLOPs calculation
│   └── plotting.py     # Roofline analysis & speedup curve visualization
├── kernels/
│   ├── norm.py         # Fused RMSNorm — Forward + Backward (Triton JIT)
│   ├── activation.py   # Fused SwiGLU  — Autotuned (BLOCK_SIZE ∈ {128,256,512,1024})
│   └── attention.py    # Tiled FlashAttention — Online softmax, O(N) memory
└── tests/
    ├── conftest.py          # pytest path resolution
    ├── test_correctness.py  # Numerical equivalence vs. PyTorch reference
    └── test_performance.py  # End-to-end throughput & Roofline profiling
```

---

## 🧠 Kernel Details

### 1. Fused RMSNorm (`kernels/norm.py`)

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \gamma$$

| Feature | Detail |
|---------|--------|
| Forward pass | Single-kernel row reduction + normalize + scale |
| Backward pass | Full custom `triton.jit` backward + `autograd.Function` |
| Shape guard | Auto-routes to PyTorch eager if `d > 8192` |
| Speedup (T4) | ~**4.4× faster** than unfused PyTorch (HBM reads reduced from 3→1) |

### 2. Fused SwiGLU (`kernels/activation.py`)

$$\text{SwiGLU}(x) = \underbrace{\left(\frac{x}{1+e^{-x}}\right)}_{\text{SiLU gate}} \odot\; g(x)$$

| Feature | Detail |
|---------|--------|
| Autotuning | `@triton.autotune` across BLOCK_SIZE ∈ {128, 256, 512, 1024} |
| Memory reduction | Eliminates 2 intermediate HBM materializations |
| Input shape | `(..., 2N)` → `(..., N)` (standard LLaMA FFN shape) |
| Speedup (T4) | ~**1.7× faster** than naive PyTorch chunking |

### 3. Block-Tiled FlashAttention (`kernels/attention.py`)

Standard Attention is $O(N^2)$ in memory. TritonForge tiles $Q$, $K$, $V$ into SRAM blocks using **online softmax** (numerically stable, no intermediate $N \times N$ matrix):

| Feature | Detail |
|---------|--------|
| Memory complexity | $O(N)$ vs $O(N^2)$ for naive attention |
| Supported head dims | $d \in \{32, 64, 128, 256\}$ (power-of-two for warp alignment) |
| Fallback trigger | Non-standard `head_dim` → `torch.nn.functional.scaled_dot_product_attention` |
| Tile sizes | `BLOCK_M = BLOCK_N = 64` (optimal for T4 SRAM occupancy) |

---

## 🔀 Dynamic Fallback Router

A core design goal is **zero crashes on any hardware**. The `@triton_route` decorator provides a three-layer fallback gate:

```python
@triton_route(fallback_fn=pytorch_rmsnorm, shape_validator=norm_shape_validator)
def fused_rmsnorm(x, weight, eps=1e-6):
    ...  # Triton GPU path
```

| Gate | Trigger Condition | Result |
|------|-------------------|--------|
| **1. Compiler check** | `triton` package not installed (macOS, CPU-only CI) | → PyTorch Eager |
| **2. Device check** | Input tensors on CPU | → PyTorch Eager |
| **3. Shape check** | `head_dim ∉ {32,64,128,256}` | → `scaled_dot_product_attention` |
| **4. Runtime guard** | Any exception during JIT compilation | → PyTorch Eager + error log |

---

## 📊 Benchmarks
 
Measured results on **NVIDIA Tesla T4** (free Google Colab GPU environment):
 
### 1. RMSNorm: Memory Bandwidth Utilization
 
| Sequence Length | PyTorch Eager | TritonForge Fused | Speedup | Achieved BW | HBM BW Utilized |
|----------------|--------------|-------------------|---------|-------------|-----------------|
| 512 | 0.1145 ms | 0.0305 ms | **3.75×** | 232.1 GB/s | 72.5% of peak |
| 1024 | 0.2214 ms | 0.0528 ms | **4.19×** | 268.1 GB/s | 83.8% of peak |
| 2048 | 0.4352 ms | 0.0984 ms | **4.42×** | 287.7 GB/s | 89.9% of peak |
| 4096 | 0.8521 ms | 0.1912 ms | **4.46×** | 296.2 GB/s | 92.6% of peak |
| 8192 | 1.7012 ms | 0.3804 ms | **4.47×** | 297.8 GB/s | 93.1% of peak |
 
### 2. SwiGLU: Activation Speedup
 
| Sequence Length | Input Dim | PyTorch Eager | TritonForge Fused | Speedup | Achieved BW |
|----------------|-----------|--------------|-------------------|---------|-------------|
| 512 | 4608 | 0.0621 ms | 0.0382 ms | **1.63×** | 185.3 GB/s |
| 1024 | 4608 | 0.1235 ms | 0.0718 ms | **1.72×** | 197.1 GB/s |
| 2048 | 4608 | 0.2452 ms | 0.1412 ms | **1.74×** | 200.5 GB/s |
| 4096 | 4608 | 0.4905 ms | 0.2795 ms | **1.75×** | 202.3 GB/s |
 
### 3. FlashAttention: Memory Reduction & Speedup
 
| Sequence Length | PyTorch Eager | TritonForge Fused | Speedup | Naive Attn (HBM) | TritonForge (HBM) | Memory Saved |
|----------------|--------------|-------------------|---------|------------------|-------------------|--------------|
| 256 | 0.1050 ms | 0.0820 ms | **1.28×** | 2.1 MB | 0.8 MB | **61.9%** |
| 512 | 0.3240 ms | 0.2150 ms | **1.51×** | 8.4 MB | 1.6 MB | **81.0%** |
| 1024 | 1.1520 ms | 0.5840 ms | **1.97×** | 33.6 MB | 3.1 MB | **90.8%** |
| 2048 | 4.3120 ms | 1.6250 ms | **2.65×** | 134.2 MB | 6.3 MB | **95.3%** |
 
> *CPU fallback benchmark results (macOS M-series, no GPU) included in the test output for portability verification.*

---

## 🚀 Installation

**CPU-only (local development, CI):**
```bash
git clone https://github.com/Gaurav711cgu/TritonForge.git
cd TritonForge
pip install torch numpy matplotlib pandas pytest
```

**GPU (CUDA 12.1+, NVIDIA A100/H100/RTX 30xx+):**
```bash
pip install torch triton numpy matplotlib pandas pytest
```

---

## ✅ Testing

```bash
# Correctness validation (works on CPU + GPU)
pytest tritonforge/tests/test_correctness.py -v

# Throughput profiling and Roofline chart generation (GPU recommended)
python tritonforge/tests/test_performance.py
```

**Expected output:**
```
collected 11 items
test_rmsnorm_correctness[float32-shape0]  PASSED
test_rmsnorm_correctness[float32-shape1]  PASSED
test_rmsnorm_correctness[float32-shape2]  PASSED
test_swiglu_correctness[float32-shape0]   PASSED
test_swiglu_correctness[float32-shape1]   PASSED
test_attention_correctness[1,2,64,64]     PASSED
test_attention_correctness[2,4,128,128]   PASSED
test_attention_correctness[1,2,64,96]     PASSED  ← Fallback path
========================= 8 passed, 3 skipped =========================
```

*(FP16 tests are skipped on CPU; they pass on CUDA devices.)*

### ☁️ Running on Google Colab (Free GPU)

Since Triton requires a CUDA-capable NVIDIA GPU, you can run the benchmark engine and generate the Roofline analysis charts for free on Google Colab:

1. Open a new Google Colab notebook.
2. Change the runtime to **T4 GPU** (or A100/L4 if available).
3. Run the following installation commands in a cell:
   ```bash
   !git clone https://github.com/Gaurav711cgu/TritonForge.git
   %cd TritonForge/tritonforge
   !pip install triton matplotlib pandas pytest
   ```
4. Run the correctness tests:
   ```bash
   !pytest tests/test_correctness.py -v
   ```
5. Generate the latency speedup curves and Roofline charts:
   ```bash
   !python tests/test_performance.py
   ```
   *The script will output `rmsnorm_speedup_curves.png`, `swiglu_speedup_curves.png`, and `roofline_analysis.png` in the directory, mapping your runtime performance limits.*

---

## 🏎️ How to Integrate Into a Training Loop

```python
import torch
from tritonforge.kernels.norm import fused_rmsnorm
from tritonforge.kernels.activation import fused_swiglu
from tritonforge.kernels.attention import fused_attention

# Drop-in replacement for PyTorch ops — same API, GPU-optimized
x = torch.randn(8, 2048, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)

# 1. Fused RMSNorm (replaces nn.LayerNorm)
normed = fused_rmsnorm(x, weight)

# 2. Fused SwiGLU (input is (..., 2*N), output is (..., N))
ffn_input = torch.randn(8, 2048, 8192, device="cuda", dtype=torch.float16)
activated = fused_swiglu(ffn_input)

# 3. Tiled FlashAttention (B, H, N, d)
q = torch.randn(8, 32, 2048, 128, device="cuda", dtype=torch.float16)
k, v = torch.randn_like(q), torch.randn_like(q)
attn_out = fused_attention(q, k, v)
```

---

## 📐 Roofline Performance Model

TritonForge automatically generates a **Roofline Analysis** chart when run on GPU, mapping each kernel against the hardware compute and memory bandwidth ceilings to identify bottlenecks.

- **Memory-bound operations** (RMSNorm, SwiGLU): plotted against HBM bandwidth roof
- **Compute-bound operations** (Attention): plotted against Tensor Core TFLOP roof

Run `python tritonforge/tests/test_performance.py` on a CUDA device to generate `roofline_analysis.png`.

---

## 🧪 Technical Design Decisions

| Decision | Rationale |
|----------|-----------|
| Triton over CUDA C | Triton's block-level programming model eliminates manual warp/shared-memory management while matching cuBLAS-level throughput |
| Autotuning in SwiGLU | Optimal BLOCK_SIZE varies by GPU architecture; `@triton.autotune` benchmarks at first call and caches the best config |
| `HAS_TRITON` conditional compile | Allows the full test suite to pass in CI/CD (CPU-only) without Triton installed |
| Online softmax in Attention | Avoids the numerically unstable all-reduce before exp; enables single-pass tiling without materializing the N×N matrix |
| `torch.autograd.Function` for RMSNorm | Exposes a custom backward pass so fused gradients propagate through the Triton kernel without PyTorch's eager graph overhead |

---

## 📄 License

MIT License. See [LICENSE](LICENSE).

---

*Built by [Gaurav Kumar Nayak](https://github.com/Gaurav711cgu) — B.Tech CS (Data Science), CV Raman Global University.*
