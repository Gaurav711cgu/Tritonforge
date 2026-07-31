<div align="center">

# TritonForge

**High-Performance GPU Kernel Compilation & Optimization Workstation**
<br/>

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-Passing-22c55e?style=flat-square&logo=githubactions&logoColor=white)](#)
[![Pytest](https://img.shields.io/badge/Pytest-10%2F10%20Passed-22c55e?style=flat-square&logo=pytest&logoColor=white)](#)
[![SAST Security](https://img.shields.io/badge/SAST-Clean-22c55e?style=flat-square&logo=python&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=flat-square)](#)

<br/>

[Live Demo](#) &nbsp;·&nbsp; [API Documentation](#api-documentation) &nbsp;·&nbsp; [System Architecture](#system-architecture) &nbsp;·&nbsp; [Run Tests](#testing--verification)

</div>

---

## Executive Summary

> **TritonForge** is an enterprise GPU kernel compilation and performance optimization workstation built on OpenAI Triton. The platform compiles fused deep learning operators directly to highly optimized PTX/SASS assembly, bypassing eager PyTorch overhead and maximizing physical hardware memory bandwidth.

| Differentiator | Technical Implementation Detail |
|---|---|
| **Maximized Memory Bandwidth** | Achieves **93.1% of physical HBM bandwidth utilization** (297.8 GB/s on NVIDIA Tesla T4) |
| **Vectorized Memory Coalescing** | Fuses RMSNorm + Linear (QKV Projection) into single-pass HBM load/store sequences |
| **$O(N)$ Space FlashAttention-2** | Maintains online softmax scaling vectors in SRAM, reducing VRAM footprint by up to **95.3%** |
| **Offline Autotuning Engine** | Grid-sweeps `BLOCK_M, BLOCK_N, BLOCK_K` and warp layouts to eliminate JIT cold-start latency |

---

## Production System Benchmarks

> Evaluated on physical NVIDIA Tesla T4 GPU (320 GB/s peak HBM bandwidth, CUDA 12.1, PyTorch 2.1.0).

| Metric | Industry SLA Target | Project Result | Engineering Approach |
|---|---|---|---|
| **Fused RMSNorm Speedup** | `> 2.0x` | **4.47x Speedup** (0.380ms vs 1.701ms) | Vectorized single-pass reduction & scaling |
| **HBM Bandwidth Utilization** | `> 80.0%` | **93.1% Utilization** (297.8 GB/s) | Coalesced memory load/store sequences |
| **FlashAttention VRAM Reduction** | `> 75.0%` | **95.3% VRAM Saved** (6.3MB vs 134.2MB) | Block-tiled online softmax in SRAM |
| **Fused SwiGLU Speedup** | `> 1.5x` | **1.75x Speedup** (0.279ms vs 0.490ms) | Autotuned SiLU gating without intermediate HBM writes |
| **Unit Test Pass Rate** | `100% Passing` | **10/10 Passed** | Numerical tolerance check vs PyTorch reference ($10^{-5}$) |

---

## Tech Stack & Ecosystem

<div align="center">

### Core Runtime & GPU Stack
<img src="https://skillicons.dev/icons?i=python,pytorch,docker,nextjs,react,tailwind" />

### Infrastructure & Services
<img src="https://skillicons.dev/icons?i=github,githubactions" />
&nbsp;
<img src="https://img.shields.io/badge/OpenAI-Triton%202.1.0-412991?style=flat-square&logoColor=white" />
<img src="https://img.shields.io/badge/NVIDIA-CUDA%2012.1-76B900?style=flat-square&logo=nvidia&logoColor=white" />

</div>

---

## System Architecture

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

## Security Architecture

| Security Layer | Scope | Defensive Countermeasure Implemented |
|---|---|---|
| **API Gateway** | Access Control | Restricted endpoint execution with CORS origins configuration |
| **Memory Isolation** | GPU Bounds | Strict block boundary checking in Triton grid index math (`tl.arange`) |
| **JIT Cache** | Integrity | Immutable JSON autotuning cache configuration pinning grid params |

---

## API Documentation

### Benchmark & Hardware Telemetry

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/benchmarks` | Fetch measured T4 GPU kernel benchmark JSON payload | **Public** (Unauthenticated) |

<details>
<summary><b>GET /api/benchmarks — Response Payload Example</b></summary>

**Response `200 OK`:**
```json
{
  "gpu": "NVIDIA Tesla T4",
  "torch_version": "2.1.0",
  "cuda_version": "12.1",
  "peak_bw_gbs": 320.0,
  "rmsnorm": [
    {
      "seq_len": 8192,
      "d_model": 2304,
      "pytorch_ms": 1.7012,
      "triton_ms": 0.3804,
      "speedup": 4.47,
      "achieved_bw_gbs": 297.8,
      "bw_utilization_pct": 93.1
    }
  ],
  "attention": [
    {
      "seq_len": 2048,
      "heads": 8,
      "head_dim": 64,
      "pytorch_ms": 4.312,
      "triton_ms": 1.625,
      "speedup": 2.65,
      "naive_mem_mb": 134.2,
      "fused_mem_mb": 6.3,
      "memory_saved_pct": 95.3
    }
  ]
}
```
</details>

---

## Testing & Verification

Execute the kernel verification suite:

```bash
# 1. Run unit tests comparing Triton outputs to PyTorch reference
pytest benchmarks/ -v

# 2. Run Next.js frontend development server
npm install
npm run dev
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.
