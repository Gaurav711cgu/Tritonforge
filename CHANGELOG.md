# Changelog

All notable changes to the TritonForge project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-10

### Added
- Raw CUDA C++ fused RMSNorm kernel (`rmsnorm_cuda.cu`) with intra-warp `__shfl_xor_sync` register reductions.
- PyTorch dynamic C++ extension loader (`rmsnorm_cuda_ext.py`).
- Autograd analytical backward pass (`norm_autograd.py`).
- SRAM-tiled FlashAttention-2 and SwiGLU Triton JIT fused kernels.
- Automated CI benchmark drift guard check (`card_vs_json_check.py`).
- Machine-readable committed benchmark JSON artifact (`benchmarks/results.json`).
- Packaging metadata (`pyproject.toml`) and Docker container setup (`Dockerfile`).

### Benchmarks
- Fused RMSNorm Speedup: 4.47x (0.380 ms vs 1.701 ms eager PyTorch) [measured]
- HBM Bandwidth Utilization: 297.8 GB/s (93.1% of Tesla T4 peak) [measured]
- FlashAttention-2 VRAM Memory Savings: 95.3% reduction [measured]
- SM Occupancy: 88.5% [measured]
