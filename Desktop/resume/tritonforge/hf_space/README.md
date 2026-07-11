---
title: Triton Forge
emoji: ⚡
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
license: mit
hardware: t4-small
short_description: Fused Triton GPU kernels — RMSNorm, SwiGLU, FlashAttn
---

# TritonForge ⚡ — Live GPU Kernel Benchmark

Interactive demo for the TritonForge GPU kernel optimization library.
Run fused Triton JIT kernels and compare against PyTorch eager on a live T4 GPU.

**Kernels:** RMSNorm (8.2×) · SwiGLU (1.7×) · FlashAttention (3.3×, O(N) memory)

→ [GitHub Repository](https://github.com/Gaurav711cgu/TritonForge)
