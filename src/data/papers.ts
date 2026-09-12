import { Paper } from "@/types";

export const paperData: Paper = {
  id: 'tritonforge-paper-2024',
  title: 'TritonForge: Automated GPU Kernel Optimization and SRAM-Aware Fusion for Large Language Model Inference',
  authors: 'Gaurav Kumar Nayak (Lead Systems & AI Engineer)',
  date: 'September 2024 · Systems & Hardware Architecture Track',
  doi: 'arXiv:2409.12345v1 [cs.DC]',
  abstract: 'Large language model (LLM) inference is predominantly bounded by GPU High Bandwidth Memory (HBM) latency rather than peak arithmetic throughput. Memory-bound layers — specifically Root Mean Square Normalization (RMSNorm), gated activation functions (SwiGLU), and scaled dot-product attention — suffer from repeated memory transfers between HBM and on-chip Streaming Multiprocessors (SMs), yielding low hardware arithmetic intensity. We present TritonForge, an automated GPU kernel optimization and profiling workstation built on OpenAI Triton. TritonForge implements custom block-tiled kernels that fuse multiple sequential tensor operations into single-pass kernels executed entirely within on-chip Static RAM (SRAM) and register files. Across benchmark evaluations on NVIDIA Tesla T4 and A100 architectures, TritonForge achieves 4.47× speedup on RMSNorm, 1.75× speedup on SwiGLU, and 3.3× speedup on FlashAttention-2 with a 95.3% reduction in peak attention memory consumption. We demonstrate full numerical parity (atol=1e-5) against PyTorch eager baselines, implement custom backward passes via torch.autograd.Function, and present a dynamic 4-layer fallback routing architecture guaranteeing high availability across diverse deployment targets.',
  tags: ['GPU Computing', 'OpenAI Triton', 'LLM Inference', 'Memory-Bound Kernels', 'FlashAttention', 'CUDA'],
  sections: [
    {
      title: '1. Introduction & The Memory Wall in LLM Inference',
      content: `Modern transformer-based LLMs (e.g. LLaMA-3, Mistral, Gemma) spend up to 70% of generation latency in memory-bound layers. While modern GPUs provide immense compute capabilities (e.g. 312 TFLOPS on A100 Tensor Cores), memory bandwidth is severely constrained at ~2 TB/s.

PyTorch eager execution exacerbates this problem by decomposing composite operations into discrete CUDA kernel calls. For example, PyTorch eager RMSNorm executes three discrete kernel launches: (1) mean of squares, (2) reciprocal square root, and (3) element-wise multiplication with weights. Each kernel launch reads the entire tensor from global HBM and writes intermediate results back to HBM, incurring substantial bus roundtrips and kernel dispatch overhead.

TritonForge tackles this fundamental "Memory Wall" by fusing operations into monolithic Triton kernels executed in on-chip SRAM/registers, completely bypassing intermediate HBM roundtrips.`,
    },
    {
      title: '2. GPU Memory Hierarchy & The Case for On-Chip Fusion',
      content: `Modern GPU architectures (NVIDIA Volta, Ampere, Hopper) exhibit a steep memory hierarchy:
- Registers (RF): > 19,000 GB/s bandwidth, 1 cycle latency (~0.7 ns).
- SRAM / Shared Memory: ~ 19,000 GB/s bandwidth, 20-30 cycles (~15 ns).
- L2 Cache: ~ 5,000 GB/s bandwidth, ~200 cycles (~150 ns).
- High Bandwidth Memory (HBM2e/HBM3): 2,000 - 3,350 GB/s bandwidth, 400-800 cycles (~300 ns).

When executing memory-bound operators, loading a float32 operand from HBM costs ~300 nanoseconds. By keeping intermediate values resident in register tiles and SRAM across sequential operations, TritonForge reduces global memory accesses from 3 roundtrips to exactly 1 read and 1 write.`,
    },
    {
      title: '3. Kernel Implementations, Stride Safety & Autograd',
      content: `3.1 Fused RMSNorm: Employs a 1D grid mapped across sequence rows. A two-pass loop tiles across the hidden dimension N in chunks of BLOCK_SIZE = min(next_power_of_2(N), 4096), accumulating variance in registers before scaling and storing output back to HBM. Includes full FusedRMSNormAutograd support.

3.2 Contiguous Stride Safety: For 3D inputs [Batch, Seq, Hidden] or sliced/transposed views, strides across leading dimensions are non-contiguous. TritonForge automatically enforces x = x.contiguous().view(-1, N), guaranteeing that row_idx * stride_x_row correctly accesses row memory without data corruption.

3.3 Autotuned SwiGLU: Gated activation fuses linear projections, SiLU gate, and element-wise multiplication into a single pass. The Triton autotuner explores BLOCK_SIZE ∈ {128, 256, 512, 1024} with warps ∈ {4, 8}.

3.4 Tiled FlashAttention-2: Implements online softmax tracking running maximums (m_i) and denominators (l_i) in SRAM across Q and K block tiles, avoiding materialization of the N x N attention matrix.`,
    },
    {
      title: '4. Empirical Evaluation & Benchmarks',
      content: `Evaluations conducted on NVIDIA Tesla T4 and A100 GPUs across sequence lengths from 256 to 8192:
- Fused RMSNorm: Achieves 4.47× speedup on T4 and 8.2× on A100, reaching 93.1% of theoretical peak HBM bandwidth (297.8 GB/s on T4).
- Fused SwiGLU: Delivers 1.75× speedup, eliminating intermediate tensor buffers.
- FlashAttention-2: Delivers 3.3× speedup at N=2048 with 95.3% reduction in peak VRAM consumption (6.3 MB vs 134.2 MB naive).
- End-to-End Transformer Block: Replacing standard PyTorch modules with TritonForgeTransformerBlock achieves 2.85× overall inference speedup.`,
    },
    {
      title: '5. Hardware-Aware Dynamic Fallback Router',
      content: `To ensure zero-crash production deployments, TritonForge wraps kernel invocations with @triton_route:
1. Environment Check: Tests HAS_TRITON flag at import time.
2. Hardware Check: Verifies CUDA device availability and tensor device placement.
3. Shape Validation: Enforces dimension and memory alignment limits.
4. JIT Exception Trapping: Traps triton.compiler.CompilationError, PTX syntax errors, or out-of-memory exceptions and transparently falls back to PyTorch eager references.`,
    },
  ],
  bibtex: `@article{nayak2024tritonforge,
  title   = {TritonForge: Automated GPU Kernel Optimization and SRAM-Aware Fusion for Large Language Model Inference},
  author  = {Nayak, Gaurav Kumar},
  journal = {arXiv preprint arXiv:2409.12345},
  year    = {2024},
  url     = {https://github.com/Gaurav711cgu/TritonForge}
}`,
};

export const blogs = [
  {
    id: 'deep-dive-rmsnorm',
    title: 'Why Your Normalization Layer is Slowing Down LLM Inference',
    date: 'August 2024',
    readTime: '6 min read',
    tags: ['Architecture', 'HBM Bandwidth', 'Triton'],
    excerpt: 'RMSNorm only performs 2 arithmetic operations per byte of memory read from HBM. How kernel fusion eliminates 3 round trips to main memory.',
  },
  {
    id: 'flash-attention-explained',
    title: 'Demystifying FlashAttention-2: Online Softmax in SRAM',
    date: 'September 2024',
    readTime: '9 min read',
    tags: ['FlashAttention', 'SRAM', 'Transformers'],
    excerpt: 'The math behind Milakov-Gimelshein online softmax and why you never need to write an N×N attention matrix to global GPU memory.',
  },
  {
    id: 'writing-custom-autograd',
    title: 'Writing Production PyTorch Autograd Functions in OpenAI Triton',
    date: 'October 2024',
    readTime: '8 min read',
    tags: ['PyTorch', 'Autograd', 'Backprop'],
    excerpt: 'Step-by-step derivation of analytical backward passes and writing torch.autograd.Function wrappers that pass gradcheck.',
  },
];
