'use client';

import React, { useState, useEffect, useRef } from 'react';
import { kernels } from '@/data/kernels';
import { defaultBenchmarks } from '@/data/benchmarks';
import { KernelSelectorCard, KernelPlayground } from '@/components/KernelCard';
import { BenchmarkComparison } from '@/components/BenchmarkComparison';
import { MemoryHierarchySvg } from '@/components/MemoryHierarchySvg';
import { PaperViewer } from '@/components/PaperViewer';
import { TelemetryModal } from '@/components/TelemetryModal';

export default function TritonForgeDashboard() {
  const [activeKernelId, setActiveKernelId] = useState<string>('norm');
  const [activeCompileStep, setActiveCompileStep] = useState<number>(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState<boolean>(true);
  const [isTelemetryOpen, setIsTelemetryOpen] = useState<boolean>(false);
  const [statsVisible, setStatsVisible] = useState<boolean>(false);
  const [activeManualTab, setActiveManualTab] = useState<'install' | 'api' | 'tune' | 'safety'>('install');
  const [githubStars, setGithubStars] = useState<number | string>('300+');

  const statsRef = useRef<HTMLDivElement>(null);
  const selectedKernel = kernels.find((k) => k.id === activeKernelId) || kernels[0];

  // Compiler animation
  useEffect(() => {
    if (!isAutoPlaying) return;
    const interval = setInterval(() => {
      setActiveCompileStep((prev) => (prev + 1) % 3);
    }, 2800);
    return () => clearInterval(interval);
  }, [isAutoPlaying]);

  // GitHub stars fetch
  useEffect(() => {
    fetch('https://api.github.com/repos/Gaurav711cgu/TritonForge')
      .then((res) => res.json())
      .then((data) => {
        if (data.stargazers_count !== undefined) {
          setGithubStars(data.stargazers_count);
        }
      })
      .catch(() => {});
  }, []);

  // Intersection observer for stats visibility
  useEffect(() => {
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setStatsVisible(true);
      },
      { threshold: 0.3 }
    );
    if (statsRef.current) obs.observe(statsRef.current);
    return () => obs.disconnect();
  }, []);

  return (
    <main className="tf-root">
      {/* ─── NAVIGATION BAR ─── */}
      <nav className="tf-nav">
        <div className="nav-inner">
          <div className="nav-logo">
            <img src="/logo-highres.png" alt="TritonForge Logo" className="brand-logo-img" />
            <span>TRITONFORGE</span>
          </div>
          <div className="nav-links">
            {[
              { label: 'MOTIVATION', href: '#motivation' },
              { label: 'KERNELS', href: '#kernels' },
              { label: 'BENCHMARKS', href: '#benchmarks' },
              { label: 'RESEARCH', href: '#research' },
              { label: 'ARCHITECTURE', href: '#architecture' },
              { label: 'INSTALL', href: '#install' },
            ].map(({ label, href }) => (
              <a key={label} href={href} className="nav-link">
                {label}
              </a>
            ))}
            <button
              onClick={() => setIsTelemetryOpen(true)}
              className="text-xs font-mono px-3 py-1.5 bg-[#E53935]/15 hover:bg-[#E53935]/30 text-[#E53935] border border-[#E53935]/40 rounded-lg transition-colors flex items-center gap-1.5"
            >
              <span className="w-2 h-2 rounded-full bg-[#4CAF50] animate-pulse"></span>
              GPU TELEMETRY
            </button>
            <a
              href="https://github.com/Gaurav711cgu/TritonForge"
              target="_blank"
              rel="noreferrer"
              className="nav-cta"
            >
              GITHUB ★ {githubStars}
            </a>
          </div>
        </div>
      </nav>

      {/* ─── HERO SECTION ─── */}
      <section className="tf-hero">
        <div className="hero-content">
          <div className="hero-badge">
            <span className="badge-dot"></span>
            OPENAI TRITON GPU WORKSTATION &bull; FAANG L5 READY
          </div>
          <h1 className="hero-title">
            HIGH-PERFORMANCE <br />
            <span className="hero-gradient-text">GPU KERNEL FUSION</span> <br />
            FOR LLM INFERENCE
          </h1>
          <p className="hero-desc">
            SRAM-aware, block-tiled OpenAI Triton kernels eliminating High Bandwidth Memory (HBM)
            roundtrips. Built with real PyTorch autograd integration, stride-safe pointer math, and
            dynamic zero-crash fallback routing.
          </p>

          <div className="hero-actions">
            <a href="#benchmarks" className="btn btn-primary">
              EXPLORE BENCHMARKS &rarr;
            </a>
            <a href="#research" className="btn btn-secondary">
              READ ARXIV PAPER
            </a>
          </div>

          {/* Quick Metrics Bar */}
          <div ref={statsRef} className="hero-stats-grid">
            <div className="hero-stat-card">
              <span className="stat-value text-[#4CAF50]">8.2×</span>
              <span className="stat-label">RMSNorm Speedup</span>
            </div>
            <div className="hero-stat-card">
              <span className="stat-value text-white">95.3%</span>
              <span className="stat-label">VRAM Reduction</span>
            </div>
            <div className="hero-stat-card">
              <span className="stat-value text-[#FFA726]">93.1%</span>
              <span className="stat-label">Peak HBM BW Saturation</span>
            </div>
            <div className="hero-stat-card">
              <span className="stat-value text-[#E53935]">atol&le;1e-5</span>
              <span className="stat-label">Numerical Parity</span>
            </div>
          </div>
        </div>
      </section>

      {/* ─── MOTIVATION: MEMORY HIERARCHY SVG ─── */}
      <section className="tf-section" id="motivation">
        <div className="section-inner">
          <MemoryHierarchySvg />
        </div>
      </section>

      {/* ─── COMPILER PIPELINE ANIMATION ─── */}
      <section id="compiler-pipeline" className="tf-section tf-dark">
        <div className="section-inner">
          <div className="mb-6">
            <span className="text-xs font-mono uppercase tracking-wider text-[#E53935] font-semibold">
              Compiler Architecture
            </span>
            <h2 className="text-xl font-mono font-bold text-white mt-1">
              THE TRITON COMPILATION PIPELINE
            </h2>
            <p className="text-xs text-[#A09D96] mt-1 max-w-xl">
              From Python AST to optimized PTX assembly: How TritonForge compiles block-level abstractions
              into high-occupancy GPU machine code.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {[
              {
                step: 0,
                name: '1. Python AST & Block IR',
                desc: 'Translates high-level Python tensor slices into Triton Block-level Intermediate Representation (TTIR).',
                sample: 'tl.load(X + row*stride + cols, mask=mask)',
              },
              {
                step: 1,
                name: '2. LLVM IR & Memory Coalescing',
                desc: 'Automatic coalescing of global memory accesses into 128-byte transactions and bank-conflict-free shared memory allocation.',
                sample: '%triton_gpu.alloc() : memref<64x64xf32, #shared>',
              },
              {
                step: 2,
                name: '3. PTX Assembly & Register Allocation',
                desc: 'Low-level NVIDIA PTX code scheduling, warp shuffle execution, and register allocation constrained to &le;255 regs/thread.',
                sample: 'ld.global.nc.v4.u32 {%r0, %r1, %r2, %r3}, [%rd1];',
              },
            ].map((s) => (
              <div
                key={s.step}
                onClick={() => {
                  setActiveCompileStep(s.step);
                  setIsAutoPlaying(false);
                }}
                className={`p-4 rounded-xl border transition-all cursor-pointer ${
                  activeCompileStep === s.step
                    ? 'bg-[#1a1916] border-[#E53935] shadow-lg shadow-[#E53935]/10'
                    : 'bg-[#13120f] border-[#2a2825] hover:border-[#3a3835]'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-mono text-sm font-bold text-white">{s.name}</h3>
                  {activeCompileStep === s.step && (
                    <span className="w-2 h-2 rounded-full bg-[#E53935]"></span>
                  )}
                </div>
                <p className="text-xs text-[#A09D96] mb-3 leading-relaxed">{s.desc}</p>
                <pre className="p-2.5 bg-[#0a0a08] border border-[#2a2825] rounded text-[11px] font-mono text-[#4CAF50] overflow-x-auto">
                  <code>{s.sample}</code>
                </pre>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── KERNELS INTERACTIVE PLAYGROUND ─── */}
      <section className="tf-section" id="kernels">
        <div className="section-inner">
          <div className="mb-6">
            <span className="text-xs font-mono uppercase tracking-wider text-[#E53935] font-semibold">
              Kernel Architecture & Implementation
            </span>
            <h2 className="text-xl font-mono font-bold text-white mt-1">
              FUSED GPU OPERATORS (TRITON JIT)
            </h2>
            <p className="text-xs text-[#A09D96] mt-1 max-w-xl">
              Inspect Triton JIT kernels alongside naive PyTorch eager baselines. All kernels feature
              hardware-aware fallback routing and contiguity checking.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {kernels.map((k) => (
              <KernelSelectorCard
                key={k.id}
                kernel={k}
                active={activeKernelId === k.id}
                onClick={() => setActiveKernelId(k.id)}
              />
            ))}
          </div>

          <KernelPlayground kernel={selectedKernel} />
        </div>
      </section>

      {/* ─── BENCHMARKS SECTION ─── */}
      <section className="tf-section tf-dark" id="benchmarks">
        <div className="section-inner">
          <BenchmarkComparison initialBenchmarks={defaultBenchmarks} />
        </div>
      </section>

      {/* ─── RESEARCH & PAPER VIEWER ─── */}
      <section className="tf-section" id="research">
        <div className="section-inner">
          <PaperViewer />
        </div>
      </section>

      {/* ─── ARCHITECTURE & HARDWARE FALLBACK ROUTER ─── */}
      <section className="tf-section tf-dark" id="architecture">
        <div className="section-inner">
          <div className="mb-6">
            <span className="text-xs font-mono uppercase tracking-wider text-[#E53935] font-semibold">
              Fault Tolerance & Production Safety
            </span>
            <h2 className="text-xl font-mono font-bold text-white mt-1">
              4-LAYER HARDWARE-AWARE DYNAMIC FALLBACK ROUTER
            </h2>
            <p className="text-xs text-[#A09D96] mt-1 max-w-2xl">
              In production environments (e.g. CI/CD runners, Mac/CPU instances, cloud edge), GPU hardware
              or CUDA drivers may be unavailable. TritonForge provides seamless `@triton_route` wrapping:
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            {[
              {
                layer: 'Layer 1: Import Check',
                rule: 'HAS_TRITON availability test',
                desc: 'Safely traps environments without Triton installed and routes to PyTorch reference.',
              },
              {
                layer: 'Layer 2: CUDA Device',
                rule: 'torch.cuda.is_available() & device placement',
                desc: 'Confirms GPU runtime presence and ensures all input tensors reside in CUDA VRAM.',
              },
              {
                layer: 'Layer 3: Boundary Guard',
                rule: 'shape_validator(args, kwargs)',
                desc: 'Validates matrix dimensions and alignment; routes non-compliant shapes safely.',
              },
              {
                layer: 'Layer 4: JIT Exception Trap',
                rule: 'CompilationError & Hardware Trap',
                desc: 'Catches PTX syntax faults or driver errors and executes fallback transparently.',
              },
            ].map((l, i) => (
              <div key={i} className="p-4 rounded-xl bg-[#13120f] border border-[#2a2825]">
                <span className="text-[10px] font-mono text-[#E53935] uppercase font-bold block mb-1">
                  {l.layer}
                </span>
                <h4 className="font-mono text-xs font-bold text-white mb-2">{l.rule}</h4>
                <p className="text-xs text-[#A09D96] leading-relaxed">{l.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── INSTALLATION & DEVELOPER MANUAL ─── */}
      <section className="tf-section" id="install">
        <div className="section-inner">
          <div className="mb-6">
            <span className="text-xs font-mono uppercase tracking-wider text-[#E53935] font-semibold">
              Getting Started
            </span>
            <h2 className="text-xl font-mono font-bold text-white mt-1">
              INSTALLATION & INTEGRATION MANUAL
            </h2>
          </div>

          <div className="p-6 bg-[#13120f] rounded-2xl border border-[#2a2825]">
            <div className="flex gap-2 border-b border-[#2a2825] pb-3 mb-4">
              {(['install', 'api', 'tune', 'safety'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveManualTab(tab)}
                  className={`px-3 py-1.5 text-xs font-mono rounded-lg transition-colors uppercase ${
                    activeManualTab === tab
                      ? 'bg-[#E53935] text-white font-bold'
                      : 'text-[#A09D96] hover:text-white'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {activeManualTab === 'install' && (
              <div>
                <p className="text-xs text-[#A09D96] mb-3">
                  Install TritonForge via pip from GitHub or local source:
                </p>
                <pre className="p-4 bg-[#0a0a08] border border-[#2a2825] rounded-xl text-xs font-mono text-[#4CAF50] overflow-x-auto">
{`# Clone repository
git clone https://github.com/Gaurav711cgu/TritonForge.git
cd TritonForge

# Install dependencies with GPU support
pip install -e ".[gpu]"

# Run verification test suite (CPU fallback + GPU)
pytest tests/ -v`}
                </pre>
              </div>
            )}

            {activeManualTab === 'api' && (
              <div>
                <p className="text-xs text-[#A09D96] mb-3">
                  Drop-in replacement for standard PyTorch transformer layers:
                </p>
                <pre className="p-4 bg-[#0a0a08] border border-[#2a2825] rounded-xl text-xs font-mono text-[#4CAF50] overflow-x-auto">
{`import torch
from tritonforge.models.transformer_block import TritonForgeTransformerBlock

# Create fused transformer block (causal autoregressive decoder)
block = TritonForgeTransformerBlock(d_model=4096, n_heads=32, ffn_dim=11008)

# Input: [Batch=2, Seq_Len=2048, Hidden_Dim=4096]
x = torch.randn(2, 2048, 4096, device="cuda")
out = block(x)  # Executes Fused RMSNorm, FlashAttention-2, and SwiGLU FFN`}
                </pre>
              </div>
            )}

            {activeManualTab === 'tune' && (
              <div>
                <p className="text-xs text-[#A09D96] mb-3">
                  Autotuning configurations evaluated at runtime and cached:
                </p>
                <pre className="p-4 bg-[#0a0a08] border border-[#2a2825] rounded-xl text-xs font-mono text-[#4CAF50] overflow-x-auto">
{`from tritonforge.core.autotune import get_swiglu_autotune_configs, global_tune_cache

# Sweeps block sizes {128, 256, 512, 1024} with warps {4, 8}
configs = get_swiglu_autotune_configs()
print(f"Loaded {len(configs)} candidate configurations for target GPU architecture.")`}
                </pre>
              </div>
            )}

            {activeManualTab === 'safety' && (
              <div>
                <p className="text-xs text-[#A09D96] mb-3">
                  Memory safety guarantees: Contiguous row stride enforcement and block tiling:
                </p>
                <pre className="p-4 bg-[#0a0a08] border border-[#2a2825] rounded-xl text-xs font-mono text-[#4CAF50] overflow-x-auto">
{`from tritonforge.kernels.norm import fused_rmsnorm

# 3D and non-contiguous sliced tensors are automatically made contiguous:
x = torch.randn(4, 16, 512, device="cuda").transpose(0, 1)[:, :, :256]
weight = torch.ones(256, device="cuda")

# Guaranteed safe: view(-1, N) flattens rows, stride(0) addresses row_idx
y = fused_rmsnorm(x, weight)`}
                </pre>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ─── FOOTER ─── */}
      <footer className="tf-footer">
        <div className="footer-inner">
          <div className="footer-brand flex items-center gap-2">
            <img src="/logo-highres.png" alt="TritonForge Logo" className="brand-logo-img" />
            <span>TRITONFORGE &bull; L5 FAANG PORTFOLIO</span>
          </div>
          <div className="footer-links">
            <a href="https://github.com/Gaurav711cgu/TritonForge" target="_blank" rel="noreferrer">
              GitHub Repository
            </a>
            <span className="text-[#2a2825]">&bull;</span>
            <a href="#research">ArXiv Research Paper</a>
            <span className="text-[#2a2825]">&bull;</span>
            <button
              onClick={() => setIsTelemetryOpen(true)}
              className="hover:text-white transition-colors"
            >
              Open Telemetry Inspector
            </button>
          </div>
        </div>
      </footer>

      {/* ─── TELEMETRY MODAL ─── */}
      <TelemetryModal
        isOpen={isTelemetryOpen}
        onClose={() => setIsTelemetryOpen(false)}
        targetKernel={selectedKernel.name}
      />
    </main>
  );
}
