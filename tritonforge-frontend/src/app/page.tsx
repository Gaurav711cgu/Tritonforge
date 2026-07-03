"use client";

import React, { useState, useEffect } from "react";

interface RMSNormRow {
  seq_len: number;
  d_model: number;
  pytorch_ms: number;
  triton_ms: number;
  speedup: number;
  achieved_bw_gbs: number;
  bw_utilization_pct: number;
}

interface SwiGLURow {
  seq_len: number;
  input_dim: number;
  pytorch_ms: number;
  triton_ms: number;
  speedup: number;
  achieved_bw_gbs: number;
}

interface AttentionRow {
  seq_len: number;
  heads: number;
  head_dim: number;
  pytorch_ms: number;
  triton_ms: number;
  speedup: number;
  naive_mem_mb: number;
  fused_mem_mb: number;
  memory_saved_pct: number;
}

interface BenchmarkJSON {
  gpu: string;
  torch_version: string;
  cuda_version: string;
  peak_bw_gbs: number;
  rmsnorm: RMSNormRow[];
  swiglu: SwiGLURow[];
  attention: AttentionRow[];
}

export default function BenchmarkDashboard() {
  const [data, setData] = useState<BenchmarkJSON | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/benchmarks")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load benchmarks:", err);
        setError("Failed to load benchmark results. Make sure benchmarks/results_T4.json exists.");
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#06090e] text-slate-100 flex flex-col items-center justify-center gap-4">
        <div className="w-12 h-12 border-4 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin" />
        <p className="text-sm font-mono text-slate-400">Loading TritonForge Benchmarks...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-[#06090e] text-slate-100 flex flex-col items-center justify-center p-6 text-center">
        <div className="bg-red-950/20 border border-red-800 p-6 rounded-xl max-w-md">
          <h3 className="text-lg font-bold text-red-500 mb-2">Error Loading Data</h3>
          <p className="text-sm text-slate-400 mb-4">{error || "No benchmark data available."}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="px-4 py-2 bg-red-900/40 hover:bg-red-900/60 border border-red-700 text-xs font-semibold rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Get max values for gauges
  const maxRmsNormBwUtil = Math.max(...data.rmsnorm.map((r) => r.bw_utilization_pct), 0.0);
  const maxSwigluBwUtil = data.swiglu.length > 0 
    ? (Math.max(...data.swiglu.map((s) => s.achieved_bw_gbs), 0.0) / data.peak_bw_gbs) * 100
    : 0.0;
  const maxAttentionMemSaved = Math.max(...data.attention.map((a) => a.memory_saved_pct), 0.0);

  return (
    <div className="min-h-screen bg-[#06090e] text-slate-100 py-10 px-4 md:px-8">
      {/* HEADER */}
      <header className="max-w-7xl mx-auto mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6 border-b border-slate-900 pb-8">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <span className="text-[10px] font-bold font-mono tracking-widest text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded uppercase">
              TritonForge Workstation
            </span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
            GPU Kernel Optimization Benchmark
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Real-time memory bandwidth utilization and speedup analysis compared against standard PyTorch eager operators.
          </p>
        </div>
        
        {/* Hardware details */}
        <div className="flex flex-wrap gap-3">
          <div className="bg-[#0b0f17] border border-slate-800/80 px-3 py-1.5 rounded-lg flex flex-col gap-0.5">
            <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Target GPU Accelerator</span>
            <span className="text-xs font-mono font-bold text-white">{data.gpu}</span>
          </div>
          <div className="bg-[#0b0f17] border border-slate-800/80 px-3 py-1.5 rounded-lg flex flex-col gap-0.5">
            <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Peak Memory Bandwidth</span>
            <span className="text-xs font-mono font-bold text-white">{data.peak_bw_gbs} GB/s</span>
          </div>
          <div className="bg-[#0b0f17] border border-slate-800/80 px-3 py-1.5 rounded-lg flex flex-col gap-0.5">
            <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">CUDA & PyTorch</span>
            <span className="text-xs font-mono font-bold text-white">CUDA {data.cuda_version} · torch {data.torch_version}</span>
          </div>
        </div>
      </header>

      {/* CORE SECTIONS */}
      <main className="max-w-7xl mx-auto space-y-12">
        
        {/* 1. RMSNORM SECTION */}
        <section className="bg-[#0b0f17] border border-slate-900/60 p-6 md:p-8 rounded-2xl">
          <div className="flex items-center justify-between mb-6 border-b border-slate-900 pb-4">
            <div>
              <h2 className="text-xl font-bold text-white">1. Fused RMSNorm (`kernels/norm.py`)</h2>
              <p className="text-xs text-slate-400 mt-0.5">Single-kernel reduction + normalize + scale fusing 3 HBM transactions into 1.</p>
            </div>
            <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-1 rounded">
              Memory-Bound Kernel
            </span>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Speedup Bar Chart */}
            <BarChart 
              data={data.rmsnorm.map((r) => ({ x: `seq=${r.seq_len}`, y: r.speedup }))} 
              yLabel="Kernel Execution Speedup (vs. PyTorch reference)"
            />
            {/* Bandwidth Utilization Gauge */}
            <Gauge 
              value={maxRmsNormBwUtil} 
              label="Peak Bandwidth Attained" 
              subLabel="Memory Bandwidth Utilized"
              colorClass="stroke-emerald-500"
            />
            {/* Raw metrics table */}
            <div className="overflow-x-auto border border-slate-900 rounded-xl bg-slate-900/20">
              <table className="min-w-full divide-y divide-slate-900 text-xs font-mono">
                <thead className="bg-slate-900/50">
                  <tr className="text-slate-400 text-left">
                    <th className="px-4 py-3 font-semibold">Sequence Length</th>
                    <th className="px-4 py-3 font-semibold">PyTorch Latency</th>
                    <th className="px-4 py-3 font-semibold">Triton Latency</th>
                    <th className="px-4 py-3 font-semibold">Speedup</th>
                    <th className="px-4 py-3 font-semibold">Achieved BW</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-900 text-slate-300">
                  {data.rmsnorm.map((r, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/30">
                      <td className="px-4 py-3 font-bold text-white">{r.seq_len}</td>
                      <td className="px-4 py-3">{r.pytorch_ms.toFixed(4)} ms</td>
                      <td className="px-4 py-3">{r.triton_ms.toFixed(4)} ms</td>
                      <td className="px-4 py-3 text-emerald-400 font-bold">{r.speedup.toFixed(2)}x</td>
                      <td className="px-4 py-3">{r.achieved_bw_gbs.toFixed(1)} GB/s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* 2. SWIGLU SECTION */}
        <section className="bg-[#0b0f17] border border-slate-900/60 p-6 md:p-8 rounded-2xl">
          <div className="flex items-center justify-between mb-6 border-b border-slate-900 pb-4">
            <div>
              <h2 className="text-xl font-bold text-white">2. Fused SwiGLU (`kernels/activation.py`)</h2>
              <p className="text-xs text-slate-400 mt-0.5">Autotuned SiLU gating operator eliminating intermediate activation writes.</p>
            </div>
            <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-1 rounded">
              Autotuned Gated Activation
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Speedup Bar Chart */}
            <BarChart 
              data={data.swiglu.map((s) => ({ x: `seq=${s.seq_len}`, y: s.speedup }))} 
              yLabel="Kernel Execution Speedup (vs. PyTorch reference)"
            />
            {/* Peak BW util gauge */}
            <Gauge 
              value={maxSwigluBwUtil} 
              label="Peak Bandwidth Utilized" 
              subLabel="Fraction of Peak BW"
              colorClass="stroke-amber-500"
            />
            {/* Raw metrics table */}
            <div className="overflow-x-auto border border-slate-900 rounded-xl bg-slate-900/20">
              <table className="min-w-full divide-y divide-slate-900 text-xs font-mono">
                <thead className="bg-slate-900/50">
                  <tr className="text-slate-400 text-left">
                    <th className="px-4 py-3 font-semibold">Sequence Length</th>
                    <th className="px-4 py-3 font-semibold">PyTorch Latency</th>
                    <th className="px-4 py-3 font-semibold">Triton Latency</th>
                    <th className="px-4 py-3 font-semibold">Speedup</th>
                    <th className="px-4 py-3 font-semibold">Achieved BW</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-900 text-slate-300">
                  {data.swiglu.map((s, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/30">
                      <td className="px-4 py-3 font-bold text-white">{s.seq_len}</td>
                      <td className="px-4 py-3">{s.pytorch_ms.toFixed(4)} ms</td>
                      <td className="px-4 py-3">{s.triton_ms.toFixed(4)} ms</td>
                      <td className="px-4 py-3 text-emerald-400 font-bold">{s.speedup.toFixed(2)}x</td>
                      <td className="px-4 py-3">{s.achieved_bw_gbs.toFixed(1)} GB/s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* 3. FLASHATTENTION SECTION */}
        <section className="bg-[#0b0f17] border border-slate-900/60 p-6 md:p-8 rounded-2xl">
          <div className="flex items-center justify-between mb-6 border-b border-slate-900 pb-4">
            <div>
              <h2 className="text-xl font-bold text-white">3. Tiled FlashAttention (`kernels/attention.py`)</h2>
              <p className="text-xs text-slate-400 mt-0.5">Block-tiled attention forward pass utilizing online softmax reductions in SRAM.</p>
            </div>
            <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-1 rounded">
              O(N) Space Complexity
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Speedup Bar Chart */}
            <BarChart 
              data={data.attention.map((a) => ({ x: `seq=${a.seq_len}`, y: a.speedup }))} 
              yLabel="Kernel Execution Speedup (vs. PyTorch reference)"
            />
            {/* Memory saved gauge */}
            <Gauge 
              value={maxAttentionMemSaved} 
              label="Peak Memory Saved" 
              subLabel="VRAM Space Saved"
              colorClass="stroke-emerald-500"
            />
            {/* Raw metrics table */}
            <div className="overflow-x-auto border border-slate-900 rounded-xl bg-slate-900/20">
              <table className="min-w-full divide-y divide-slate-900 text-xs font-mono">
                <thead className="bg-slate-900/50">
                  <tr className="text-slate-400 text-left">
                    <th className="px-4 py-3 font-semibold">Sequence Length</th>
                    <th className="px-4 py-3 font-semibold">PyTorch Latency</th>
                    <th className="px-4 py-3 font-semibold">Triton Latency</th>
                    <th className="px-4 py-3 font-semibold">Speedup</th>
                    <th className="px-4 py-3 font-semibold">Memory Saved</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-900 text-slate-300">
                  {data.attention.map((a, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/30">
                      <td className="px-4 py-3 font-bold text-white">{a.seq_len}</td>
                      <td className="px-4 py-3">{a.pytorch_ms.toFixed(4)} ms</td>
                      <td className="px-4 py-3">{a.triton_ms.toFixed(4)} ms</td>
                      <td className="px-4 py-3 text-emerald-400 font-bold">{a.speedup.toFixed(2)}x</td>
                      <td className="px-4 py-3 text-amber-500">{a.memory_saved_pct.toFixed(1)}% ({a.fused_mem_mb.toFixed(1)}MB vs {a.naive_mem_mb.toFixed(1)}MB)</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}

// --- SUB-COMPONENTS ---

interface BarChartProps {
  data: { x: string; y: number }[];
  yLabel: string;
}

const BarChart: React.FC<BarChartProps> = ({ data, yLabel }) => {
  const maxVal = Math.max(...data.map((d) => d.y), 1.0);
  
  return (
    <div className="bg-slate-900/30 border border-slate-900/60 p-4 rounded-xl flex flex-col">
      <h4 className="text-xs font-semibold text-slate-400 mb-4">{yLabel}</h4>
      <div className="flex items-end gap-3 h-36 pt-4 px-2 mt-auto">
        {data.map((item, idx) => {
          const heightPct = (item.y / maxVal) * 80 + 10; // offset slightly for visibility
          return (
            <div key={idx} className="flex-1 flex flex-col items-center gap-1 group">
              <div className="w-full relative flex items-end justify-center h-24">
                {/* Tooltip */}
                <div className="absolute -top-6 bg-emerald-500 text-slate-950 text-[10px] font-bold px-1.5 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 font-mono">
                  {item.y.toFixed(2)}x
                </div>
                {/* Bar */}
                <div 
                  style={{ height: `${heightPct}%` }}
                  className="w-full bg-emerald-500/20 group-hover:bg-emerald-500/40 border-t-2 border-emerald-500 rounded-t transition-all duration-300"
                />
              </div>
              <span className="text-[9px] font-mono text-slate-500 mt-1 whitespace-nowrap">{item.x}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

interface GaugeProps {
  value: number;
  label: string;
  subLabel: string;
  colorClass?: string;
}

const Gauge: React.FC<GaugeProps> = ({ value, label, subLabel, colorClass = "stroke-amber-500" }) => {
  const radius = 40;
  const strokeWidth = 6;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (Math.min(value, 100) / 100) * circumference;

  return (
    <div className="bg-slate-900/30 border border-slate-900/60 p-4 rounded-xl flex flex-col items-center justify-center">
      <h4 className="text-xs font-semibold text-slate-400 mb-2 text-center w-full">{label}</h4>
      <div className="relative w-28 h-28 flex items-center justify-center mt-2">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            className="stroke-slate-800/80 fill-none"
            strokeWidth={strokeWidth}
          />
          {/* Active progress arc */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            className={`fill-none transition-all duration-1000 ${colorClass}`}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute text-center">
          <span className="text-xl font-bold font-mono text-white">{value.toFixed(1)}%</span>
          <p className="text-[8px] text-slate-500 font-medium tracking-wider uppercase mt-0.5">{subLabel}</p>
        </div>
      </div>
    </div>
  );
};
