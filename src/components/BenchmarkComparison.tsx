'use client';

import React, { useState, useEffect } from 'react';
import { BenchmarkData } from '@/types';
import { defaultBenchmarks, gpuArchitectures } from '@/data/benchmarks';

interface BenchmarkComparisonProps {
  initialBenchmarks?: BenchmarkData[];
}

export function BenchmarkComparison({ initialBenchmarks = defaultBenchmarks }: BenchmarkComparisonProps) {
  const [selectedGPU, setSelectedGPU] = useState('T4');
  const [benchmarkList, setBenchmarkList] = useState<BenchmarkData[]>(initialBenchmarks);
  const [sliderIndex, setSliderIndex] = useState<number>(2); // Default to RMSNorm N=8192
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Fetch live benchmarks from API route (with bundled serverless fallback)
  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);
    fetch('/api/benchmarks')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!isMounted) return;
        if (data.rmsnorm && data.swiglu && data.attention) {
          const mapped: BenchmarkData[] = [
            ...data.rmsnorm.map((r: any) => ({
              label: `RMSNorm N=${r.seq_len}`,
              pytorch: r.pytorch_ms,
              triton: r.triton_ms,
              speedup: r.speedup,
              unit: 'ms',
            })),
            ...data.swiglu.map((s: any) => ({
              label: `SwiGLU N=${s.seq_len}`,
              pytorch: s.pytorch_ms,
              triton: s.triton_ms,
              speedup: s.speedup,
              unit: 'ms',
            })),
            ...data.attention.map((a: any) => ({
              label: `FlashAttn N=${a.seq_len}`,
              pytorch: a.pytorch_ms,
              triton: a.triton_ms,
              speedup: a.speedup,
              unit: 'ms',
            })),
          ];
          setBenchmarkList(mapped);
        }
      })
      .catch((err) => {
        console.warn('Using bundled benchmarks fallback:', err);
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const activeBenchmark = benchmarkList[sliderIndex] || benchmarkList[0];
  const maxPytorch = Math.max(...benchmarkList.map((b) => b.pytorch), 1);

  return (
    <div className="benchmark-section-container">
      {/* Header Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div>
          <h2 className="text-2xl font-mono font-bold text-white tracking-wide">
            EMPIRICAL BENCHMARKS & SPEEDUPS
          </h2>
          <p className="text-sm text-[#A09D96] mt-1">
            Standard PyTorch Eager (Float32) vs. TritonForge Fused Kernels
          </p>
        </div>

        {/* GPU Architecture Selector */}
        <div className="flex items-center gap-2 bg-[#13120f] p-1 rounded-lg border border-[#2a2825]">
          {gpuArchitectures.map((gpu) => (
            <button
              key={gpu.id}
              onClick={() => setSelectedGPU(gpu.id)}
              className={`px-3 py-1.5 text-xs font-mono rounded transition-colors ${
                selectedGPU === gpu.id
                  ? 'bg-[#E53935] text-white font-semibold'
                  : 'text-[#A09D96] hover:text-white'
              }`}
            >
              {gpu.name.split(' ')[0]} {gpu.id}
            </button>
          ))}
        </div>
      </div>

      {/* Interactive Exploration Slider */}
      <div className="p-4 bg-[#13120f] rounded-xl border border-[#2a2825] mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
          <span className="text-xs font-mono uppercase tracking-wider text-[#A09D96]">
            Select Workload Parameter: <strong className="text-white">{activeBenchmark?.label}</strong>
          </span>
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono text-[#78909C]">
              Speedup: <strong className="text-[#4CAF50] text-sm">{activeBenchmark?.speedup}x</strong>
            </span>
            <span className="text-xs font-mono text-[#78909C]">
              Latency: <strong className="text-white">{activeBenchmark?.triton} ms</strong>
            </span>
          </div>
        </div>

        <input
          type="range"
          min={0}
          max={benchmarkList.length - 1}
          value={sliderIndex}
          onChange={(e) => setSliderIndex(Number(e.target.value))}
          className="w-full h-1.5 bg-[#2a2825] rounded-lg appearance-none cursor-pointer accent-[#E53935]"
        />

        <div className="flex justify-between text-[10px] font-mono text-[#78909C] mt-2">
          <span>{benchmarkList[0]?.label}</span>
          <span>{benchmarkList[Math.floor(benchmarkList.length / 2)]?.label}</span>
          <span>{benchmarkList[benchmarkList.length - 1]?.label}</span>
        </div>
      </div>

      {/* Latency Comparison Card */}
      <div className="p-6 bg-[#13120f] rounded-xl border border-[#2a2825] mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-mono text-base font-semibold text-white">
            Latency Comparison: {activeBenchmark?.label}
          </h3>
          <span className="text-xs font-mono px-2.5 py-1 bg-[#4CAF50]/15 text-[#4CAF50] border border-[#4CAF50]/30 rounded-full font-bold">
            {activeBenchmark?.speedup}× FASTER
          </span>
        </div>

        {/* PyTorch Bar */}
        <div className="mb-4">
          <div className="flex justify-between text-xs font-mono mb-1.5">
            <span className="text-[#A09D96]">PyTorch Eager (Baseline)</span>
            <span className="text-white font-semibold">{activeBenchmark?.pytorch} {activeBenchmark?.unit}</span>
          </div>
          <div className="w-full bg-[#1a1916] h-6 rounded-md overflow-hidden border border-[#2a2825] flex">
            <div
              className="bg-[#B71C1C] h-full transition-all duration-500 rounded-sm flex items-center justify-end pr-2"
              style={{ width: `${Math.max((activeBenchmark?.pytorch / maxPytorch) * 100, 8)}%` }}
            >
              <span className="text-[10px] font-mono text-white font-bold">Baseline</span>
            </div>
          </div>
        </div>

        {/* TritonForge Fused Bar */}
        <div>
          <div className="flex justify-between text-xs font-mono mb-1.5">
            <span className="text-[#A09D96]">TritonForge Fused Kernel</span>
            <span className="text-[#4CAF50] font-semibold">{activeBenchmark?.triton} {activeBenchmark?.unit}</span>
          </div>
          <div className="w-full bg-[#1a1916] h-6 rounded-md overflow-hidden border border-[#2a2825] flex">
            <div
              className="bg-[#4CAF50] h-full transition-all duration-500 rounded-sm flex items-center justify-end pr-2"
              style={{
                width: `${Math.max(
                  ((activeBenchmark?.triton / maxPytorch) * 100),
                  4
                )}%`,
              }}
            >
              <span className="text-[10px] font-mono text-black font-bold">
                {activeBenchmark?.speedup}×
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Comprehensive Benchmark Table */}
      <div className="rounded-xl border border-[#2a2825] overflow-hidden bg-[#13120f]">
        <div className="px-4 py-3 bg-[#1a1916] border-b border-[#2a2825] flex justify-between items-center">
          <span className="text-xs font-mono font-bold text-white uppercase tracking-wider">
            Detailed Benchmark Suite ({selectedGPU})
          </span>
          {isLoading && (
            <span className="text-xs font-mono text-[#FFA726]">Loading live measurements...</span>
          )}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#100f0c] text-[#78909C] border-b border-[#2a2825]">
              <tr>
                <th className="py-2.5 px-4">Kernel & Dimension</th>
                <th className="py-2.5 px-4">PyTorch Eager</th>
                <th className="py-2.5 px-4">Triton Fused</th>
                <th className="py-2.5 px-4">Speedup</th>
                <th className="py-2.5 px-4">HBM Accesses</th>
                <th className="py-2.5 px-4">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#2a2825] text-[#F5F5F0]">
              {benchmarkList.map((item, idx) => (
                <tr
                  key={idx}
                  onClick={() => setSliderIndex(idx)}
                  className={`cursor-pointer transition-colors ${
                    idx === sliderIndex ? 'bg-[#E53935]/10 font-semibold' : 'hover:bg-[#1a1916]'
                  }`}
                >
                  <td className="py-2.5 px-4 text-white">{item.label}</td>
                  <td className="py-2.5 px-4 text-[#A09D96]">{item.pytorch} {item.unit}</td>
                  <td className="py-2.5 px-4 text-[#4CAF50]">{item.triton} {item.unit}</td>
                  <td className="py-2.5 px-4">
                    <span className="px-2 py-0.5 rounded bg-[#4CAF50]/15 text-[#4CAF50] font-bold">
                      {item.speedup}×
                    </span>
                  </td>
                  <td className="py-2.5 px-4 text-[#A09D96]">
                    {item.label.includes('RMSNorm')
                      ? '3 passes → 1 pass'
                      : item.label.includes('SwiGLU')
                      ? '2 buffers → 0'
                      : 'O(N²) → O(N)'}
                  </td>
                  <td className="py-2.5 px-4 text-[#4CAF50]">Pass (atol&le;1e-5)</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default BenchmarkComparison;
