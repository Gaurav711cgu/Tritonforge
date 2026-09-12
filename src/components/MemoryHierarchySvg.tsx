'use client';

import React, { useState } from 'react';
import { memoryTiers } from '@/data/benchmarks';

export function MemoryHierarchySvg() {
  const [activeTier, setActiveTier] = useState<number>(0);
  const [hoveredTier, setHoveredTier] = useState<number | null>(null);

  const selectedTier = hoveredTier !== null ? memoryTiers[hoveredTier] : memoryTiers[activeTier];

  return (
    <div className="memory-hierarchy-container p-6 bg-[#13120f] rounded-2xl border border-[#2a2825]">
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6 mb-6">
        <div>
          <span className="text-xs font-mono uppercase tracking-wider text-[#E53935] font-semibold">
            GPU Microarchitecture Insights
          </span>
          <h2 className="text-xl font-mono font-bold text-white mt-1">
            GPU MEMORY HIERARCHY & THE FUSION ADVANTAGE
          </h2>
          <p className="text-xs text-[#A09D96] mt-1 max-w-2xl">
            Why kernel fusion matters: On-chip registers and SRAM provide ~19 TB/s bandwidth at ~1-20 cycle latency,
            whereas global HBM provides only ~2 TB/s at 400-800 cycles latency.
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono text-[#78909C]">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#E53935]"></span>
            On-Chip (Fast)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#78909C]"></span>
            Off-Chip HBM (Slow)
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
        {/* Interactive SVG Diagram */}
        <div className="lg:col-span-6 flex justify-center">
          <svg
            viewBox="0 0 400 260"
            className="w-full max-w-[420px] h-auto drop-shadow-2xl select-none"
          >
            <defs>
              <linearGradient id="tier-gradient-0" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#E53935" />
                <stop offset="100%" stopColor="#B71C1C" />
              </linearGradient>
              <linearGradient id="tier-gradient-1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#FFA726" />
                <stop offset="100%" stopColor="#F57C00" />
              </linearGradient>
              <linearGradient id="tier-gradient-2" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#4CAF50" />
                <stop offset="100%" stopColor="#2E7D32" />
              </linearGradient>
              <linearGradient id="tier-gradient-3" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#78909C" />
                <stop offset="100%" stopColor="#37474F" />
              </linearGradient>
            </defs>

            {memoryTiers.map((tier, index) => {
              const isSelected = (hoveredTier !== null ? hoveredTier : activeTier) === index;
              return (
                <polygon
                  key={tier.name}
                  points={tier.points}
                  fill={`url(#tier-gradient-${index})`}
                  opacity={isSelected ? 1.0 : 0.65}
                  stroke={isSelected ? '#ffffff' : '#2a2825'}
                  strokeWidth={isSelected ? 2.5 : 1}
                  className="transition-all duration-300 cursor-pointer"
                  onClick={() => setActiveTier(index)}
                  onMouseEnter={() => setHoveredTier(index)}
                  onMouseLeave={() => setHoveredTier(null)}
                />
              );
            })}

            {/* Labels in the SVG */}
            <text x="200" y="48" fill="#ffffff" fontSize="10" fontWeight="bold" textAnchor="middle" fontFamily="monospace">
              REGISTERS (1 cycle, &gt;19 TB/s)
            </text>
            <text x="200" y="103" fill="#ffffff" fontSize="10" fontWeight="bold" textAnchor="middle" fontFamily="monospace">
              SRAM / SHARED MEMORY (~20 cycles)
            </text>
            <text x="200" y="158" fill="#ffffff" fontSize="10" fontWeight="bold" textAnchor="middle" fontFamily="monospace">
              L2 CACHE (~200 cycles, ~5 TB/s)
            </text>
            <text x="200" y="213" fill="#ffffff" fontSize="10" fontWeight="bold" textAnchor="middle" fontFamily="monospace">
              HBM2e / HBM3 (400-800 cycles, ~2 TB/s)
            </text>
          </svg>
        </div>

        {/* Dynamic Tier Inspection Card */}
        <div className="lg:col-span-6 flex flex-col gap-4">
          <div
            className="p-5 rounded-xl border transition-all duration-300"
            style={{
              backgroundColor: '#1a1916',
              borderColor: selectedTier.color,
            }}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono font-bold text-white px-2.5 py-1 rounded bg-black/40 border border-[#2a2825]">
                {selectedTier.name}
              </span>
              <span className="text-xs font-mono font-semibold" style={{ color: selectedTier.color }}>
                {selectedTier.latency}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="p-3 bg-[#13120f] rounded-lg border border-[#2a2825]">
                <span className="text-[10px] font-mono text-[#78909C] uppercase block">Capacity</span>
                <span className="text-xs font-mono font-bold text-white">{selectedTier.size}</span>
              </div>
              <div className="p-3 bg-[#13120f] rounded-lg border border-[#2a2825]">
                <span className="text-[10px] font-mono text-[#78909C] uppercase block">Bandwidth</span>
                <span className="text-xs font-mono font-bold text-[#4CAF50]">{selectedTier.bandwidth}</span>
              </div>
            </div>

            <div className="p-3 bg-[#13120f] rounded-lg border border-[#2a2825]">
              <span className="text-[10px] font-mono text-[#E53935] uppercase tracking-wider block mb-1">
                TritonForge Execution Invariant
              </span>
              <p className="text-xs text-[#F5F5F0] leading-relaxed">
                {selectedTier.tritonRole}
              </p>
            </div>
          </div>

          {/* PyTorch vs Triton Data Movement Comparison */}
          <div className="p-4 bg-[#1a1916] rounded-xl border border-[#2a2825] text-xs font-mono">
            <span className="text-[#78909C] uppercase text-[10px] block mb-2 font-bold">
              Memory Movement Comparison (RMSNorm Layer):
            </span>
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between p-2 rounded bg-[#B71C1C]/10 border border-[#B71C1C]/20 text-[#FFA726]">
                <span>PyTorch Eager:</span>
                <span className="font-bold text-white">3 Read + 3 Write Roundtrips to HBM</span>
              </div>
              <div className="flex items-center justify-between p-2 rounded bg-[#4CAF50]/10 border border-[#4CAF50]/20 text-[#4CAF50]">
                <span>TritonForge Fused:</span>
                <span className="font-bold text-white">1 Read + 1 Write (All intermediates in SRAM)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MemoryHierarchySvg;
