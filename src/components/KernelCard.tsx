'use client';

import React, { useState } from 'react';
import { KernelSpec } from '@/types';

interface KernelCardProps {
  kernel: KernelSpec;
  active: boolean;
  onClick: () => void;
}

export function KernelSelectorCard({ kernel, active, onClick }: KernelCardProps) {
  return (
    <div
      onClick={onClick}
      className={`kernel-card ${active ? 'active' : ''}`}
      style={{
        cursor: 'pointer',
        borderLeft: active ? `3px solid ${kernel.accent}` : '3px solid transparent',
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-mono text-base font-semibold text-white tracking-wide">
          {kernel.name}
        </h3>
        <span
          className="text-xs px-2 py-0.5 rounded-full font-mono font-medium"
          style={{
            backgroundColor: `${kernel.accent}22`,
            color: kernel.accent,
            border: `1px solid ${kernel.accent}44`,
          }}
        >
          {kernel.tag}
        </span>
      </div>
      <p className="text-xs text-[#A09D96] line-clamp-2 mb-3">
        {kernel.description}
      </p>
      <div className="flex items-center justify-between text-xs font-mono text-[#78909C]">
        <span>Fallback: Protected</span>
        <span className="text-[#E53935] hover:underline flex items-center gap-1">
          Inspect Kernel &rarr;
        </span>
      </div>
    </div>
  );
}

interface KernelPlaygroundProps {
  kernel: KernelSpec;
}

export function KernelPlayground({ kernel }: KernelPlaygroundProps) {
  const [viewMode, setViewMode] = useState<'triton' | 'naive'>('triton');
  const [copied, setCopied] = useState(false);

  const activeCode = viewMode === 'triton' ? kernel.code : kernel.naiveCode;

  const handleCopy = () => {
    navigator.clipboard.writeText(activeCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="kernel-display-card">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-[#2a2825]">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-mono font-bold text-white">{kernel.name}</h2>
            <span
              className="text-xs px-2.5 py-0.5 rounded-full font-mono font-medium"
              style={{
                backgroundColor: `${kernel.accent}22`,
                color: kernel.accent,
                border: `1px solid ${kernel.accent}44`,
              }}
            >
              {kernel.tag}
            </span>
          </div>
          <p className="text-sm text-[#A09D96] mt-1">{kernel.description}</p>
        </div>

        {/* View mode toggle */}
        <div className="flex items-center gap-2 bg-[#13120f] p-1 rounded-lg border border-[#2a2825] self-start md:self-auto">
          <button
            onClick={() => setViewMode('triton')}
            className={`px-3 py-1 text-xs font-mono rounded transition-colors ${
              viewMode === 'triton'
                ? 'bg-[#E53935] text-white font-semibold'
                : 'text-[#A09D96] hover:text-white'
            }`}
          >
            Triton Kernel (Optimized)
          </button>
          <button
            onClick={() => setViewMode('naive')}
            className={`px-3 py-1 text-xs font-mono rounded transition-colors ${
              viewMode === 'naive'
                ? 'bg-[#B71C1C] text-white font-semibold'
                : 'text-[#A09D96] hover:text-white'
            }`}
          >
            PyTorch Naive (Eager)
          </button>
        </div>
      </div>

      {/* Math & Highlights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-4">
        <div className="p-3.5 bg-[#13120f] rounded-lg border border-[#2a2825]">
          <span className="text-xs font-mono text-[#78909C] uppercase tracking-wider block mb-1">
            Mathematical Formulation
          </span>
          <code className="text-sm font-mono text-[#F5F5F0]">{kernel.math}</code>
        </div>
        <div className="p-3.5 bg-[#13120f] rounded-lg border border-[#2a2825]">
          <span className="text-xs font-mono text-[#78909C] uppercase tracking-wider block mb-1">
            Hardware Guard & Dynamic Fallback
          </span>
          <span className="text-xs font-mono text-[#FFA726]">{kernel.fallback}</span>
        </div>
      </div>

      {/* Code Inspector */}
      <div className="relative rounded-lg overflow-hidden border border-[#2a2825] bg-[#090908]">
        <div className="flex items-center justify-between px-4 py-2.5 bg-[#13120f] border-b border-[#2a2825] text-xs font-mono">
          <span className="text-[#A09D96]">
            {viewMode === 'triton' ? 'tritonforge/kernels/' + kernel.id + '.py' : 'naive_reference.py'}
          </span>
          <button
            onClick={handleCopy}
            className="text-xs text-[#A09D96] hover:text-white transition-colors flex items-center gap-1.5"
          >
            {copied ? (
              <span className="text-[#4CAF50] font-medium">Copied to Clipboard!</span>
            ) : (
              <span>Copy Snippet</span>
            )}
          </button>
        </div>

        <pre className="p-4 text-xs font-mono text-[#F5F5F0] overflow-x-auto leading-relaxed max-h-[420px]">
          <code>{activeCode}</code>
        </pre>
      </div>

      {/* Architectural Highlights */}
      <div className="mt-4 pt-4 border-t border-[#2a2825]">
        <span className="text-xs font-mono text-[#78909C] uppercase tracking-wider block mb-2">
          Key Performance & Safety Invariants:
        </span>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {kernel.highlights.map((highlight, idx) => (
            <div key={idx} className="flex items-start gap-2 text-xs text-[#A09D96]">
              <span className="text-[#E53935] mt-0.5 font-mono">&#x25B8;</span>
              <span>{highlight}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default KernelSelectorCard;
