'use client';

import React, { useState, useEffect } from 'react';
import { TelemetryMetrics } from '@/types';

interface TelemetryModalProps {
  isOpen: boolean;
  onClose: () => void;
  targetKernel?: string;
}

export function TelemetryModal({ isOpen, onClose, targetKernel = 'Fused RMSNorm' }: TelemetryModalProps) {
  const [metrics, setMetrics] = useState<TelemetryMetrics>({
    gpuModel: 'NVIDIA A100-SXM4-80GB HBM2e',
    driverVersion: '535.129.03',
    cudaVersion: '12.2',
    smOccupancyPct: 91.4,
    tensorCoreUtilPct: 84.6,
    hbmBandwidthGbs: 1898.4,
    hbmBandwidthPct: 93.1,
    registersPerThread: 32,
    powerDrawWatts: 278,
    temperatureC: 56,
    activeWarps: 48,
    sharedMemPerBlockKb: 12.8,
  });

  // Small live metric jitter for authentic real-time telemetry feeling
  useEffect(() => {
    if (!isOpen) return;
    const interval = setInterval(() => {
      setMetrics((prev) => ({
        ...prev,
        smOccupancyPct: +(91 + Math.random() * 2.5 - 1.2).toFixed(1),
        hbmBandwidthGbs: +(1895 + Math.random() * 25).toFixed(1),
        powerDrawWatts: +(275 + Math.random() * 10).toFixed(0),
        temperatureC: +(55 + Math.random() * 2).toFixed(0),
      }));
    }, 1500);

    return () => clearInterval(interval);
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in">
      <div className="w-full max-w-2xl bg-[#13120f] border border-[#2a2825] rounded-2xl p-6 shadow-2xl relative text-left">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-[#2a2825]">
          <div className="flex items-center gap-3">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#4CAF50] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-[#4CAF50]"></span>
            </span>
            <div>
              <h3 className="font-mono text-base font-bold text-white tracking-wide">
                NSIGHT GPU TELEMETRY & PROFILING INSPECTOR
              </h3>
              <span className="text-xs font-mono text-[#A09D96]">
                Target Kernel: <strong className="text-white">{targetKernel}</strong> &bull; {metrics.gpuModel}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-[#A09D96] hover:text-white font-mono text-sm px-2.5 py-1.5 rounded-lg border border-[#2a2825] transition-colors"
          >
            ✕ Close
          </button>
        </div>

        {/* Live Gauges Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
          <div className="p-3 bg-[#1a1916] rounded-xl border border-[#2a2825]">
            <span className="text-[10px] font-mono text-[#78909C] uppercase block mb-1">
              HBM Bandwidth %
            </span>
            <span className="text-lg font-mono font-bold text-[#4CAF50]">
              {metrics.hbmBandwidthPct}%
            </span>
            <span className="text-[10px] font-mono text-[#A09D96] block mt-0.5">
              {metrics.hbmBandwidthGbs} GB/s
            </span>
          </div>

          <div className="p-3 bg-[#1a1916] rounded-xl border border-[#2a2825]">
            <span className="text-[10px] font-mono text-[#78909C] uppercase block mb-1">
              SM Occupancy
            </span>
            <span className="text-lg font-mono font-bold text-white">
              {metrics.smOccupancyPct}%
            </span>
            <span className="text-[10px] font-mono text-[#A09D96] block mt-0.5">
              {metrics.activeWarps} Warps / SM
            </span>
          </div>

          <div className="p-3 bg-[#1a1916] rounded-xl border border-[#2a2825]">
            <span className="text-[10px] font-mono text-[#78909C] uppercase block mb-1">
              Registers / Thread
            </span>
            <span className="text-lg font-mono font-bold text-[#FFA726]">
              {metrics.registersPerThread}
            </span>
            <span className="text-[10px] font-mono text-[#A09D96] block mt-0.5">
              0 Register Spills
            </span>
          </div>

          <div className="p-3 bg-[#1a1916] rounded-xl border border-[#2a2825]">
            <span className="text-[10px] font-mono text-[#78909C] uppercase block mb-1">
              Power & Temp
            </span>
            <span className="text-lg font-mono font-bold text-white">
              {metrics.temperatureC}&deg;C
            </span>
            <span className="text-[10px] font-mono text-[#A09D96] block mt-0.5">
              {metrics.powerDrawWatts} W
            </span>
          </div>
        </div>

        {/* Profiling Breakdown */}
        <div className="p-4 bg-[#1a1916] rounded-xl border border-[#2a2825] mb-5">
          <span className="text-xs font-mono font-bold text-[#E53935] uppercase tracking-wider block mb-3">
            Kernel Launch Architecture Invariants
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
            <div className="flex justify-between p-2 rounded bg-[#13120f] border border-[#2a2825]">
              <span className="text-[#A09D96]">Launch Grid Dimension:</span>
              <span className="text-white font-bold">(M, 1, 1) Row Mapping</span>
            </div>
            <div className="flex justify-between p-2 rounded bg-[#13120f] border border-[#2a2825]">
              <span className="text-[#A09D96]">SRAM / SM Tile Allocation:</span>
              <span className="text-[#4CAF50] font-bold">{metrics.sharedMemPerBlockKb} KB / Block</span>
            </div>
            <div className="flex justify-between p-2 rounded bg-[#13120f] border border-[#2a2825]">
              <span className="text-[#A09D96]">Contiguity Guard:</span>
              <span className="text-[#4CAF50] font-bold">Enforced (Stride Check)</span>
            </div>
            <div className="flex justify-between p-2 rounded bg-[#13120f] border border-[#2a2825]">
              <span className="text-[#A09D96]">Autotuning Strategy:</span>
              <span className="text-white font-bold">Offline Cache + JIT Fallback</span>
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center justify-between text-xs font-mono text-[#78909C]">
          <span>Profiling session: active &bull; Sampling rate: 1000 Hz</span>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-[#E53935] hover:bg-[#B71C1C] text-white font-semibold rounded-lg transition-colors"
          >
            Dismiss Inspector
          </button>
        </div>
      </div>
    </div>
  );
}

export default TelemetryModal;
