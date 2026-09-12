import { BenchmarkData } from "@/types";

export const defaultBenchmarks: BenchmarkData[] = [
  { label: 'RMSNorm N=512',   pytorch: 0.74, triton: 0.09, speedup: 8.2, unit: 'ms' },
  { label: 'RMSNorm N=2048',  pytorch: 3.59, triton: 0.44, speedup: 8.1, unit: 'ms' },
  { label: 'RMSNorm N=8192',  pytorch: 13.76,triton: 1.71, speedup: 8.0, unit: 'ms' },
  { label: 'SwiGLU N=512',    pytorch: 1.26, triton: 0.73, speedup: 1.7, unit: 'ms' },
  { label: 'SwiGLU N=4096',   pytorch: 13.24,triton: 7.84, speedup: 1.7, unit: 'ms' },
  { label: 'Attention N=1024',pytorch: 14.72,triton: 5.38, speedup: 2.7, unit: 'ms' },
  { label: 'Attention N=2048',pytorch: 61.35,triton: 18.4, speedup: 3.3, unit: 'ms' },
];

export const gpuArchitectures = [
  { id: 'T4',   name: 'Tesla T4 (Turing)',      peakBw: '320 GB/s',   smCount: 40,  sramPerSM: '64 KB' },
  { id: 'A100', name: 'NVIDIA A100 (Ampere)',   peakBw: '2,039 GB/s', smCount: 108, sramPerSM: '164 KB' },
  { id: 'H100', name: 'NVIDIA H100 (Hopper)',   peakBw: '3,350 GB/s', smCount: 132, sramPerSM: '228 KB' },
];

export const memoryTiers = [
  {
    name: 'REGISTERS (RF)',
    size: '256 KB / SM (A100)',
    bandwidth: '> 19,000 GB/s',
    latency: '1 cycle (~0.7 ns)',
    color: '#E53935',
    accent: 'rgba(229, 57, 53, 0.15)',
    tritonRole: 'Thread-level variables, intermediate math results, loop variables. Zero latency.',
    points: '160,20 240,20 270,70 130,70',
  },
  {
    name: 'SRAM / SHARED MEMORY',
    size: '164 KB / SM (A100)',
    bandwidth: '~ 19,000 GB/s',
    latency: '20–30 cycles (~15 ns)',
    color: '#FFA726',
    accent: 'rgba(255, 167, 38, 0.15)',
    tritonRole: 'Block tiles loaded via tl.load(). Reused across threads in a block. Eliminates HBM roundtrips.',
    points: '130,75 270,75 300,125 100,125',
  },
  {
    name: 'L2 CACHE',
    size: '40 MB (A100)',
    bandwidth: '~ 5,000 GB/s',
    latency: '200 cycles (~150 ns)',
    color: '#4CAF50',
    accent: 'rgba(76, 175, 80, 0.15)',
    tritonRole: 'Automatically caches HBM loads across SMs. Managed by hardware, but spatial locality maximizes hits.',
    points: '100,130 300,130 330,180 70,180',
  },
  {
    name: 'HBM2e (MAIN MEMORY)',
    size: '80 GB (A100)',
    bandwidth: '2,039 GB/s',
    latency: '400–800 cycles (~300 ns)',
    color: '#78909C',
    accent: 'rgba(120, 144, 156, 0.15)',
    tritonRole: 'Tensor storage. PyTorch eager hits this on EVERY op. Triton hits it once for input, once for output.',
    points: '70,185 330,185 360,235 40,235',
  },
];
