export interface BenchmarkData {
  label: string;
  pytorch: number;
  triton: number;
  speedup: number;
  unit: string;
}

export interface KernelSpec {
  id: string;
  name: string;
  tag: string;
  color: string;
  accent: string;
  description: string;
  math: string;
  highlights: string[];
  fallback: string;
  code: string;
  naiveCode: string;
}

export interface PaperSection {
  title: string;
  content: string;
}

export interface Paper {
  id: string;
  title: string;
  authors: string;
  date: string;
  abstract: string;
  sections: PaperSection[];
  tags: string[];
  doi: string;
  bibtex: string;
}

export interface MemoryTier {
  name: string;
  size: string;
  bandwidth: string;
  latency: string;
  color: string;
  accent: string;
  tritonRole: string;
  points: string;
}

export interface TelemetryMetrics {
  gpuModel: string;
  driverVersion: string;
  cudaVersion: string;
  smOccupancyPct: number;
  tensorCoreUtilPct: number;
  hbmBandwidthGbs: number;
  hbmBandwidthPct: number;
  registersPerThread: number;
  powerDrawWatts: number;
  temperatureC: number;
  activeWarps: number;
  sharedMemPerBlockKb: number;
}
