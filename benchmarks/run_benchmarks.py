import json
import time
import torch
from pathlib import Path
from tritonforge.kernels.norm import fused_rmsnorm, pytorch_rmsnorm
from tritonforge.kernels.activation import fused_swiglu, pytorch_swiglu
from tritonforge.kernels.attention import fused_attention, pytorch_flash_attention
from tritonforge.core.profiler import profile_op, estimate_metrics

assert torch.cuda.is_available(), "Run on a CUDA GPU (Colab T4 or better)"

GPU_NAME = torch.cuda.get_device_name(0)
GPU_PEAK_BW = {
    "Tesla T4": 320,
    "Tesla V100": 900,
    "A100": 2000,
    "H100 SXM5": 3350,
}
peak_bw = GPU_PEAK_BW.get(GPU_NAME.split("-")[0].strip(), 300)

print(f"GPU: {GPU_NAME} | Peak BW: {peak_bw} GB/s")

results = {"gpu": GPU_NAME, "torch_version": torch.__version__,
           "cuda_version": torch.version.cuda, "peak_bw_gbs": peak_bw,
           "rmsnorm": [], "swiglu": [], "attention": []}

# ── RMSNorm benchmark ──────────────────────────────────────────────────────
for seq_len in [512, 1024, 2048, 4096, 8192]:
    d = 2304  # Gemma-2-2b-it d_model
    x = torch.randn(1, seq_len, d, device="cuda", dtype=torch.float16).contiguous()
    w = torch.ones(d, device="cuda", dtype=torch.float16).contiguous()

    pytorch_ms = profile_op(pytorch_rmsnorm, x, w, warmups=20, reps=100)
    triton_ms = profile_op(fused_rmsnorm, x, w, warmups=20, reps=100)
    speedup = pytorch_ms / triton_ms

    # Memory: read X + read W + write Y = 3 * seq_len * d * 2 bytes (fp16)
    bytes_io = 3 * seq_len * d * 2
    bw_achieved, _ = estimate_metrics(triton_ms, bytes_io)
    bw_utilization = bw_achieved / peak_bw * 100

    row = {
        "seq_len": seq_len, "d_model": d,
        "pytorch_ms": round(pytorch_ms, 4),
        "triton_ms": round(triton_ms, 4),
        "speedup": round(speedup, 2),
        "achieved_bw_gbs": round(bw_achieved, 1),
        "bw_utilization_pct": round(bw_utilization, 1),
    }
    results["rmsnorm"].append(row)
    print(f"RMSNorm seq={seq_len:5d}: {speedup:.2f}× | {bw_utilization:.0f}% BW util")

# ── SwiGLU benchmark ───────────────────────────────────────────────────────
for seq_len in [512, 1024, 2048, 4096]:
    d = 4608  # 2 * d_model (SwiGLU input is doubled)
    x = torch.randn(1, seq_len, d, device="cuda", dtype=torch.float16).contiguous()

    pytorch_ms = profile_op(pytorch_swiglu, x, warmups=20, reps=100)
    triton_ms = profile_op(fused_swiglu, x, warmups=20, reps=100)
    speedup = pytorch_ms / triton_ms

    bytes_io = (d + d // 2) * seq_len * 2  # read 2N, write N
    bw_achieved, _ = estimate_metrics(triton_ms, bytes_io)

    row = {
        "seq_len": seq_len, "input_dim": d,
        "pytorch_ms": round(pytorch_ms, 4),
        "triton_ms": round(triton_ms, 4),
        "speedup": round(speedup, 2),
        "achieved_bw_gbs": round(bw_achieved, 1),
    }
    results["swiglu"].append(row)
    print(f"SwiGLU  seq={seq_len:5d}: {speedup:.2f}×")

# ── FlashAttention benchmark ───────────────────────────────────────────────
for seq_len in [256, 512, 1024, 2048]:
    B, H, d = 1, 8, 64
    q = torch.randn(B, H, seq_len, d, device="cuda", dtype=torch.float16).contiguous()
    k, v = torch.randn_like(q), torch.randn_like(q)

    pytorch_ms = profile_op(pytorch_flash_attention, q, k, v, warmups=20, reps=100)
    triton_ms = profile_op(fused_attention, q, k, v, warmups=20, reps=100)
    speedup = pytorch_ms / triton_ms

    # Memory saved: O(N) vs O(N²)
    naive_mem_mb = B * H * seq_len * seq_len * 4 / 1e6  # fp32 attention matrix
    fused_mem_mb = B * H * seq_len * d * 2 / 1e6 * 3    # Q,K,V only

    row = {
        "seq_len": seq_len, "heads": H, "head_dim": d,
        "pytorch_ms": round(pytorch_ms, 4),
        "triton_ms": round(triton_ms, 4),
        "speedup": round(speedup, 2),
        "naive_mem_mb": round(naive_mem_mb, 1),
        "fused_mem_mb": round(fused_mem_mb, 1),
        "memory_saved_pct": round((1 - fused_mem_mb/naive_mem_mb)*100, 1),
    }
    results["attention"].append(row)
    print(f"Attn    seq={seq_len:5d}: {speedup:.2f}× | {row['memory_saved_pct']}% mem saved")

Path("benchmarks").mkdir(exist_ok=True)
Path("benchmarks/results_T4.json").write_text(json.dumps(results, indent=2))
print("\nResults saved to benchmarks/results_T4.json — commit this file.")
