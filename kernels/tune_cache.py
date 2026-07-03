"""
Offline Autotuning Cache for Triton Kernels.

Solves the cold-start problem: instead of recompiling and profiling
all Triton autotune configurations at the first forward pass in production,
this module:

  1. Runs an offline profiling sweep over common (M, N, K) shapes
     and saves the best config per shape to `triton_tune_cache.json`.

  2. At inference startup, loads the pre-computed cache and pins each
     kernel to its optimal config, eliminating all runtime compilation
     latency spikes.

Usage:
    # Run once during CI / model deployment pipeline
    python -m tritonforge.kernels.tune_cache --output triton_tune_cache.json

    # In your serving code, call before first forward pass:
    from tritonforge.kernels.tune_cache import load_tuning_cache
    load_tuning_cache("triton_tune_cache.json")
"""
import json
import argparse
import time
import torch

# Common LLM projection shapes to pre-tune for
# (M=sequence_len, N=out_features, K=hidden_dim)
_DEFAULT_SHAPES = [
    # Single-token generation (GEMV path)
    (1,   4096, 4096),
    (1,   8192, 4096),
    (1,  11008, 4096),
    # Short prefills
    (16,  4096, 4096),
    (32,  4096, 4096),
    (64,  4096, 4096),
    # Long prefills
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
]

# Candidate configs to sweep: (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
_CANDIDATE_CONFIGS = [
    (16,  64,  32, 4, 3),
    (32,  64,  32, 4, 3),
    (16, 128,  32, 4, 3),
    (32, 128,  64, 8, 4),
    (64, 128,  64, 8, 4),
    (32,  64,  64, 4, 4),
]


def _benchmark_config(x, norm_w, W, block_m, block_n, block_k, num_warps, num_stages, eps=1e-6, repeats=10):
    """Time a single kernel config over `repeats` forward passes."""
    try:
        import triton
        import triton.language as tl
        from tritonforge.kernels.fused_norm_linear import _rmsnorm_linear_gemm_kernel
    except ImportError:
        return float("inf")

    M, K = x.shape
    N, _ = W.shape
    Y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    is_aligned = (M % block_m == 0) and (K % block_k == 0) and (N % block_n == 0)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    # Warmup
    for _ in range(3):
        _rmsnorm_linear_gemm_kernel[grid](
            x, W, norm_w, Y,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1),
            M, N, K, eps,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            IS_ALIGNED=is_aligned,
            num_warps=num_warps, num_stages=num_stages,
        )

    if x.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        _rmsnorm_linear_gemm_kernel[grid](
            x, W, norm_w, Y,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1),
            M, N, K, eps,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            IS_ALIGNED=is_aligned,
            num_warps=num_warps, num_stages=num_stages,
        )
    if x.is_cuda:
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - t0) / repeats
    return elapsed


def run_offline_tuning(shapes=None, output_path="triton_tune_cache.json", device="cuda"):
    """
    Profile all candidate configs over the given shapes and save the best
    config per (M, N, K) shape to a JSON cache file.
    """
    shapes = shapes or _DEFAULT_SHAPES
    cache = {}

    print(f"[TuneCache] Starting offline autotuning sweep on {device}...")
    print(f"[TuneCache] Shapes: {len(shapes)}, Configs: {len(_CANDIDATE_CONFIGS)}")

    for (M, N, K) in shapes:
        if M == 1:
            # GEMV path has no autotune configs to sweep
            cache[f"{M},{N},{K}"] = {"path": "gemv"}
            print(f"  Shape ({M},{N},{K}) -> GEMV path (no autotune needed)")
            continue

        x      = torch.randn(M, K, device=device, dtype=torch.float32)
        norm_w = torch.ones(K,    device=device, dtype=torch.float32)
        W      = torch.randn(N, K, device=device, dtype=torch.float32)

        best_time   = float("inf")
        best_config = None

        for (bm, bn, bk, nw, ns) in _CANDIDATE_CONFIGS:
            # Skip configs where block dims exceed matrix dims
            if bm > M or bn > N or bk > K:
                continue

            t = _benchmark_config(x, norm_w, W, bm, bn, bk, nw, ns)
            if t < best_time:
                best_time   = t
                best_config = {
                    "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                    "num_warps": nw, "num_stages": ns,
                    "latency_ms": round(best_time * 1000, 4)
                }

        cache[f"{M},{N},{K}"] = best_config
        print(f"  Shape ({M:4d},{N:5d},{K:5d}) -> "
              f"BM={best_config['BLOCK_M']:2d} BN={best_config['BLOCK_N']:3d} "
              f"BK={best_config['BLOCK_K']:2d} | {best_config['latency_ms']:.3f}ms")

    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\n[TuneCache] Cache saved to '{output_path}' ({len(cache)} entries).")
    return cache


def load_tuning_cache(cache_path="triton_tune_cache.json"):
    """
    Load a pre-computed tuning cache and pin each kernel to its optimal
    config, bypassing all runtime autotuning overhead.

    Call this once at server startup before any forward passes.
    """
    try:
        from tritonforge.kernels.fused_norm_linear import _rmsnorm_linear_gemm_kernel
    except ImportError:
        print("[TuneCache] Triton not available. Skipping cache load.")
        return

    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
    except FileNotFoundError:
        print(f"[TuneCache] No cache found at '{cache_path}'. "
              f"Run `python -m tritonforge.kernels.tune_cache` to generate one.")
        return

    pinned = 0
    for shape_key, cfg in cache.items():
        if cfg.get("path") == "gemv":
            continue
        M, N, K = map(int, shape_key.split(","))
        key = (M, N, K)
        # Pin the kernel's autotune to the pre-computed best config
        _rmsnorm_linear_gemm_kernel.cache[key] = triton.Config(
            kwargs={
                "BLOCK_M":    cfg["BLOCK_M"],
                "BLOCK_N":    cfg["BLOCK_N"],
                "BLOCK_K":    cfg["BLOCK_K"],
                "IS_ALIGNED": (M % cfg["BLOCK_M"] == 0 and
                               K % cfg["BLOCK_K"] == 0 and
                               N % cfg["BLOCK_N"] == 0),
            },
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
        pinned += 1

    print(f"[TuneCache] Loaded {pinned} pre-tuned configs from '{cache_path}'. "
          f"Cold-start compilation eliminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Triton autotuning sweep")
    parser.add_argument("--output", default="triton_tune_cache.json",
                        help="Path to save the tuning cache JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run profiling on")
    args = parser.parse_args()
    run_offline_tuning(output_path=args.output, device=args.device)
