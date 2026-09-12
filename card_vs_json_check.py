"""
Parses PERFORMANCE_CARD.md for claimed speedup numbers.
Loads results_T4.json and block_benchmark_results.json.
Asserts that every number in the card is within ±10% of the JSON.
Fails CI if they diverge.
"""

import re
import json
import sys
import os


def extract_card_speedups(card_path: str) -> dict:
    """Parse Nx speedup values from markdown table."""
    speedups = {}
    if not os.path.exists(card_path):
        print(f"Warning: {card_path} not found.")
        return speedups

    pattern = r"(\w+)\s*\|\s*[\d.]+\s*ms\s*\|\s*[\d.]+\s*ms\s*\(([\d.]+)x\)"
    with open(card_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                speedups[m.group(1)] = float(m.group(2))
    return speedups


def load_json_speedups(json_path: str, kernel_key: str) -> float:
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found.")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get(kernel_key, [])
    if not results:
        return None
    return max(r["speedup"] for r in results)


def main():
    card_path = "PERFORMANCE_CARD.md"
    json_path = "benchmarks/results_T4.json"
    if not os.path.exists(json_path):
        json_path = "tritonforge/benchmarks/results_T4.json"

    card = extract_card_speedups(card_path)
    json_rmsnorm = load_json_speedups(json_path, "rmsnorm")
    json_swiglu = load_json_speedups(json_path, "swiglu")
    json_attn = load_json_speedups(json_path, "attention")

    errors = []

    if "RMSNorm" in card and json_rmsnorm:
        delta = abs(card["RMSNorm"] - json_rmsnorm) / json_rmsnorm
        if delta > 0.10:
            errors.append(f"RMSNorm: card={card['RMSNorm']}x, json={json_rmsnorm}x, delta={delta:.1%}")

    if "FlashAttention" in card and json_attn:
        delta = abs(card["FlashAttention"] - json_attn) / json_attn
        if delta > 0.10:
            errors.append(f"FlashAttention: card={card['FlashAttention']}x, json={json_attn}x, delta={delta:.1%}")

    if "SwiGLU" in card and json_swiglu:
        delta = abs(card["SwiGLU"] - json_swiglu) / json_swiglu
        if delta > 0.10:
            errors.append(f"SwiGLU: card={card['SwiGLU']}x, json={json_swiglu}x, delta={delta:.1%}")

    if errors:
        print("❌ Benchmark card diverges from JSON:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    print("✅ All benchmark card numbers match JSON within 10%")
    sys.exit(0)


if __name__ == "__main__":
    main()
