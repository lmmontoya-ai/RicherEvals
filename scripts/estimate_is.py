"""Importance sampling estimator for p(Z|P) from a generations JSON artifact."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for importance sampling estimation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to generations.json")
    parser.add_argument("--label", default="Z4", help="Z label to estimate")
    parser.add_argument("--clip", type=float, default=None, help="Optional weight clip")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap samples")
    parser.add_argument("--seed", type=int, default=1337, help="Bootstrap seed")
    return parser.parse_args()


def estimate_k(weights: List[float], tail_frac: float = 0.2) -> Optional[float]:
    """Approximate PSIS k from the tail of the weight distribution."""
    w = [x for x in weights if x > 0]
    n = len(w)
    if n < 10:
        return None
    w.sort()
    m = max(5, int(n * tail_frac))
    tail = w[-m:]
    threshold = tail[0]
    excess = [x - threshold for x in tail if x > threshold]
    if len(excess) < 5:
        return None
    mean = sum(excess) / len(excess)
    var = sum((x - mean) ** 2 for x in excess) / max(len(excess) - 1, 1)
    if var <= 0:
        return None
    k = 0.5 * (1.0 - (mean * mean) / var)
    return k


def compute_estimate(weights: List[float], indicators: List[float]) -> float:
    """Compute the self-normalized IS estimate for a binary event."""
    w_sum = sum(weights)
    if w_sum == 0:
        return 0.0
    return sum(w * z for w, z in zip(weights, indicators)) / w_sum


def main() -> None:
    """Load a generations file and report IS diagnostics."""
    args = parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        raise SystemExit("No results found")

    weights = []
    indicators = []

    for item in results:
        log_p = float(item.get("logprob_sum_p", 0.0))
        log_q = float(item.get("logprob_sum_mix", 0.0))
        w = math.exp(log_p - log_q)
        if args.clip is not None:
            w = min(w, args.clip)
        z = bool(item.get("detections_final", {}).get(args.label, False))
        weights.append(w)
        indicators.append(1.0 if z else 0.0)

    estimate = compute_estimate(weights, indicators)
    naive = sum(indicators) / len(indicators)

    w_sum = sum(weights)
    w_sq_sum = sum(w * w for w in weights)
    ess = (w_sum * w_sum) / w_sq_sum if w_sq_sum > 0 else 0.0
    max_w = max(weights) if weights else 0.0
    max_w_share = max_w / w_sum if w_sum > 0 else 0.0
    k_hat = estimate_k(weights)

    ci = None
    if args.bootstrap and args.bootstrap > 0:
        rng = random.Random(args.seed)
        boot = []
        n = len(weights)
        for _ in range(args.bootstrap):
            idx = [rng.randrange(n) for _ in range(n)]
            b_w = [weights[i] for i in idx]
            b_z = [indicators[i] for i in idx]
            boot.append(compute_estimate(b_w, b_z))
        boot.sort()
        lo = boot[int(0.025 * len(boot))]
        hi = boot[int(0.975 * len(boot))]
        ci = [lo, hi]

    output = {
        "n": len(indicators),
        "label": args.label,
        "estimate": estimate,
        "ess": ess,
        "max_weight_share": max_w_share,
        "psis_k": k_hat,
        "naive": naive,
        "bootstrap_ci": ci,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
