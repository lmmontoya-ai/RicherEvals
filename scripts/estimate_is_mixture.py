"""Importance sampling estimator for mixture proposals with logged alpha probs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for mixture IS estimation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to generations.json")
    parser.add_argument("--label", default="Z4")
    parser.add_argument("--alphas", default="0.6,0.8")
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
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


def logsumexp(a: float, b: float) -> float:
    """Stable log-sum-exp for two scalars."""
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def main() -> None:
    """Load a mixture run and report IS diagnostics."""
    args = parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        raise SystemExit("No results found")

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    if len(alphas) != len(weights):
        raise ValueError("alphas and weights length mismatch")

    weights_is = []
    indicators = []

    for item in results:
        log_p = float(item.get("logprob_sum_p", 0.0))
        log_alpha = item.get("logprob_sum_alpha", {})

        if len(alphas) == 2:
            a0 = str(alphas[0])
            a1 = str(alphas[1])
            log_q0 = float(log_alpha.get(a0))
            log_q1 = float(log_alpha.get(a1))
            log_q_mix = logsumexp(
                math.log(weights[0]) + log_q0,
                math.log(weights[1]) + log_q1,
            )
        else:
            # General case
            logs = []
            for a, w in zip(alphas, weights):
                la = str(a)
                logs.append(math.log(w) + float(log_alpha.get(la)))
            m = max(logs)
            log_q_mix = m + math.log(sum(math.exp(x - m) for x in logs))

        w = math.exp(log_p - log_q_mix)
        z = bool(item.get("detections_final", {}).get(args.label, False))
        weights_is.append(w)
        indicators.append(1.0 if z else 0.0)

    estimate = compute_estimate(weights_is, indicators)
    naive = sum(indicators) / len(indicators)

    w_sum = sum(weights_is)
    w_sq_sum = sum(w * w for w in weights_is)
    ess = (w_sum * w_sum) / w_sq_sum if w_sq_sum > 0 else 0.0
    max_w = max(weights_is) if weights_is else 0.0
    max_w_share = max_w / w_sum if w_sum > 0 else 0.0
    k_hat = estimate_k(weights_is)

    ci = None
    if args.bootstrap and args.bootstrap > 0:
        rng = random.Random(args.seed)
        boot = []
        n = len(weights_is)
        for _ in range(args.bootstrap):
            idx = [rng.randrange(n) for _ in range(n)]
            b_w = [weights_is[i] for i in idx]
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
