"""Smoke test for paired decoding endpoints."""

from __future__ import annotations

import argparse
import sys

from richerevals.modeling import load_model_and_tokenizer
from richerevals.paired_decoding import GenerationConfig, PairedDecoder


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the smoke test."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="hf-internal-testing/tiny-random-gpt2",
        help="HF model id for smoke test.",
    )
    parser.add_argument("--device", default="mps", help="Device: mps/cuda/cpu")
    parser.add_argument("--max-new", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    """Run a paired-decoding endpoint check."""
    args = parse_args()
    model, tokenizer, device = load_model_and_tokenizer(
        args.model, device=args.device, dtype=None
    )
    decoder = PairedDecoder(model, tokenizer)

    config = GenerationConfig(
        max_new_tokens=args.max_new,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        do_sample=False,
    )

    prompt_p = "Hello world"
    prompt_q = "Hello world"

    res_p = decoder.generate(
        prompt_p, prompt_q, mode="interpolate", value=1.0, config=config
    )
    res_q = decoder.generate(
        prompt_p, prompt_q, mode="interpolate", value=0.0, config=config
    )

    max_diff_p = max(abs(a - b) for a, b in zip(res_p.logprobs_mix, res_p.logprobs_p))
    max_diff_q = max(abs(a - b) for a, b in zip(res_q.logprobs_mix, res_q.logprobs_q))

    print(f"device={device}")
    print(f"alpha=1.0 max |mix - p| = {max_diff_p:.6g}")
    print(f"alpha=0.0 max |mix - q| = {max_diff_q:.6g}")

    if max_diff_p > 1e-5 or max_diff_q > 1e-5:
        print("Smoke test failed: endpoint mismatch")
        return 1

    # Extrapolation should run without errors.
    _ = decoder.generate(
        prompt_p, prompt_q, mode="extrapolate", value=1.0, config=config
    )
    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
