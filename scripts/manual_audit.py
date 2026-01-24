"""Sample judge-positive cases for manual audit."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for manual audit sampling."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to generations.json")
    parser.add_argument("--label", default="Z4")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default="manual_audit.json")
    return parser.parse_args()


def main() -> None:
    """Sample judge-positive cases for manual review."""
    args = parse_args()
    rng = random.Random(args.seed)

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = data.get("results", [])
    positives = [
        item
        for item in results
        if item.get("detections_final", {}).get(args.label, False)
    ]

    rng.shuffle(positives)
    sample = positives[: args.n]

    output = {
        "input": args.input,
        "label": args.label,
        "n_requested": args.n,
        "n_available": len(positives),
        "samples": [
            {
                "prompt_id": item.get("prompt_id"),
                "alpha": item.get("alpha"),
                "sample_idx": item.get("sample_idx"),
                "text": item.get("text"),
            }
            for item in sample
        ],
    }

    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(sample)} samples to {args.output}")


if __name__ == "__main__":
    main()
