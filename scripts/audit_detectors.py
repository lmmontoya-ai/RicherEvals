"""Heuristic detector audit helper.

Runs a small batch of prompts and logs heuristic detector outputs for manual review.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer

from richerevals.config import load_yaml
from richerevals.detectors import run_detectors
from richerevals.promptbank import load_promptbank
from richerevals.prompting import format_chat_prompt
from richerevals.run_logger import RunLogger
from richerevals.seed import get_seed, set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for detector audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default="configs/estimation.yaml")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model-id", default=None)
    return parser.parse_args()


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def main() -> None:
    """Run a small detector audit and log results."""
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    exp_cfg = load_yaml(args.config)
    cfg = deep_update(base_cfg, exp_cfg)

    if args.model_id:
        cfg.setdefault("model", {})["id"] = args.model_id

    run_cfg = cfg.get("run", {})
    seed = get_seed(run_cfg.get("seed"))
    set_seed(seed, deterministic=run_cfg.get("deterministic", False))

    runs_dir = cfg.get("logging", {}).get("runs_dir")
    if runs_dir:
        os.environ["RICHEREVALS_RUNS_DIR"] = str(runs_dir)

    logger = RunLogger(run_name="detector_audit", config=cfg)

    promptbank_path = cfg.get("promptbank", {}).get("path", "data/promptbank_v0.jsonl")
    items = load_promptbank(promptbank_path)
    z_items = [item for item in items if item.z_candidates]
    z_items = z_items[: args.limit]

    model_id = cfg.get("model", {}).get("id")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = cfg.get("model", {}).get("device") or "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()

    results = []
    for item in z_items:
        prompt_text = format_chat_prompt(tokenizer, item.prompt)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=80, do_sample=True)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        meta = dict(item.meta)
        meta["z_candidates"] = item.z_candidates
        detections = run_detectors(item.prompt, text, meta)
        results.append(
            {
                "prompt_id": item.id,
                "prompt": item.prompt,
                "response": text,
                "detections": {k: v.label for k, v in detections.items()},
            }
        )

    logger.log_json("detector_audit", {"results": results})
    print(f"Detector audit saved to {logger.base_dir}")


if __name__ == "__main__":
    main()
