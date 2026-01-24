"""Batch discovery run across multiple prompts."""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from typing import Any, Dict

from richerevals.config import load_yaml
from richerevals.modeling import load_model_and_tokenizer
from richerevals.paired_decoding import GenerationConfig, PairedDecoder
from richerevals.promptbank import filter_by_tag, load_promptbank
from richerevals.prompting import format_chat_prompt
from richerevals.q_builders import build_q
from richerevals.run_logger import RunLogger
from richerevals.seed import get_seed, set_seed


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch discovery runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default="configs/discovery_batch.yaml")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run discovery across a batch of prompts."""
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    exp_cfg = load_yaml(args.config)
    cfg = deep_update(deepcopy(base_cfg), exp_cfg)

    if args.model_id:
        cfg.setdefault("model", {})["id"] = args.model_id
    if args.limit is not None:
        cfg.setdefault("batch", {})["prompt_limit"] = args.limit
    if args.samples is not None:
        cfg.setdefault("batch", {})["samples_per_gamma"] = args.samples
    if args.batch_size is not None:
        cfg.setdefault("batch", {})["batch_size"] = args.batch_size

    run_cfg = cfg.get("run", {})
    seed = get_seed(run_cfg.get("seed"))
    set_seed(seed, deterministic=run_cfg.get("deterministic", False))

    runs_dir = cfg.get("logging", {}).get("runs_dir")
    if runs_dir:
        os.environ["RICHEREVALS_RUNS_DIR"] = str(runs_dir)

    logger = RunLogger(run_name=run_cfg.get("name", "discovery_batch"), config=cfg)

    promptbank_path = cfg.get("promptbank", {}).get("path", "data/promptbank_v0.jsonl")
    items = load_promptbank(promptbank_path)

    tag_filter = cfg.get("batch", {}).get("tag_filter")
    if tag_filter:
        items = filter_by_tag(items, tag_filter)

    limit = cfg.get("batch", {}).get("prompt_limit", 10)
    items = items[: int(limit)]

    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("id")
    if not model_id:
        raise ValueError("model.id is required")

    model, tokenizer, device = load_model_and_tokenizer(
        model_id,
        device=model_cfg.get("device"),
        dtype=model_cfg.get("dtype"),
    )
    decoder = PairedDecoder(model, tokenizer)

    sampling_cfg = cfg.get("sampling", {})
    gen_cfg = GenerationConfig(
        max_new_tokens=int(sampling_cfg.get("max_new_tokens", 128)),
        temperature=float(sampling_cfg.get("temperature", 1.0)),
        top_p=float(sampling_cfg.get("top_p", 1.0)),
        top_k=int(sampling_cfg.get("top_k", 0)),
        do_sample=True,
    )

    gamma_values = cfg.get("prompting", {}).get("gamma_values", [1.0])
    q_strategy = cfg.get("prompting", {}).get("q_strategy", "dominant_mode")
    samples_per_gamma = int(cfg.get("batch", {}).get("samples_per_gamma", 5))
    batch_size = int(cfg.get("batch", {}).get("batch_size", 1))

    results = []
    for item in items:
        prompt_p = item.prompt
        prompt_q = build_q(
            prompt_p, item.dominant_mode, item.z_candidates, item.meta, q_strategy
        )
        prompt_p_fmt = format_chat_prompt(tokenizer, prompt_p)
        prompt_q_fmt = format_chat_prompt(tokenizer, prompt_q)
        for gamma in gamma_values:
            remaining = samples_per_gamma
            sample_idx = 0
            while remaining > 0:
                step_batch = min(batch_size, remaining)
                batch = decoder.generate_batch(
                    prompt_p=prompt_p_fmt,
                    prompt_q=prompt_q_fmt,
                    mode="extrapolate",
                    value=float(gamma),
                    config=gen_cfg,
                    batch_size=step_batch,
                )
                for res in batch.results:
                    results.append(
                        {
                            "prompt_id": item.id,
                            "gamma": gamma,
                            "sample_idx": sample_idx,
                            "text": res.text,
                            "tokens": res.tokens,
                            "logprob_sum_mix": float(sum(res.logprobs_mix)),
                        }
                    )
                    sample_idx += 1
                remaining -= step_batch

    logger.log_json(
        "generations",
        {"results": results, "device": device, "prompt_count": len(items)},
    )
    logger.log_text("status", "completed")
    print(f"Discovery batch complete: {logger.run_id}")


if __name__ == "__main__":
    main()
