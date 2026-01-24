"""Entry point for discovery experiments (stub)."""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from typing import Any, Dict

from richerevals.config import load_yaml
from richerevals.modeling import load_model_and_tokenizer
from richerevals.paired_decoding import GenerationConfig, PairedDecoder
from richerevals.promptbank import load_promptbank
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
    """Parse CLI arguments for discovery runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Base config YAML.",
    )
    parser.add_argument(
        "--config",
        default="configs/discovery.yaml",
        help="Experiment config YAML.",
    )
    parser.add_argument("--run-name", default=None, help="Override run name.")
    parser.add_argument("--prompt-id", default=None, help="Override prompt id.")
    parser.add_argument("--model-id", default=None, help="Override model id.")
    return parser.parse_args()


def main() -> None:
    """Run a single discovery experiment."""
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    exp_cfg = load_yaml(args.config)
    cfg = deep_update(deepcopy(base_cfg), exp_cfg)

    if args.run_name:
        cfg.setdefault("run", {})["name"] = args.run_name
    if args.prompt_id:
        cfg.setdefault("prompting", {})["prompt_id"] = args.prompt_id
    if args.model_id:
        cfg.setdefault("model", {})["id"] = args.model_id

    run_cfg = cfg.get("run", {})
    seed = get_seed(run_cfg.get("seed"))
    set_seed(seed, deterministic=run_cfg.get("deterministic", False))

    runs_dir = cfg.get("logging", {}).get("runs_dir")
    if runs_dir:
        os.environ["RICHEREVALS_RUNS_DIR"] = str(runs_dir)

    logger = RunLogger(run_name=run_cfg.get("name", "run"), config=cfg)
    logger.log_text("status", "initialized")

    promptbank_path = cfg.get("promptbank", {}).get("path", "data/promptbank_v0.jsonl")
    items = load_promptbank(promptbank_path)
    prompt_id = cfg.get("prompting", {}).get("prompt_id") or items[0].id
    item = next(i for i in items if i.id == prompt_id)

    q_strategy = cfg.get("prompting", {}).get("q_strategy", "dominant_mode")
    prompt_p = item.prompt
    prompt_q = build_q(
        prompt_p, item.dominant_mode, item.z_candidates, item.meta, q_strategy
    )

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

    prompt_p_fmt = format_chat_prompt(tokenizer, prompt_p)
    prompt_q_fmt = format_chat_prompt(tokenizer, prompt_q)

    sampling_cfg = cfg.get("sampling", {})
    gen_cfg = GenerationConfig(
        max_new_tokens=int(sampling_cfg.get("max_new_tokens", 128)),
        temperature=float(sampling_cfg.get("temperature", 1.0)),
        top_p=float(sampling_cfg.get("top_p", 1.0)),
        top_k=int(sampling_cfg.get("top_k", 0)),
        do_sample=True,
    )

    results = []
    for gamma in cfg.get("prompting", {}).get("gamma_values", [1.0]):
        res = decoder.generate(
            prompt_p=prompt_p_fmt,
            prompt_q=prompt_q_fmt,
            mode="extrapolate",
            value=float(gamma),
            config=gen_cfg,
        )
        results.append(
            {
                "prompt_id": item.id,
                "mode": "extrapolate",
                "gamma": gamma,
                "text": res.text,
                "tokens": res.tokens,
                "logprob_sum_mix": float(sum(res.logprobs_mix)),
            }
        )

    logger.log_json(
        "generations",
        {
            "prompt_id": item.id,
            "prompt_p": prompt_p,
            "prompt_q": prompt_q,
            "results": results,
        },
    )
    logger.log_text("status", "completed")
    print(f"Initialized run: {logger.run_id}")
    print(f"Discovery complete for prompt {item.id} on device {device}.")


if __name__ == "__main__":
    main()
