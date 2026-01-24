"""Mixture proposal sampling for interpolation with multi-alpha logprob tracking."""

from __future__ import annotations

import argparse
import os
import random
from copy import deepcopy
from typing import Any, Dict

from richerevals.config import load_yaml
from richerevals.detectors import run_detectors
from richerevals.judge import JudgeResult, judge_response
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
    """Parse CLI arguments for mixture estimation runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default="configs/estimation.yaml")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--alphas", default="0.6,0.8")
    parser.add_argument("--use-judge", action="store_true")
    parser.add_argument("--run-name", default="interp_mixture")
    return parser.parse_args()


def main() -> None:
    """Run mixture proposal sampling with multi-alpha logprob tracking."""
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    exp_cfg = load_yaml(args.config)
    cfg = deep_update(deepcopy(base_cfg), exp_cfg)

    cfg.setdefault("model", {})["id"] = args.model_id
    cfg.setdefault("prompting", {})["prompt_id"] = args.prompt_id
    cfg.setdefault("sampling", {})["max_new_tokens"] = args.max_new
    cfg.setdefault("sampling", {})["batch_size"] = args.batch_size

    run_cfg = cfg.get("run", {})
    run_cfg["name"] = args.run_name
    seed = get_seed(run_cfg.get("seed"))
    set_seed(seed, deterministic=run_cfg.get("deterministic", False))

    runs_dir = cfg.get("logging", {}).get("runs_dir")
    if runs_dir:
        os.environ["RICHEREVALS_RUNS_DIR"] = str(runs_dir)

    logger = RunLogger(run_name=run_cfg.get("name", "interp_mixture"), config=cfg)

    items = load_promptbank(
        cfg.get("promptbank", {}).get("path", "data/promptbank_v0.jsonl")
    )
    item = next(i for i in items if i.id == args.prompt_id)

    q_strategy = cfg.get("prompting", {}).get("q_strategy", "auto")
    prompt_p = item.prompt
    prompt_q = build_q(
        prompt_p, item.dominant_mode, item.z_candidates, item.meta, q_strategy
    )

    model, tokenizer, device = load_model_and_tokenizer(
        args.model_id,
        device=cfg.get("model", {}).get("device"),
        dtype=cfg.get("model", {}).get("dtype"),
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

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if len(alphas) < 2:
        raise ValueError("Provide at least two alphas for mixture")

    meta = dict(item.meta)
    meta["z_candidates"] = item.z_candidates

    results = []
    remaining = args.samples
    batch_size = args.batch_size
    rng = random.Random(seed)

    while remaining > 0:
        step_batch = min(batch_size, remaining)
        chosen = [rng.choice(alphas) for _ in range(step_batch)]
        alpha_used = chosen[0]
        # For simplicity, generate one alpha at a time per batch.
        batch = decoder.generate_batch(
            prompt_p=prompt_p_fmt,
            prompt_q=prompt_q_fmt,
            mode="interpolate",
            value=float(alpha_used),
            config=gen_cfg,
            batch_size=step_batch,
            log_alpha_values=alphas,
        )

        for res in batch.results:
            detections = run_detectors(prompt_p, res.text, meta)
            det_labels = {k: v.label for k, v in detections.items()}
            judge_labels = {}
            judge_errors = {}
            if args.use_judge:
                for label, flag in det_labels.items():
                    if not flag:
                        continue
                    try:
                        judge_result: JudgeResult = judge_response(
                            prompt_p, res.text, label
                        )
                        judge_labels[label] = judge_result.label
                    except Exception as exc:
                        judge_errors[label] = str(exc)

            final_labels = det_labels.copy()
            if judge_labels:
                for label, jflag in judge_labels.items():
                    final_labels[label] = bool(det_labels.get(label)) and bool(jflag)

            results.append(
                {
                    "prompt_id": item.id,
                    "mode": "interpolate",
                    "alpha": alpha_used,
                    "text": res.text,
                    "tokens": res.tokens,
                    "logprob_sum_p": float(sum(res.logprobs_p)),
                    "logprob_sum_mix": float(sum(res.logprobs_mix)),
                    "logprob_sum_alpha": {
                        str(k): v for k, v in res.logprob_sum_alpha.items()
                    },
                    "detections": det_labels,
                    "detections_judge": judge_labels,
                    "detections_final": final_labels,
                    "judge_errors": judge_errors,
                }
            )

        remaining -= step_batch

    logger.log_json(
        "generations",
        {
            "prompt_id": item.id,
            "prompt_p": prompt_p,
            "prompt_q": prompt_q,
            "alphas": alphas,
            "results": results,
        },
    )
    logger.log_text("status", "completed")
    print(f"Mixture run complete: {logger.run_id}")


if __name__ == "__main__":
    main()
