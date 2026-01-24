"""Entry point for estimation experiments (stub)."""

from __future__ import annotations

import argparse
import os
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
    """Parse CLI arguments for estimation runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Base config YAML.",
    )
    parser.add_argument(
        "--config",
        default="configs/estimation.yaml",
        help="Experiment config YAML.",
    )
    parser.add_argument("--run-name", default=None, help="Override run name.")
    parser.add_argument("--prompt-id", default=None, help="Override prompt id.")
    parser.add_argument("--model-id", default=None, help="Override model id.")
    parser.add_argument(
        "--samples", type=int, default=None, help="Override samples per alpha."
    )
    parser.add_argument(
        "--max-new", type=int, default=None, help="Override max_new_tokens."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size."
    )
    parser.add_argument(
        "--alpha-values",
        default=None,
        help="Comma-separated alpha values to override config.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Write checkpoint after this many samples.",
    )
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="Verify heuristic positives with LLM judge.",
    )
    return parser.parse_args()


def main() -> None:
    """Run interpolation-based estimation and log detections."""
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
    if args.samples is not None:
        cfg.setdefault("sampling", {})["samples_per_alpha"] = args.samples
    if args.max_new is not None:
        cfg.setdefault("sampling", {})["max_new_tokens"] = args.max_new
    if args.batch_size is not None:
        cfg.setdefault("sampling", {})["batch_size"] = args.batch_size
    if args.alpha_values is not None:
        parsed = [float(x.strip()) for x in args.alpha_values.split(",") if x.strip()]
        cfg.setdefault("prompting", {})["alpha_values"] = parsed

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

    q_strategy = cfg.get("prompting", {}).get("q_strategy", "auto")
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
    samples_per_alpha = int(sampling_cfg.get("samples_per_alpha", 1))
    meta = dict(item.meta)
    meta["z_candidates"] = item.z_candidates

    use_judge = bool(args.use_judge)

    checkpoint_every = args.checkpoint_every
    samples_seen = 0

    alpha_values = cfg.get("prompting", {}).get("alpha_values", [1.0])
    for alpha in alpha_values:
        batch_size = int(sampling_cfg.get("batch_size", 1))
        remaining = samples_per_alpha
        sample_idx = 0
        while remaining > 0:
            step_batch = min(batch_size, remaining)
            batch = decoder.generate_batch(
                prompt_p=prompt_p_fmt,
                prompt_q=prompt_q_fmt,
                mode="interpolate",
                value=float(alpha),
                config=gen_cfg,
                batch_size=step_batch,
            )
            for res in batch.results:
                detections = run_detectors(prompt_p, res.text, meta)
                det_labels = {k: v.label for k, v in detections.items()}
                judge_labels = {}
                judge_errors = {}

                if use_judge:
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
                        final_labels[label] = bool(det_labels.get(label)) and bool(
                            jflag
                        )

                results.append(
                    {
                        "prompt_id": item.id,
                        "mode": "interpolate",
                        "alpha": alpha,
                        "sample_idx": sample_idx,
                        "text": res.text,
                        "tokens": res.tokens,
                        "logprob_sum_p": float(sum(res.logprobs_p)),
                        "logprob_sum_q": float(sum(res.logprobs_q)),
                        "logprob_sum_mix": float(sum(res.logprobs_mix)),
                        "detections": det_labels,
                        "detections_judge": judge_labels,
                        "detections_final": final_labels,
                        "judge_errors": judge_errors,
                    }
                )
                sample_idx += 1
                samples_seen += 1
                if checkpoint_every and samples_seen % checkpoint_every == 0:
                    logger.log_json(
                        "generations",
                        {
                            "prompt_id": item.id,
                            "prompt_p": prompt_p,
                            "prompt_q": prompt_q,
                            "alpha_values": alpha_values,
                            "results": results,
                        },
                    )
            remaining -= step_batch

    logger.log_json(
        "generations",
        {
            "prompt_id": item.id,
            "prompt_p": prompt_p,
            "prompt_q": prompt_q,
            "alpha_values": alpha_values,
            "results": results,
        },
    )
    logger.log_text("status", "completed")
    print(f"Initialized run: {logger.run_id}")
    print(f"Estimation complete for prompt {item.id} on device {device}.")


if __name__ == "__main__":
    main()
