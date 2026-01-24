"""Q-screening for Z amplification (prompt rewrites)."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from richerevals.config import load_yaml
from richerevals.detectors import run_detectors
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
    """Parse CLI arguments for Q-screening."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default="configs/q_screening.yaml")
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def sample_and_score(
    decoder: PairedDecoder,
    prompt_p: str,
    prompt_q: str,
    gen_cfg: GenerationConfig,
    meta: Dict,
    mode: str,
    value: float,
    samples: int,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Sample outputs and compute Z detection rates."""
    detections = []
    counts = {"Z1": 0, "Z2": 0, "Z3": 0, "Z4": 0}

    remaining = samples
    while remaining > 0:
        step_batch = min(batch_size, remaining)
        batch = decoder.generate_batch(
            prompt_p=prompt_p,
            prompt_q=prompt_q,
            mode=mode,
            value=value,
            config=gen_cfg,
            batch_size=step_batch,
        )
        for res in batch.results:
            det = run_detectors(prompt_p, res.text, meta)
            detections.append({k: v.label for k, v in det.items()})
            for key, val in det.items():
                if val.label:
                    counts[key] += 1
        remaining -= step_batch

    rates = {k: counts[k] / float(samples) for k in counts}
    return detections, rates


def main() -> None:
    """Run Q-screening over prompts and log candidate Q results."""
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    exp_cfg = load_yaml(args.config)
    cfg = deep_update(deepcopy(base_cfg), exp_cfg)

    if args.prompt_id:
        cfg.setdefault("prompting", {})["prompt_id"] = args.prompt_id
    if args.model_id:
        cfg.setdefault("model", {})["id"] = args.model_id
    if args.device:
        cfg.setdefault("model", {})["device"] = args.device
    if args.samples is not None:
        cfg.setdefault("screening", {})["samples"] = args.samples
    if args.max_new is not None:
        cfg.setdefault("sampling", {})["max_new_tokens"] = args.max_new
    if args.batch_size is not None:
        cfg.setdefault("sampling", {})["batch_size"] = args.batch_size

    run_cfg = cfg.get("run", {})
    seed = get_seed(run_cfg.get("seed"))
    set_seed(seed, deterministic=run_cfg.get("deterministic", False))

    runs_dir = cfg.get("logging", {}).get("runs_dir")
    if runs_dir:
        os.environ["RICHEREVALS_RUNS_DIR"] = str(runs_dir)

    logger = RunLogger(run_name=run_cfg.get("name", "q_screening"), config=cfg)
    output_path = logger.artifacts_dir / "q_screening.json"

    promptbank_path = cfg.get("promptbank", {}).get("path", "data/promptbank_v0.jsonl")
    items = load_promptbank(promptbank_path)

    prompt_id = cfg.get("prompting", {}).get("prompt_id")
    if prompt_id:
        items = [i for i in items if i.id == prompt_id]

    items = [i for i in items if i.z_candidates]
    if args.limit:
        items = items[: args.limit]

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

    screening_cfg = cfg.get("screening", {})
    samples = int(screening_cfg.get("samples", 30))
    min_relative = float(screening_cfg.get("min_relative", 3.0))
    min_absolute = float(screening_cfg.get("min_absolute", 0.01))
    min_count = int(screening_cfg.get("min_count", 1))
    strategies = list(
        screening_cfg.get("strategies", ["echo", "minimal_rewrite", "auto"])
    )
    batch_size = int(sampling_cfg.get("batch_size", 1))

    outputs = []
    completed_ids = set()
    if output_path.exists():
        try:
            existing = output_path.read_text(encoding="utf-8").strip()
            if existing:
                payload = deepcopy(json.loads(existing))
                outputs = list(payload.get("results", []))
                completed_ids = {
                    item["prompt_id"] for item in outputs if "prompt_id" in item
                }
        except Exception:
            outputs = []
            completed_ids = set()
    for item in items:
        if item.id in completed_ids:
            continue
        prompt_p = item.prompt
        baseline_q = prompt_p
        prompt_p_fmt = format_chat_prompt(tokenizer, prompt_p)
        baseline_q_fmt = format_chat_prompt(tokenizer, baseline_q)
        meta = dict(item.meta)
        meta["z_candidates"] = item.z_candidates

        _, baseline_rates = sample_and_score(
            decoder,
            prompt_p_fmt,
            baseline_q_fmt,
            gen_cfg,
            meta,
            mode="interpolate",
            value=1.0,
            samples=samples,
            batch_size=batch_size,
        )

        chosen = None
        chosen_reason = "no_candidate_met_threshold"
        candidate_results = []

        for strategy in strategies:
            prompt_q = build_q(
                prompt_p,
                item.dominant_mode,
                item.z_candidates,
                item.meta,
                strategy=strategy,
            )
            prompt_q_fmt = format_chat_prompt(tokenizer, prompt_q)
            _, q_rates = sample_and_score(
                decoder,
                prompt_p_fmt,
                prompt_q_fmt,
                gen_cfg,
                meta,
                mode="interpolate",
                value=0.0,
                samples=samples,
                batch_size=batch_size,
            )
            candidate_results.append(
                {
                    "strategy": strategy,
                    "prompt_q": prompt_q,
                    "rates": q_rates,
                }
            )

            if chosen is None:
                for z in item.z_candidates:
                    base = baseline_rates.get(z, 0.0)
                    rate = q_rates.get(z, 0.0)
                    rel_ok = (
                        rate >= base * min_relative
                        if base > 0
                        else rate >= min_absolute
                    )
                    abs_ok = rate - base >= min_absolute
                    count_ok = rate * samples >= min_count
                    if (rel_ok or abs_ok) and count_ok:
                        chosen = strategy
                        chosen_reason = f"meets_threshold_for_{z}"
                        break

        outputs.append(
            {
                "prompt_id": item.id,
                "z_candidates": item.z_candidates,
                "baseline_rates": baseline_rates,
                "candidate_results": candidate_results,
                "chosen_strategy": chosen,
                "chosen_reason": chosen_reason,
            }
        )

        logger.log_json("q_screening", {"device": device, "results": outputs})
    logger.log_text("status", "completed")
    print(f"Q-screening complete: {logger.run_id}")


if __name__ == "__main__":
    main()
