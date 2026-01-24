"""Paired decoding primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    """Sampling configuration for paired decoding."""

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    eos_token_id: Optional[int] = None


@dataclass
class PairedResult:
    """Single sampled sequence with logprob traces."""

    tokens: List[int]
    text: str
    logprobs_p: List[float]
    logprobs_q: List[float]
    logprobs_mix: List[float]
    logprob_sum_alpha: Dict[float, float] = field(default_factory=dict)


@dataclass
class PairedBatchResult:
    """Batch of paired decoding results."""

    results: List[PairedResult]


def logit_interpolation(logits_p: torch.Tensor, logits_q: torch.Tensor, alpha: float):
    """Interpolate between two logit tensors."""
    return alpha * logits_p + (1.0 - alpha) * logits_q


def logit_extrapolation(logits_p: torch.Tensor, logits_q: torch.Tensor, gamma: float):
    """Extrapolate logits away from the Q distribution."""
    return logits_p + gamma * (logits_p - logits_q)


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if temperature == 1.0:
        return logits
    return logits / temperature


def _filter_top_k_top_p(
    logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0
) -> torch.Tensor:
    """Apply top-k and/or top-p filtering in logit space."""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = probs.cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


class PairedDecoder:
    """Paired decoding loop with KV-cache sync and logprob tracking."""

    def __init__(self, model, tokenizer):
        """Initialize with a HF model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    def _encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text and move tensors to the model device."""
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        return input_ids, attention_mask

    @torch.no_grad()
    def _forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], past=None
    ) -> Tuple[torch.Tensor, Tuple]:
        """Run a single forward step and return next-token logits."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past,
        )
        logits = outputs.logits[:, -1, :]
        return logits, outputs.past_key_values

    def generate(
        self,
        prompt_p: str,
        prompt_q: str,
        mode: str,
        value: float,
        config: GenerationConfig,
        log_alpha_values: Optional[List[float]] = None,
    ) -> PairedResult:
        """Generate a single sequence from a paired P/Q sampler."""
        if mode not in {"interpolate", "extrapolate"}:
            raise ValueError("mode must be 'interpolate' or 'extrapolate'")

        eos_id = (
            config.eos_token_id
            if config.eos_token_id is not None
            else self.tokenizer.eos_token_id
        )

        input_ids_p, attention_mask_p = self._encode(prompt_p)
        input_ids_q, attention_mask_q = self._encode(prompt_q)

        logits_p, past_p = self._forward(input_ids_p, attention_mask_p)
        logits_q, past_q = self._forward(input_ids_q, attention_mask_q)

        tokens: List[int] = []
        logprobs_p: List[float] = []
        logprobs_q: List[float] = []
        logprobs_mix: List[float] = []

        logprob_sum_alpha = {a: 0.0 for a in (log_alpha_values or [])}

        for _ in range(config.max_new_tokens):
            if mode == "interpolate":
                logits_mix = logit_interpolation(logits_p, logits_q, value)
            else:
                logits_mix = logit_extrapolation(logits_p, logits_q, value)

            logits_mix = _apply_temperature(logits_mix, config.temperature)
            logits_mix = _filter_top_k_top_p(
                logits_mix, top_k=config.top_k, top_p=config.top_p
            )

            if config.do_sample:
                probs = torch.softmax(logits_mix, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits_mix, dim=-1, keepdim=True)

            token_id = int(next_token.item())
            tokens.append(token_id)

            logprob_mix = F.log_softmax(logits_mix, dim=-1).gather(-1, next_token)
            logprobs_mix.append(float(logprob_mix.item()))

            logits_p_scaled = _apply_temperature(logits_p, config.temperature)
            logits_q_scaled = _apply_temperature(logits_q, config.temperature)
            logprob_p = F.log_softmax(logits_p_scaled, dim=-1).gather(-1, next_token)
            logprob_q = F.log_softmax(logits_q_scaled, dim=-1).gather(-1, next_token)
            logprobs_p.append(float(logprob_p.item()))
            logprobs_q.append(float(logprob_q.item()))

            if log_alpha_values:
                if mode != "interpolate":
                    raise ValueError(
                        "log_alpha_values only supported for interpolate mode"
                    )
                for alpha in log_alpha_values:
                    logits_a = logit_interpolation(logits_p, logits_q, alpha)
                    logits_a = _apply_temperature(logits_a, config.temperature)
                    logits_a = _filter_top_k_top_p(
                        logits_a, top_k=config.top_k, top_p=config.top_p
                    )
                    logprob_a = F.log_softmax(logits_a, dim=-1).gather(-1, next_token)
                    logprob_sum_alpha[alpha] += float(logprob_a.item())

            if eos_id is not None and token_id == eos_id:
                break

            next_token_p = next_token.to(self.device)
            next_token_q = next_token.to(self.device)
            attention_mask_p = torch.cat(
                [attention_mask_p, torch.ones_like(next_token_p)], dim=1
            )
            attention_mask_q = torch.cat(
                [attention_mask_q, torch.ones_like(next_token_q)], dim=1
            )
            logits_p, past_p = self._forward(next_token_p, attention_mask_p, past_p)
            logits_q, past_q = self._forward(next_token_q, attention_mask_q, past_q)

        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return PairedResult(
            tokens, text, logprobs_p, logprobs_q, logprobs_mix, logprob_sum_alpha
        )

    def generate_batch(
        self,
        prompt_p: str,
        prompt_q: str,
        mode: str,
        value: float,
        config: GenerationConfig,
        batch_size: int,
        log_alpha_values: Optional[List[float]] = None,
    ) -> PairedBatchResult:
        """Generate a batch of sequences from the paired sampler."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_size == 1:
            return PairedBatchResult(
                [
                    self.generate(
                        prompt_p, prompt_q, mode, value, config, log_alpha_values
                    )
                ]
            )

        if mode not in {"interpolate", "extrapolate"}:
            raise ValueError("mode must be 'interpolate' or 'extrapolate'")

        eos_id = (
            config.eos_token_id
            if config.eos_token_id is not None
            else self.tokenizer.eos_token_id
        )

        input_ids_p, attention_mask_p = self._encode(prompt_p)
        input_ids_q, attention_mask_q = self._encode(prompt_q)

        input_ids_p = input_ids_p.repeat(batch_size, 1)
        input_ids_q = input_ids_q.repeat(batch_size, 1)
        attention_mask_p = attention_mask_p.repeat(batch_size, 1)
        attention_mask_q = attention_mask_q.repeat(batch_size, 1)

        logits_p, past_p = self._forward(input_ids_p, attention_mask_p)
        logits_q, past_q = self._forward(input_ids_q, attention_mask_q)

        tokens_batch: List[List[int]] = [[] for _ in range(batch_size)]
        logprobs_p_batch: List[List[float]] = [[] for _ in range(batch_size)]
        logprobs_q_batch: List[List[float]] = [[] for _ in range(batch_size)]
        logprobs_mix_batch: List[List[float]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        logprob_sum_alpha = [
            {a: 0.0 for a in (log_alpha_values or [])} for _ in range(batch_size)
        ]

        for _ in range(config.max_new_tokens):
            if mode == "interpolate":
                logits_mix = logit_interpolation(logits_p, logits_q, value)
            else:
                logits_mix = logit_extrapolation(logits_p, logits_q, value)

            logits_mix = _apply_temperature(logits_mix, config.temperature)
            logits_mix = _filter_top_k_top_p(
                logits_mix, top_k=config.top_k, top_p=config.top_p
            )

            if eos_id is not None and finished.any():
                eos_mask = torch.zeros_like(logits_mix)
                eos_mask[:, :] = float("-inf")
                eos_mask[:, eos_id] = 0.0
                logits_mix = torch.where(finished.unsqueeze(-1), eos_mask, logits_mix)

            if config.do_sample:
                probs = torch.softmax(logits_mix, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits_mix, dim=-1, keepdim=True)

            token_ids = next_token.squeeze(-1).tolist()
            for i, tid in enumerate(token_ids):
                tokens_batch[i].append(int(tid))

            logprob_mix = F.log_softmax(logits_mix, dim=-1).gather(-1, next_token)

            logits_p_scaled = _apply_temperature(logits_p, config.temperature)
            logits_q_scaled = _apply_temperature(logits_q, config.temperature)
            logprob_p = F.log_softmax(logits_p_scaled, dim=-1).gather(-1, next_token)
            logprob_q = F.log_softmax(logits_q_scaled, dim=-1).gather(-1, next_token)

            for i in range(batch_size):
                logprobs_mix_batch[i].append(float(logprob_mix[i].item()))
                logprobs_p_batch[i].append(float(logprob_p[i].item()))
                logprobs_q_batch[i].append(float(logprob_q[i].item()))

            if log_alpha_values:
                if mode != "interpolate":
                    raise ValueError(
                        "log_alpha_values only supported for interpolate mode"
                    )
                for alpha in log_alpha_values:
                    logits_a = logit_interpolation(logits_p, logits_q, alpha)
                    logits_a = _apply_temperature(logits_a, config.temperature)
                    logits_a = _filter_top_k_top_p(
                        logits_a, top_k=config.top_k, top_p=config.top_p
                    )
                    logprob_a = F.log_softmax(logits_a, dim=-1).gather(-1, next_token)
                    for i in range(batch_size):
                        logprob_sum_alpha[i][alpha] += float(logprob_a[i].item())

            if eos_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_id)
                if bool(finished.all()):
                    break

            attention_mask_p = torch.cat(
                [attention_mask_p, torch.ones_like(next_token)], dim=1
            )
            attention_mask_q = torch.cat(
                [attention_mask_q, torch.ones_like(next_token)], dim=1
            )
            logits_p, past_p = self._forward(next_token, attention_mask_p, past_p)
            logits_q, past_q = self._forward(next_token, attention_mask_q, past_q)

        results: List[PairedResult] = []
        for i in range(batch_size):
            text = self.tokenizer.decode(tokens_batch[i], skip_special_tokens=True)
            results.append(
                PairedResult(
                    tokens_batch[i],
                    text,
                    logprobs_p_batch[i],
                    logprobs_q_batch[i],
                    logprobs_mix_batch[i],
                    logprob_sum_alpha[i],
                )
            )

        return PairedBatchResult(results)
