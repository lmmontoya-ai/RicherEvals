"""Prompt formatting utilities."""

from __future__ import annotations

from typing import Dict, List


def format_chat_prompt(tokenizer, prompt: str) -> str:
    """Apply chat template if available; otherwise return the raw prompt."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return prompt
    return prompt
