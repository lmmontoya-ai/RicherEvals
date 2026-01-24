"""Model loading utilities."""

from __future__ import annotations

from typing import Optional


def resolve_device(preferred: Optional[str] = None) -> str:
    """Resolve the best available device (cuda/mps/cpu)."""
    try:
        import torch

        if preferred:
            return preferred
        if torch.cuda.is_available():
            return "cuda"
        has_mps_backend = getattr(torch.backends, "mps", None) is not None
        if has_mps_backend and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return preferred or "cpu"
    return "cpu"


def _dtype_from_str(name: Optional[str]):
    """Map a string name to a torch dtype, if available."""
    if name is None:
        return None
    name = name.lower()
    try:
        import torch

        if name in {"fp16", "float16"}:
            return torch.float16
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if name in {"fp32", "float32"}:
            return torch.float32
    except Exception:
        return None
    return None


def load_model_and_tokenizer(
    model_id: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
):
    """Load a causal LM and tokenizer on the requested device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device)
    dtype = _dtype_from_str(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if dtype is not None:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()

    return model, tokenizer, device
