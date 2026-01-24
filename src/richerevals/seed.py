"""Reproducible seeding utilities."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set Python, NumPy, and torch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(False)
    except Exception:
        return


def get_seed(seed: Optional[int]) -> int:
    """Return an integer seed, defaulting to 1337."""
    return int(seed) if seed is not None else 1337
