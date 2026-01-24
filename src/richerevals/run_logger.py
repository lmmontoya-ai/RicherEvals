"""Simple local experiment logger."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import save_json


def _utc_timestamp() -> str:
    """Return a UTC timestamp string for run IDs."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _git_hash() -> str | None:
    """Return the current git commit hash, if available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    """Return the installed package version, if available."""
    try:
        import importlib.metadata as metadata

        return metadata.version(name)
    except Exception:
        return None


def get_runs_dir() -> Path:
    """Resolve the base directory for run artifacts."""
    return Path(os.environ.get("RICHEREVALS_RUNS_DIR", "runs")).resolve()


def env_info() -> Dict[str, Any]:
    """Collect environment and dependency metadata for reproducibility."""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_hash": _git_hash(),
        "transformers": _pkg_version("transformers"),
        "accelerate": _pkg_version("accelerate"),
        "numpy": _pkg_version("numpy"),
        "torch": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        has_mps_backend = getattr(torch.backends, "mps", None) is not None
        info["mps_available"] = has_mps_backend and torch.backends.mps.is_available()
    except Exception:
        info["torch"] = None
    return info


class RunLogger:
    """Minimal run logger that writes config, env, and artifacts to disk."""

    def __init__(
        self,
        run_name: Optional[str] = None,
        base_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a run directory and write config + environment."""
        stamp = _utc_timestamp()
        run_name = run_name or "run"
        self.run_id = f"{stamp}_{run_name}"
        self.base_dir = (base_dir or get_runs_dir()) / self.run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.base_dir / "metrics.jsonl"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        if config is not None:
            save_json(config, self.base_dir / "config.json")
        save_json(env_info(), self.base_dir / "env.json")

    def log_metrics(self, step: int, **metrics: float) -> None:
        """Append a metrics record to metrics.jsonl."""
        record = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")

    def log_text(self, name: str, text: str) -> Path:
        """Write a text artifact under the run's artifacts directory."""
        path = self.artifacts_dir / f"{name}.txt"
        with path.open("w", encoding="utf-8") as f:
            f.write(text)
        return path

    def log_json(self, name: str, payload: Dict[str, Any]) -> Path:
        """Write a JSON artifact under the run's artifacts directory."""
        path = self.artifacts_dir / f"{name}.json"
        save_json(payload, path)
        return path
