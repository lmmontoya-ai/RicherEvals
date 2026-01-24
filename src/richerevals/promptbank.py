"""PromptBank loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class PromptItem:
    """Structured prompt metadata entry."""

    id: str
    prompt: str
    domain: str
    tags: List[str]
    dominant_mode: str
    z_candidates: List[str]
    meta: Dict


def load_promptbank(path: str | Path) -> List[PromptItem]:
    """Load a JSONL promptbank into PromptItem objects."""
    items: List[PromptItem] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(
                PromptItem(
                    id=data["id"],
                    prompt=data["prompt"],
                    domain=data.get("domain", ""),
                    tags=list(data.get("tags", [])),
                    dominant_mode=data.get("dominant_mode", ""),
                    z_candidates=list(data.get("z_candidates", [])),
                    meta=dict(data.get("meta", {})),
                )
            )
    return items


def filter_by_tag(items: Iterable[PromptItem], tag: str) -> List[PromptItem]:
    """Filter prompt items by tag."""
    return [item for item in items if tag in item.tags]
