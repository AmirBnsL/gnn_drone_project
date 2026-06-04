"""Resolve GNN checkpoint paths under Website/Theoritical/resources/."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

_THEORY_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = _THEORY_ROOT / "resources" / "checkpoints"

REQUIRED_FILES = (
    "strict_local_negotiator_best_v1.pt",
    "bertsekas_best_digits.pt",
    "best_gatv2_digits.pth",
    "normalization_stats_digits.pt",
)


def checkpoint_paths() -> Dict[str, Path]:
    return {name: CHECKPOINT_DIR / name for name in REQUIRED_FILES}


def missing_checkpoints() -> List[str]:
    return [name for name in REQUIRED_FILES if not (CHECKPOINT_DIR / name).is_file()]


def ensure_checkpoints() -> Dict[str, Path]:
    missing = missing_checkpoints()
    if missing:
        readme = _THEORY_ROOT / "resources" / "README.md"
        raise FileNotFoundError(
            f"Missing checkpoint(s) in {CHECKPOINT_DIR}: {', '.join(missing)}. "
            f"See {readme}"
        )
    return checkpoint_paths()
