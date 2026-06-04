"""
Central path + import shims for notebooks and scripts.

Call `setup_project_paths()` once at notebook start (repo root = gnn_drone_project).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def repo_root() -> Path:
    """Return gnn_drone_project directory (parent of merged_work)."""
    here = Path(__file__).resolve().parent
    return here.parent.parent


def setup_project_paths() -> Path:
    """
    Add repo root and decentralised/architectures to sys.path; register local_negotiator shim.
    """
    root = repo_root()
    arch = root / "decentralised" / "architectures"
    setpoint_src = root / "gnn" / "setpoint_prediction3" / "src"

    for p in (root, arch, setpoint_src):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    if "local_negotiator" not in sys.modules:
        mod = importlib.import_module("local_negotiator_v2")
        sys.modules["local_negotiator"] = mod

    from merged_work.models_creation.dataset_pipeline import sync_assignment_radii

    sync_assignment_radii()

    return root
