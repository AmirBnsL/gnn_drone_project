"""Ensure gnn_drone_project root is on sys.path (no torch/GNN imports)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def ensure_repo_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / "merged_work").is_dir():
        raise RuntimeError(f"gnn_drone_project root not found (looked at {root})")
    arch = root / "decentralised" / "architectures"
    # Do not add setpoint_prediction3/src here — its inference.py shadows import names
    for p in (root, arch):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    if "local_negotiator" not in sys.modules:
        mod = importlib.import_module("local_negotiator_v2")
        sys.modules["local_negotiator"] = mod

    import viz_sim_config as vcfg
    from merged_work.models_creation.dataset_pipeline import sync_assignment_radii

    sync_assignment_radii(vcfg.COMM_RADIUS, vcfg.SLOT_VISIBILITY_RADIUS)

    return root
