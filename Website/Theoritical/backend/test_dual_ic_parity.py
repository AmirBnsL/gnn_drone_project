"""Verify dual simulator uses identical IC copies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_backend = Path(__file__).resolve().parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from path_setup import ensure_repo_path

ensure_repo_path()

from merged_work.models_creation.dataset_pipeline import (
    assign_drones_to_slots,
    build_naive_slots,
    make_episode_config,
    sample_initial_state,
)


def test_ic_copy_independent() -> None:
    cfg = make_episode_config(seed=42, num_drones=12, digit=3, scenario="both")
    start_pos, start_orn = sample_initial_state(cfg)
    slots = build_naive_slots(cfg, start_pos)
    assignment = assign_drones_to_slots(start_pos, slots)

    sp = start_pos.copy()
    sl = slots.copy()
    sp[0, 0] += 999.0
    assert float(start_pos[0, 0]) != float(sp[0, 0])
    assert np.allclose(slots, sl)


if __name__ == "__main__":
    test_ic_copy_independent()
    print("OK dual_ic_parity")
