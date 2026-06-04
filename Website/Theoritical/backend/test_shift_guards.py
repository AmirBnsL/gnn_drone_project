"""Shift eligibility and safety clamp smoke tests (no PyFlyt)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_backend = Path(__file__).resolve().parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from slot_shift_safety import evaluate_shift_eligibility, safety_clamp_setpoint, shift_slots_z

import viz_sim_config as vcfg


def test_safety_clamp_pushes_away() -> None:
    pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    proposed = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    obs = np.array([[0.5, 0.0, 1.0]], dtype=np.float32)
    out = safety_clamp_setpoint(pos, proposed, obs)
    assert out[0] <= proposed[0] + 1e-3


def test_shift_once_blocks_second() -> None:
    ok, _ = evaluate_shift_eligibility(
        step=100,
        stuck_flags=[True],
        slot_errors=[1.0],
        obstacles_present=True,
        steps_since_shift=100,
        did_shift=True,
    )
    assert not ok


if __name__ == "__main__":
    test_safety_clamp_pushes_away()
    test_shift_once_blocks_second()
    dz = shift_slots_z(np.array([[0, 0, 1]], dtype=np.float32))[0, 2]
    assert abs(dz - 3.0) < 1e-4
    print("OK shift_guards")
