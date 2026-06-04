"""Tests for named simulation save/load."""

from __future__ import annotations

import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent
_repo = _backend.parents[2]
for p in (_repo, _backend):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from session import SimulationSession
from sim_persistence import SAVES_DIR, list_saves, load_run, save_run


def test_save_load_roundtrip():
    session = SimulationSession()
    session.apply_config(6, "path")
    session.apply_formation(3)
    tiny = {
        "frames": [
            {
                "center": [0.0, 0.0],
                "drones": [],
                "slots": [],
                "assignment": [],
                "obstacles": [],
            }
        ],
        "dt": 0.01,
        "converged": False,
        "stopped_reason": "max_steps",
        "final_max_slot_error": 2.5,
    }
    session.trajectory_central = dict(tiny)
    session.trajectory_decentral = dict(tiny)
    session.obstacles_visible = True

    name = "_test_roundtrip_run"
    try:
        save_run(name, session)
        assert name in list_saves()

        fresh = SimulationSession()
        load_run(name, fresh)
        assert fresh.num_drones == 6
        assert fresh.scenario == "path"
        assert fresh.digit == 3
        assert fresh.trajectory_central is not None
        assert fresh.trajectory_decentral is not None
        assert fresh.trajectory_central["final_max_slot_error"] == 2.5
    finally:
        p = SAVES_DIR / f"{name}.json"
        if p.is_file():
            p.unlink()


def _run_all() -> None:
    test_save_load_roundtrip()
    print("test_sim_persistence: all passed")


if __name__ == "__main__":
    _run_all()
