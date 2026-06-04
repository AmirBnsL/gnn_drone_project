"""Tests for run-until-converged step cap helper."""

from __future__ import annotations

import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

import viz_sim_config as vcfg


def test_simulation_step_cap_default():
    assert vcfg.RUN_UNTIL_CONVERGED is True
    assert vcfg.MAX_STEPS_CONVERGED == 6000
    assert vcfg.simulation_step_cap(None) == 6000


def test_simulation_step_cap_override():
    assert vcfg.simulation_step_cap(10) == 10


def _run_all() -> None:
    test_simulation_step_cap_default()
    test_simulation_step_cap_override()
    print("test_simulation_step_cap: all passed")


if __name__ == "__main__":
    _run_all()
