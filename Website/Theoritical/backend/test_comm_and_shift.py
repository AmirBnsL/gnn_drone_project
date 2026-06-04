"""Tests for comm radius messaging and Z slot-shift gating."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_backend = Path(__file__).resolve().parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

_repo = Path(__file__).resolve().parents[3]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from agents.comm_orchestrator import exchange_messages, swarm_has_shift_proposal
from agents.drone_agent import DroneAgent
from slot_shift_safety import evaluate_shift_eligibility, shift_slots_z

import viz_sim_config as vcfg


def test_comm_radius_blocks_distant_pairs() -> None:
    agents = [DroneAgent(i, vcfg.COMM_RADIUS, 5, 10) for i in range(3)]
    positions = [
        np.array([0.0, 0.0, 1.0]),
        np.array([3.0, 0.0, 1.0]),
        np.array([20.0, 0.0, 1.0]),
    ]
    yaws = [0.0, 0.0, 0.0]
    local_vels = [np.zeros(3), np.zeros(3), np.zeros(3)]
    alerts = [False, False, False]
    stuck = [False, True, False]

    exchange_messages(agents, positions, yaws, local_vels, alerts, stuck, step=0)

    assert agents[0].inbox_shift_proposals
    assert not agents[2].inbox_shift_proposals


def test_swarm_has_shift_proposal() -> None:
    from agents.messages import ShiftProposal

    agents = [DroneAgent(0, vcfg.COMM_RADIUS, 5, 10), DroneAgent(1, vcfg.COMM_RADIUS, 5, 10)]
    agents[0].local_stuck = True
    agents[0].outbox_shift_proposals = [ShiftProposal(0, 0)]
    assert swarm_has_shift_proposal(agents)


def test_z_shift_raises_all_slots() -> None:
    slots = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    shifted = shift_slots_z(slots, delta_z=2.0)
    assert np.allclose(shifted[:, 0], slots[:, 0])
    assert np.allclose(shifted[:, 1], slots[:, 1])
    assert np.allclose(shifted[:, 2], slots[:, 2] + 2.0)


def test_shift_timing_constants() -> None:
    assert vcfg.SHIFT_MIN_STEPS == 20
    assert vcfg.SHIFT_MAX_ALONG_DIST == 3.0


def test_shift_blocked_during_warmup() -> None:
    ok, reason = evaluate_shift_eligibility(
        step=0,
        stuck_flags=[True],
        slot_errors=[1.0],
        obstacles_present=True,
        steps_since_shift=10_000,
        did_shift=False,
    )
    assert not ok
    assert reason == "warmup"

    ok19, reason19 = evaluate_shift_eligibility(
        step=19,
        stuck_flags=[True],
        slot_errors=[1.0],
        obstacles_present=True,
        steps_since_shift=10_000,
        did_shift=False,
    )
    assert not ok19 and reason19 == "warmup"

    ok20, reason20 = evaluate_shift_eligibility(
        step=20,
        stuck_flags=[True],
        slot_errors=[1.0],
        obstacles_present=True,
        steps_since_shift=10_000,
        did_shift=False,
    )
    assert ok20 and reason20 == "ok"


def test_mailbox_cleared_each_exchange() -> None:
    agents = [DroneAgent(0, vcfg.COMM_RADIUS, 5, 10)]
    agents[0].inbox_shift_proposals.append(
        __import__("agents.messages", fromlist=["ShiftProposal"]).ShiftProposal(0, 0)
    )
    exchange_messages(
        agents,
        [np.zeros(3)],
        [0.0],
        [np.zeros(3)],
        [False],
        [False],
        step=1,
    )
    assert len(agents[0].inbox_shift_proposals) == 0


if __name__ == "__main__":
    test_comm_radius_blocks_distant_pairs()
    test_swarm_has_shift_proposal()
    test_z_shift_raises_all_slots()
    test_shift_timing_constants()
    test_shift_blocked_during_warmup()
    test_mailbox_cleared_each_exchange()
    print("OK comm_and_shift")
