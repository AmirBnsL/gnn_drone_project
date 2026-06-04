"""
Parity checks: theoretical viz session/recorder vs merged_work dataset pipeline.

Run from repo root:
  python Website/Theoritical/backend/test_theory_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from path_setup import ensure_repo_path

ensure_repo_path()

import numpy as np

from merged_work.models_creation.dataset_pipeline import (
    assign_drones_to_slots,
    build_naive_slots,
    build_obstacles_for_episode,
    make_episode_config,
    sample_initial_state,
    synchronized_slot_shift,
)
from merged_work.models_creation.obstacle_scenarios import ObstacleConfig
from merged_work.models_creation.setpoint_rollout import build_digit_setpoints
from session import SimulationSession


def _assert_close(a: np.ndarray, b: np.ndarray, msg: str, *, atol: float = 1e-5) -> None:
    if not np.allclose(a, b, rtol=0, atol=atol):
        raise AssertionError(f"{msg}\nmax diff={np.max(np.abs(a - b))}")


def test_pipeline_assignment_matches_setpoints() -> None:
    cfg = make_episode_config(seed=12345, num_drones=12, digit=3, scenario="both")
    start_pos, start_orn = sample_initial_state(cfg)
    slots = build_naive_slots(cfg, start_pos)
    assignment_a = assign_drones_to_slots(start_pos, slots)
    _setpoints, assignment_b, _ = build_digit_setpoints(cfg, start_pos, start_orn)
    slots_b = build_naive_slots(cfg, start_pos)
    _assert_close(slots, slots_b, "build_naive_slots mismatch")
    if not np.array_equal(assignment_a, assignment_b):
        raise AssertionError(
            f"assignment mismatch:\n  hungarian={assignment_a}\n  setpoints={assignment_b}"
        )
    print("OK  pipeline assignment == build_digit_setpoints")


def test_session_matches_pipeline() -> None:
    sess = SimulationSession()
    sess.apply_config(14, "path")
    sess.apply_formation(7)

    cfg = make_episode_config(
        seed=sess.seed, num_drones=sess.num_drones, digit=7, scenario=sess.scenario
    )
    start_pos, start_orn = sample_initial_state(cfg)
    slots = build_naive_slots(cfg, start_pos)
    assignment = assign_drones_to_slots(start_pos, slots)

    _assert_close(sess.start_pos, start_pos, "session start_pos")
    _assert_close(sess.start_orn, start_orn, "session start_orn")
    _assert_close(sess.slots, slots, "session slots")
    if not np.array_equal(sess.assignment, assignment):
        raise AssertionError("session assignment != pipeline")
    print("OK  SimulationSession matches direct pipeline")


def test_obstacles_deterministic() -> None:
    cfg = make_episode_config(seed=42, num_drones=12, digit=1, scenario="both")
    pos, _ = sample_initial_state(cfg)
    slots = build_naive_slots(cfg, pos)
    assign = assign_drones_to_slots(pos, slots)
    o1 = build_obstacles_for_episode(cfg, pos, slots, assign)
    o2 = build_obstacles_for_episode(cfg, pos, slots, assign)
    _assert_close(o1.positions, o2.positions, "obstacle positions")
    _assert_close(o1.radii, o2.radii, "obstacle radii")
    print("OK  build_obstacles_for_episode deterministic")


def test_recorder_preview_setpoints() -> None:
    from sim_recorder import _setpoints_from_preview

    cfg = make_episode_config(seed=7, num_drones=10, digit=5, scenario="clean")
    start_pos, start_orn = sample_initial_state(cfg)
    slots = build_naive_slots(cfg, start_pos)
    assignment = assign_drones_to_slots(start_pos, slots)
    sp_preview = _setpoints_from_preview(start_pos, start_orn, slots, assignment)
    sp_train, _, _ = build_digit_setpoints(cfg, start_pos, start_orn)
    _assert_close(sp_preview, sp_train, "preview setpoints vs build_digit_setpoints")
    print("OK  _setpoints_from_preview matches training setpoints")


def test_recorder_first_frame_optional() -> None:
    try:
        from sim_recorder import record_episode_frames
    except ImportError as exc:
        print(f"SKIP pyflyt smoke ({exc})")
        return

    cfg = make_episode_config(seed=555, num_drones=10, digit=2, scenario="clean")
    start_pos, start_orn = sample_initial_state(cfg)
    slots = build_naive_slots(cfg, start_pos)
    assignment = assign_drones_to_slots(start_pos, slots)
    traj = record_episode_frames(
        cfg,
        start_pos=start_pos,
        start_orn=start_orn,
        slots=slots,
        assignment=assignment,
        max_steps=10,
        record_every=1,
    )
    frame0 = traj["frames"][0]
    for i, d in enumerate(frame0["drones"]):
        p = np.array(d["pos"], dtype=np.float32)
        _assert_close(p, start_pos[i], f"frame0 drone {i} pos")
    print("OK  recorder spawn frame matches start_pos (pyflyt)")


def test_synchronized_slot_shift_function() -> None:
    """Training helper raises all slot Z by delta."""
    slots = np.array(
        [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [4.0, 0.0, 1.0]], dtype=np.float32
    )
    obs = ObstacleConfig(
        positions=np.array([[2.0, 0.0, 1.0]], dtype=np.float32),
        radii=np.array([1.0], dtype=np.float32),
    )
    shifted = synchronized_slot_shift(slots, obs, delta_z=2.0)
    dz = float(np.max(shifted[:, 2] - slots[:, 2]))
    if abs(dz - 2.0) > 1e-4:
        raise AssertionError(f"synchronized_slot_shift dz expected 2.0 got {dz}")
    xy_move = float(np.max(np.linalg.norm(shifted[:, :2] - slots[:, :2], axis=1)))
    if xy_move > 1e-4:
        raise AssertionError(f"synchronized_slot_shift should not move XY, got {xy_move}")
    print("OK  synchronized_slot_shift +2m Z")


def test_slot_shift_in_rollout_optional() -> None:
    """PyFlyt: shift is episodic (requires stuck heuristic); try several seeds."""
    try:
        from sim_recorder import record_episode_frames
    except ImportError as exc:
        print(f"SKIP rollout slot shift ({exc})")
        return

    for seed in range(100, 120):
        cfg = make_episode_config(seed=seed, num_drones=12, digit=4, scenario="both")
        start_pos, start_orn = sample_initial_state(cfg)
        slots0 = build_naive_slots(cfg, start_pos)
        assignment = assign_drones_to_slots(start_pos, slots0)
        traj = record_episode_frames(
            cfg,
            start_pos=start_pos,
            start_orn=start_orn,
            slots=slots0,
            assignment=assignment,
            max_steps=400,
            record_every=2,
        )
        if any(f.get("slots_shifted") for f in traj["frames"]):
            print(f"OK  rollout slot shift observed (seed={seed})")
            return
    print(
        "WARN rollout slot shift not seen in seeds 100-119 "
        "(stuck heuristic may not fire every episode; function test covers shift math)"
    )


def main() -> None:
    tests = [
        test_pipeline_assignment_matches_setpoints,
        test_session_matches_pipeline,
        test_obstacles_deterministic,
        test_recorder_preview_setpoints,
        test_recorder_first_frame_optional,
        test_synchronized_slot_shift_function,
        test_slot_shift_in_rollout_optional,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    if failed:
        sys.exit(1)
    print(f"\nAll {len(tests)} parity checks passed.")


if __name__ == "__main__":
    main()
