"""Unit tests for assignment bijection repair."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_backend = Path(__file__).resolve().parent
_root = _backend.parents[2]
for p in (_backend, _root):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from assignment_repair import repair_assignment_bijection

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore


def _is_bijection(assignment: np.ndarray, n: int) -> bool:
    if assignment.shape[0] != n:
        return False
    if np.any(assignment < 0) or np.any(assignment >= n):
        return False
    return len(np.unique(assignment)) == n


def test_repair_unassigned_drone():
    n = 4
    start_pos = np.random.randn(n, 3).astype(np.float32)
    slots = start_pos + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    broken = np.array([-1, 1, 2, 3], dtype=np.int64)
    repaired, info = repair_assignment_bijection(broken, start_pos, slots)
    assert info["repaired"] is True
    assert info["num_fixed"] >= 1
    assert _is_bijection(repaired, n)


def test_repair_duplicate_slots():
    n = 4
    start_pos = np.array(
        [[0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1]], dtype=np.float32
    )
    slots = start_pos + np.array([0, 2, 0], dtype=np.float32)
    broken = np.array([0, 0, 2, 3], dtype=np.int64)
    repaired, info = repair_assignment_bijection(broken, start_pos, slots)
    assert info["repaired"] is True
    assert _is_bijection(repaired, n)


def test_repair_already_valid():
    n = 3
    start_pos = np.eye(3, dtype=np.float32)
    slots = start_pos + 1.0
    good = np.array([0, 1, 2], dtype=np.int64)
    repaired, info = repair_assignment_bijection(good, start_pos, slots)
    assert info["repaired"] is False
    assert np.array_equal(repaired, good)


def test_setpoints_rejects_invalid_assignment():
    from sim_frame_utils import setpoints_from_slots

    n = 3
    start_orn = np.zeros((n, 3), dtype=np.float32)
    slots = np.arange(n, dtype=np.float32)[:, None] * np.array([1, 1, 1])
    slots = np.tile(slots, (1, 3)).astype(np.float32)
    try:
        setpoints_from_slots(start_orn, slots, np.array([-1, 0, 1], dtype=np.int64))
        raised = False
    except ValueError as e:
        raised = True
        assert "invalid assignment" in str(e)
    assert raised


def test_setpoints_with_repaired_assignment():
    from sim_frame_utils import setpoints_from_slots

    n = 4
    start_pos = np.array(
        [[0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1]], dtype=np.float32
    )
    slots = start_pos + np.array([0, 2, 0], dtype=np.float32)
    start_orn = np.zeros((n, 3), dtype=np.float32)
    broken = np.array([-1, 1, 2, 3], dtype=np.int64)
    repaired, _ = repair_assignment_bijection(broken, start_pos, slots)
    sp = setpoints_from_slots(start_orn, slots, repaired)
    for i in range(n):
        si = int(repaired[i])
        assert sp[i, 0] == slots[si, 0]
        assert sp[i, 1] == slots[si, 1]


def _run_all() -> None:
    test_repair_unassigned_drone()
    test_repair_duplicate_slots()
    test_repair_already_valid()
    print("assignment_repair: ok")
    try:
        test_setpoints_rejects_invalid_assignment()
        test_setpoints_with_repaired_assignment()
        print("setpoints guards: ok")
    except ModuleNotFoundError as e:
        print(f"setpoints tests skipped ({e})")
    print("test_assignment_repair: all passed")


if __name__ == "__main__":
    _run_all()
