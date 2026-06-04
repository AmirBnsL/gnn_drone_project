"""Hungarian repair for incomplete Bertsekas assignments (numpy/scipy only)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def _hungarian_assign(start_pos: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Minimum-distance drone-to-slot matching (same as merged_work assign_drones_to_slots)."""
    dist = np.linalg.norm(
        np.asarray(start_pos, dtype=np.float32)[:, None, :2]
        - np.asarray(slots, dtype=np.float32)[None, :, :2],
        axis=2,
    )
    _, col_ind = linear_sum_assignment(dist)
    return col_ind.astype(np.int64)


def repair_assignment_bijection(
    assignment: np.ndarray,
    start_pos: np.ndarray,
    slots: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ensure drone i -> unique slot in [0, N). Keeps valid Bertsekas pairs; Hungarian
    on remaining bad drones and free slots. Full Hungarian if > half are bad.
    """
    assignment = np.asarray(assignment, dtype=np.int64).reshape(-1)
    n = int(assignment.shape[0])
    if n == 0:
        return assignment.copy(), {"repaired": False, "num_fixed": 0, "full_replacement": False}

    start_pos = np.asarray(start_pos, dtype=np.float32)
    slots = np.asarray(slots, dtype=np.float32)
    if slots.shape[0] != n:
        raise ValueError(f"slots rows {slots.shape[0]} != assignment length {n}")

    result = np.full(n, -1, dtype=np.int64)
    used_slots: set[int] = set()
    bad_drones: List[int] = []

    for i in range(n):
        si = int(assignment[i])
        if 0 <= si < n and si not in used_slots:
            result[i] = si
            used_slots.add(si)
        else:
            bad_drones.append(i)

    num_bad = len(bad_drones)
    if num_bad == 0:
        return result, {"repaired": False, "num_fixed": 0, "full_replacement": False}

    if num_bad > n // 2:
        full = _hungarian_assign(start_pos, slots)
        return full, {"repaired": True, "num_fixed": num_bad, "full_replacement": True}

    free_slots = [s for s in range(n) if s not in used_slots]
    if len(free_slots) != len(bad_drones):
        full = _hungarian_assign(start_pos, slots)
        return full, {"repaired": True, "num_fixed": num_bad, "full_replacement": True}

    sub_pos = start_pos[bad_drones]
    sub_slots = slots[free_slots]
    dist = np.linalg.norm(sub_pos[:, None, :2] - sub_slots[None, :, :2], axis=2)
    _, col_ind = linear_sum_assignment(dist)
    for local_i, slot_local in enumerate(col_ind):
        result[bad_drones[local_i]] = int(free_slots[int(slot_local)])

    return result, {"repaired": True, "num_fixed": num_bad, "full_replacement": False}
