"""Shared trajectory frame helpers for central and decentral recorders."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from merged_work.models_creation.setpoint_rollout import LIDAR_MAX_RANGE, planar_lidar

PHYSICS_HZ = 240
DT = 1.0 / PHYSICS_HZ
ALERT_EPS = 0.05


def obstacle_list(obs_cfg) -> List[List[float]]:
    if obs_cfg.positions.size == 0:
        return []
    out: List[List[float]] = []
    for pos, r in zip(obs_cfg.positions, obs_cfg.radii):
        out.append([float(pos[0]), float(pos[1]), float(pos[2]), float(r)])
    return out


def drone_alerts(
    positions: List[np.ndarray],
    yaws: List[float],
    obstacles_lidar: np.ndarray,
) -> List[bool]:
    alerts = []
    for gp, yaw in zip(positions, yaws):
        if obstacles_lidar.size == 0:
            alerts.append(False)
            continue
        lidar = planar_lidar(gp, yaw, obstacles_lidar)
        alerts.append(bool(float(np.min(lidar)) < LIDAR_MAX_RANGE - ALERT_EPS))
    return alerts


def frame_payload(
    positions: List[np.ndarray],
    yaws: List[float],
    slots: np.ndarray,
    assignment: np.ndarray,
    obs_cfg,
    alerts: List[bool],
    *,
    slots_shifted: bool = False,
    mode: str = "central",
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    center = [
        float(np.mean([p[0] for p in positions])),
        float(np.mean([p[1] for p in positions])),
    ]
    drones = []
    for i, (gp, yaw, alert) in enumerate(zip(positions, yaws, alerts)):
        drones.append(
            {
                "id": i,
                "pos": [float(gp[0]), float(gp[1]), float(gp[2])],
                "yaw": float(yaw),
                "alert": bool(alert),
            }
        )
    slot_list = [
        [float(slots[j, 0]), float(slots[j, 1]), float(slots[j, 2])]
        for j in range(slots.shape[0])
    ]
    frame: Dict[str, Any] = {
        "center": center,
        "drones": drones,
        "slots": slot_list,
        "assignment": [int(assignment[k]) for k in range(len(assignment))],
        "obstacles": obstacle_list(obs_cfg),
        "slots_shifted": bool(slots_shifted),
        "mode": mode,
    }
    if meta:
        frame["meta"] = meta
    return frame


def _validate_slot_index(si: int, n: int, drone_idx: int) -> None:
    if not (0 <= si < n):
        raise ValueError(
            f"invalid assignment[{drone_idx}]={si}; expected slot index in [0, {n})"
        )


def setpoints_from_slots(
    start_orn: np.ndarray,
    slots: np.ndarray,
    assignment: np.ndarray,
) -> np.ndarray:
    n = assignment.shape[0]
    setpoints = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        si = int(assignment[i])
        _validate_slot_index(si, n, i)
        setpoints[i, 0] = slots[si, 0]
        setpoints[i, 1] = slots[si, 1]
        setpoints[i, 2] = start_orn[i, 2]
        setpoints[i, 3] = slots[si, 2]
    return setpoints


def max_assigned_slot_error(
    positions: List[np.ndarray],
    slots: np.ndarray,
    assignment: np.ndarray,
) -> float:
    """Max 3D distance from each drone to its assigned slot."""
    if not positions or slots.size == 0:
        return float("inf")
    slots = np.asarray(slots, dtype=np.float32)
    assignment = np.asarray(assignment, dtype=np.int64)
    best = 0.0
    for i, gp in enumerate(positions):
        si = int(assignment[i])
        if si < 0 or si >= slots.shape[0]:
            continue
        tgt = slots[si, :3]
        err = float(np.linalg.norm(np.asarray(gp, dtype=np.float32)[:3] - tgt))
        best = max(best, err)
    return best


def apply_setpoints_from_slots(
    setpoints: np.ndarray, slots: np.ndarray, assignment: np.ndarray
) -> None:
    n = setpoints.shape[0]
    for i in range(n):
        si = int(assignment[i])
        _validate_slot_index(si, n, i)
        setpoints[i, 0] = slots[si, 0]
        setpoints[i, 1] = slots[si, 1]
        setpoints[i, 3] = slots[si, 2]
