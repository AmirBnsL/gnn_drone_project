"""Website-only slot Z-shift, strict stuck gates, and obstacle safety clamp."""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

STUCK_POS_ERR = 0.4


def _detect_slot_blocked(
    local_vel_window: np.ndarray,
    desired_direction: np.ndarray,
    obstacle_direction: np.ndarray,
    obstacle_distance_along_slot: float,
    *,
    vel_eps: float,
    max_along_dist: float,
) -> bool:
    if local_vel_window.size == 0:
        return False
    speeds = np.linalg.norm(local_vel_window, axis=1)
    if float(np.max(speeds)) > vel_eps:
        return False
    dir_norm = float(np.linalg.norm(desired_direction[:2]))
    obs_norm = float(np.linalg.norm(obstacle_direction[:2]))
    if dir_norm < 1e-6 or obs_norm < 1e-6:
        return False
    goal_dir = desired_direction[:2] / dir_norm
    obs_dir = obstacle_direction[:2] / obs_norm
    if obstacle_distance_along_slot < 0.0 or obstacle_distance_along_slot > max_along_dist:
        return False
    cos_sim = float(np.dot(goal_dir, obs_dir))
    return cos_sim > 0.3

import viz_sim_config as vcfg


def shift_slots_z(slots: np.ndarray, delta_z: float | None = None) -> np.ndarray:
    """Raise entire formation by delta_z (m); XY unchanged."""
    dz = float(vcfg.SLOT_SHIFT_DELTA_Z if delta_z is None else delta_z)
    shifted = np.asarray(slots, dtype=np.float32).copy()
    shifted[:, 2] += dz
    return shifted


def slot_xy_error(drone_pos: np.ndarray, setpoint: np.ndarray) -> float:
    tgt = np.array([setpoint[0], setpoint[1], setpoint[3]], dtype=np.float32)
    return float(np.linalg.norm(tgt[:2] - drone_pos[:2]))


def min_obstacle_surface_distance(
    position: np.ndarray, obstacles_lidar: np.ndarray
) -> float:
    if obstacles_lidar.size == 0:
        return float("inf")
    gp = np.asarray(position, dtype=np.float32)
    best = float("inf")
    for obs in obstacles_lidar:
        d = float(np.linalg.norm(gp[:2] - obs[:2])) - float(obs[2])
        best = min(best, d)
    return best


def min_planar_lidar(position: np.ndarray, yaw: float, obstacles_lidar: np.ndarray) -> float:
    if obstacles_lidar.size == 0:
        return float("inf")
    from merged_work.models_creation.setpoint_rollout import planar_lidar

    return float(np.min(planar_lidar(position, yaw, obstacles_lidar)))


def check_stuck_strict(
    local_lin_vel: np.ndarray,
    drone_pos: np.ndarray,
    target_pos: np.ndarray,
    obstacles: np.ndarray,
    vel_history: deque,
) -> bool:
    """Stricter stuck than rollout default: full velocity window + tighter along-ray."""
    from merged_work.models_creation.setpoint_rollout import _nearest_obstacle_along

    vel_history.append(local_lin_vel.copy())
    need = int(vcfg.VEL_STUCK_HISTORY_LEN) if vcfg.SHIFT_REQUIRE_FULL_VEL_WINDOW else 2
    if len(vel_history) < need:
        return False

    speeds = [float(np.linalg.norm(v)) for v in vel_history]
    if float(np.mean(speeds)) > float(vcfg.VEL_STUCK_EPS):
        return False

    window = np.stack(list(vel_history), axis=0)
    err_w = target_pos[:2] - drone_pos[:2]
    pe = float(np.linalg.norm(err_w))
    if pe < STUCK_POS_ERR or pe < 1e-6:
        return False
    if pe > float(vcfg.SHIFT_MAX_SLOT_ERROR):
        return False

    goal_dir = err_w / pe
    along, obs_dir = _nearest_obstacle_along(drone_pos, goal_dir, obstacles)
    desired = np.array([err_w[0], err_w[1], 0.0], dtype=np.float32)
    return _detect_slot_blocked(
        window,
        desired,
        np.array([obs_dir[0], obs_dir[1], 0.0], dtype=np.float32),
        along,
        vel_eps=float(vcfg.VEL_STUCK_EPS),
        max_along_dist=float(vcfg.SHIFT_MAX_ALONG_DIST),
    )


def local_stuck_flags_strict(
    env,
    active: List[int],
    setpoints: np.ndarray,
    obstacles_lidar: np.ndarray,
    vel_hists: dict,
) -> Tuple[List[bool], List[float]]:
    import pybullet as p

    flags: List[bool] = []
    errors: List[float] = []
    for i, di in enumerate(active):
        st = env.drones[di].state
        gp = np.array(st[3], copy=True)
        ge = np.array(st[1], copy=True)
        gv = np.array(st[2], copy=True)
        R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(ge))).reshape(3, 3)
        tgt = np.array([setpoints[i, 0], setpoints[i, 1], setpoints[i, 3]])
        flags.append(
            check_stuck_strict(R.T @ gv, gp, tgt, obstacles_lidar, vel_hists[di])
        )
        errors.append(slot_xy_error(gp, setpoints[i]))
    return flags, errors


def evaluate_shift_eligibility(
    step: int,
    stuck_flags: List[bool],
    slot_errors: List[float],
    obstacles_present: bool,
    steps_since_shift: int,
    did_shift: bool,
) -> Tuple[bool, str]:
    if not obstacles_present:
        return False, "no_obstacles"
    if step < int(vcfg.SHIFT_MIN_STEPS):
        return False, "warmup"
    if steps_since_shift < int(vcfg.SHIFT_COOLDOWN_STEPS):
        return False, "cooldown"
    if vcfg.SLOT_SHIFT_ONCE and did_shift:
        return False, "shift_once"
    local_stuck = int(sum(1 for s in stuck_flags if s))
    if local_stuck < int(vcfg.SHIFT_REQUIRE_LOCAL_STUCK_COUNT):
        return False, "insufficient_local_stuck"
    return True, "ok"


def _obstacle_surface_xy(position: np.ndarray, obs: np.ndarray) -> float:
    gp = np.asarray(position, dtype=np.float32)
    oc = np.asarray(obs[:2], dtype=np.float32)
    return float(np.linalg.norm(gp[:2] - oc)) - float(obs[2])


def compute_apf_setpoint_viz(
    drone_pos: np.ndarray,
    original_sp: np.ndarray,
    obstacles_lidar: np.ndarray,
    other_positions: List[np.ndarray],
    *,
    yaw: float = 0.0,
    obs_influence: float | None = None,
    force_cap: float | None = None,
    mode: str = "central",
) -> np.ndarray:
    """Website APF wrapper: proximity override + configurable repulsion strength."""
    from merged_work.models_creation.setpoint_rollout import (
        LIDAR_MAX_RANGE,
        _obstacle_visible_teacher,
        apf_repulsive_force,
    )

    if obstacles_lidar.size == 0:
        return np.asarray(original_sp, dtype=np.float32).copy()

    dp = np.asarray(drone_pos, dtype=np.float32)
    sp = np.asarray(original_sp, dtype=np.float32).copy()
    if obs_influence is None:
        obs_inf = float(
            vcfg.CENTRAL_APF_OBS_INFLUENCE
            if mode == "central"
            else vcfg.DECENTRAL_APF_OBS_INFLUENCE
        )
    else:
        obs_inf = float(obs_influence)
    if force_cap is None:
        cap = float(
            vcfg.CENTRAL_APF_FORCE_CAP
            if mode == "central"
            else vcfg.DECENTRAL_APF_FORCE_CAP
        )
    else:
        cap = float(force_cap)
    prox_margin = float(vcfg.APF_PROXIMITY_MARGIN)
    clearance = float(vcfg.SAFETY_CLEARANCE)

    force = np.zeros(2, dtype=np.float32)
    for obs in obstacles_lidar:
        obs_arr = np.asarray(obs, dtype=np.float32)
        visible = _obstacle_visible_teacher(dp, yaw, obs_arr, LIDAR_MAX_RANGE)
        if not visible:
            surf = _obstacle_surface_xy(dp, obs_arr)
            if surf > clearance + prox_margin:
                continue
        force += apf_repulsive_force(dp[:2], obs_arr[:2], float(obs_arr[2]), obs_inf, 2.0)

    for op in other_positions:
        op = np.asarray(op, dtype=np.float32)
        if float(np.linalg.norm(dp[:2] - op[:2])) > LIDAR_MAX_RANGE:
            continue
        force += apf_repulsive_force(dp[:2], op[:2], 0.3, 2.0, 1.5)

    mag = float(np.linalg.norm(force))
    if mag > cap:
        force = force / mag * cap

    sp[0] += force[0]
    sp[1] += force[1]
    return sp


def safety_clamp_setpoint(
    position: np.ndarray,
    proposed: np.ndarray,
    obstacles_lidar: np.ndarray,
) -> np.ndarray:
    """
    If proposed XY setpoint moves toward an obstacle within clearance, nudge away.
    Does not replace APF/GNN; last-line guard before PyFlyt.
    """
    out = np.asarray(proposed, dtype=np.float32).copy()
    if obstacles_lidar.size == 0:
        return out

    pos = np.asarray(position, dtype=np.float32)
    clearance = float(vcfg.SAFETY_CLEARANCE)
    cmd_xy = np.array([out[0], out[1]], dtype=np.float32)
    delta = cmd_xy - pos[:2]
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm < 1e-6:
        return out

    for obs in obstacles_lidar:
        oc = obs[:2].astype(np.float32)
        r = float(obs[2])
        to_obs = oc - pos[:2]
        dist_center = float(np.linalg.norm(to_obs))
        surface = dist_center - r
        if surface > clearance + 0.5:
            continue
        # Repulse command away from obstacle center in XY
        if dist_center < 1e-4:
            away = -delta / max(delta_norm, 1e-6)
        else:
            away = (pos[:2] - oc) / dist_center
        into = float(np.dot(delta / delta_norm, (oc - pos[:2]) / max(dist_center, 1e-6)))
        if into > 0.2 or surface < clearance:
            push_scale = float(vcfg.SAFETY_CLAMP_PUSH_SCALE)
            push = away * min(1.2 * push_scale, (clearance - surface + 0.25) * push_scale)
            cmd_xy = pos[:2] + delta + push
            out[0] = float(cmd_xy[0])
            out[1] = float(cmd_xy[1])

    return out
