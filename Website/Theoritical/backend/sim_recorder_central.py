"""Centralized Hungarian + APF + Z slot-shift simulation."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

import numpy as np
import pybullet as p
from PyFlyt.core import Aviary

from merged_work.models_creation.dataset_pipeline import (
    EpisodeConfig,
    ObstacleConfig,
    build_naive_slots,
    build_obstacles_for_episode,
    obstacles_to_lidar_array,
    sample_initial_state,
)
from merged_work.models_creation.setpoint_rollout import (
    build_digit_setpoints,
    spawn_spheres,
)

import viz_sim_config as vcfg
from obstacle_utils import filter_obstacles_by_altitude, obstacles_to_3d_array
from json_sanitize import finite_or_none
from sim_frame_utils import (
    DT,
    apply_setpoints_from_slots,
    drone_alerts,
    frame_payload,
    max_assigned_slot_error,
    setpoints_from_slots,
)
from slot_shift_safety import (
    compute_apf_setpoint_viz,
    evaluate_shift_eligibility,
    local_stuck_flags_strict,
    min_obstacle_surface_distance,
    min_planar_lidar,
    safety_clamp_setpoint,
    shift_slots_z,
)

PHYSICS_HZ = 240


def _reset_stuck_state(vel_hists: Dict[int, deque]) -> None:
    for hist in vel_hists.values():
        hist.clear()


def record_episode_frames(
    cfg: EpisodeConfig,
    *,
    start_pos: np.ndarray | None = None,
    start_orn: np.ndarray | None = None,
    slots: np.ndarray | None = None,
    assignment: np.ndarray | None = None,
    obs_cfg: ObstacleConfig | None = None,
    max_steps: int | None = None,
    record_every: int | None = None,
) -> Dict[str, Any]:
    step_cap = vcfg.simulation_step_cap(max_steps)
    if record_every is None:
        record_every = vcfg.RECORD_EVERY

    if start_pos is None or start_orn is None:
        start_pos, start_orn = sample_initial_state(cfg)

    if slots is not None and assignment is not None:
        slots = np.asarray(slots, dtype=np.float32).copy()
        assignment = np.asarray(assignment, dtype=np.int64).copy()
        setpoints = setpoints_from_slots(start_orn, slots, assignment)
    else:
        setpoints, assignment, _ = build_digit_setpoints(cfg, start_pos, start_orn)
        slots = build_naive_slots(cfg, start_pos)

    if obs_cfg is None:
        obs_cfg = build_obstacles_for_episode(cfg, start_pos, slots, assignment)
    obstacles_lidar = obstacles_to_lidar_array(obs_cfg)
    obstacles_3d = obstacles_to_3d_array(obs_cfg)

    env = Aviary(
        start_pos=start_pos.copy(),
        start_orn=start_orn.copy(),
        drone_type="quadx",
        render=False,
    )
    env.set_mode(7)
    client = env._client
    if obstacles_lidar.size > 0:
        spawn_spheres(obs_cfg, client)
        env.register_all_new_bodies()

    n = cfg.num_drones
    active = list(range(n))
    vel_hist = {i: deque(maxlen=vcfg.VEL_STUCK_HISTORY_LEN) for i in range(n)}
    converged_count = 0
    frames: List[Dict[str, Any]] = []
    did_shift = False
    stuck_checks = 0
    steps_since_shift = 10_000
    shift_block_reasons = {
        "no_obstacles": 0,
        "warmup": 0,
        "cooldown": 0,
        "shift_once": 0,
        "insufficient_local_stuck": 0,
    }

    spawn_pos = [start_pos[i].copy() for i in range(n)]
    spawn_yaw = [float(start_orn[i, 2]) for i in range(n)]
    frames.append(
        frame_payload(
            spawn_pos,
            spawn_yaw,
            slots,
            assignment,
            obs_cfg,
            drone_alerts(spawn_pos, spawn_yaw, obstacles_lidar),
            mode="central",
        )
    )

    last_positions: List[np.ndarray] = [p.copy() for p in spawn_pos]

    step = -1
    for step in range(step_cap):
        shifted_this_step = False
        shift_blocked_reason = "n/a"
        positions_pre: List[np.ndarray] = []
        yaws_pre: List[float] = []
        stuck_flags: List[bool] = []
        slot_errors: List[float] = []

        for i, di in enumerate(active):
            st = env.drones[di].state
            gp = np.array(st[3], copy=True)
            ge = np.array(st[1], copy=True)
            positions_pre.append(gp)
            yaws_pre.append(float(ge[2]))

        stuck_flags, slot_errors = local_stuck_flags_strict(
            env, active, setpoints, obstacles_lidar, vel_hist
        )
        stuck_any = any(stuck_flags)
        if stuck_any:
            stuck_checks += 1

        shift_ok, shift_blocked_reason = evaluate_shift_eligibility(
            step,
            stuck_flags,
            slot_errors,
            obstacles_lidar.size > 0,
            steps_since_shift,
            did_shift,
        )
        if shift_blocked_reason in shift_block_reasons:
            shift_block_reasons[shift_blocked_reason] += 1
        if shift_ok and stuck_any:
            slots = shift_slots_z(slots)
            did_shift = True
            shifted_this_step = True
            steps_since_shift = 0
            apply_setpoints_from_slots(setpoints, slots, assignment)
            _reset_stuck_state(vel_hist)
            converged_count = 0
        else:
            steps_since_shift += 1

        others_cache = {di: np.array(env.drones[di].state[3]) for di in active}
        min_surf = float("inf")
        min_lidar = float("inf")
        for i, di in enumerate(active):
            st = env.drones[di].state
            gp = np.array(st[3], copy=True)
            ge = np.array(st[1], copy=True)
            obs_at_alt = filter_obstacles_by_altitude(float(gp[2]), obstacles_3d)
            min_surf = min(
                min_surf,
                min_obstacle_surface_distance(gp, obs_at_alt),
            )
            min_lidar = min(
                min_lidar,
                min_planar_lidar(gp, float(ge[2]), obs_at_alt),
            )
            others = [others_cache[dj] for dj in active if dj != di]
            mod = compute_apf_setpoint_viz(
                gp,
                setpoints[i],
                obs_at_alt,
                others,
                yaw=float(ge[2]),
                mode="central",
            )
            mod = safety_clamp_setpoint(gp, mod, obs_at_alt)
            env.set_setpoint(di, mod)

        env.step()

        pos_for_conv = [
            np.array(env.drones[di].state[3], copy=True) for di in active
        ]
        last_positions = pos_for_conv
        max_slot_err = max_assigned_slot_error(pos_for_conv, slots, assignment)
        if max_slot_err < vcfg.CONV_THRESHOLD:
            converged_count += 1
        else:
            converged_count = 0

        if step % record_every == 0 or converged_count >= vcfg.CONV_STEPS:
            pos_list, yaw_list = [], []
            for i, di in enumerate(active):
                st = env.drones[di].state
                pos_list.append(np.array(st[3], copy=True))
                yaw_list.append(float(st[1][2]))
            slot_err_frame = max_assigned_slot_error(pos_list, slots, assignment)
            frames.append(
                frame_payload(
                    pos_list,
                    yaw_list,
                    slots,
                    assignment,
                    obs_cfg,
                    drone_alerts(pos_list, yaw_list, obstacles_lidar),
                    slots_shifted=shifted_this_step,
                    mode="central",
                    meta={
                        "shift_fired": shifted_this_step,
                        "shift_type": "z_altitude" if shifted_this_step else None,
                        "shift_blocked_reason": shift_blocked_reason,
                        "shift_eligible": shift_ok,
                        "local_stuck_count": int(sum(stuck_flags)),
                        "min_obstacle_surface_distance": finite_or_none(min_surf),
                        "min_lidar": finite_or_none(min_lidar),
                        "max_slot_error": slot_err_frame,
                        "did_shift_total": did_shift,
                        "shift_block_reasons": dict(shift_block_reasons),
                    },
                )
            )

        if converged_count >= vcfg.CONV_STEPS:
            break

    env.disconnect()

    final_max_slot_error = max_assigned_slot_error(
        last_positions, slots, assignment
    )

    converged = converged_count >= vcfg.CONV_STEPS
    stopped_reason = "converged" if converged else "max_steps"
    return {
        "frames": frames,
        "dt": DT * float(record_every),
        "converged": converged,
        "stopped_reason": stopped_reason,
        "max_steps_cap": step_cap,
        "steps": step + 1,
        "final_max_slot_error": final_max_slot_error,
        "did_slot_shift": did_shift,
        "stuck_checks": stuck_checks,
        "shift_block_reasons": shift_block_reasons,
        "mode": "central",
    }
