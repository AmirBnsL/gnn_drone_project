"""Decentralized GNN simulation: messages in comm radius, strict Z shift, safety shield."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

import numpy as np
import pybullet as p
import torch
from PyFlyt.core import Aviary

from agents.comm_orchestrator import (
    exchange_messages,
    sequential_turn_log,
    swarm_has_shift_proposal,
)
from agents.drone_agent import DroneAgent
from website_gnn.assignment_runner import assign_swarm
from website_gnn.setpoint_runner import (
    build_setpoint_graph,
    forward_setpoint,
    pred_to_global_setpoints,
)
from merged_work.models_creation.dataset_pipeline import (
    EpisodeConfig,
    ObstacleConfig,
    build_obstacles_for_episode,
    obstacles_to_lidar_array,
)
from merged_work.models_creation.digit_formations import build_digit_one_hot
from merged_work.models_creation.setpoint_rollout import (
    INTEGRAL_WINDOW,
    LIDAR_MAX_RANGE,
    planar_lidar,
    spawn_spheres,
)

import viz_sim_config as vcfg
from json_sanitize import finite_or_none
from obstacle_utils import filter_obstacles_by_altitude, obstacles_to_3d_array
from sim_frame_utils import (
    ALERT_EPS,
    DT,
    apply_setpoints_from_slots,
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


def _reset_stuck_state(vel_hists: Dict[int, deque]) -> None:
    for hist in vel_hists.values():
        hist.clear()


def _alerts_at_altitude(
    positions: List[np.ndarray],
    yaws: List[float],
    obstacles_3d: np.ndarray,
) -> List[bool]:
    alerts: List[bool] = []
    for gp, yaw in zip(positions, yaws):
        obs_at_alt = filter_obstacles_by_altitude(float(gp[2]), obstacles_3d)
        if obs_at_alt.size == 0:
            alerts.append(False)
        else:
            lidar = planar_lidar(gp, yaw, obs_at_alt)
            alerts.append(bool(float(np.min(lidar)) < LIDAR_MAX_RANGE - ALERT_EPS))
    return alerts


def record_episode_frames_decentral(
    cfg: EpisodeConfig,
    *,
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    slots: np.ndarray,
    assignment: np.ndarray | None = None,
    obs_cfg: ObstacleConfig | None = None,
    max_steps: int | None = None,
    record_every: int | None = None,
) -> Dict[str, Any]:
    step_cap = vcfg.simulation_step_cap(max_steps)
    if record_every is None:
        record_every = vcfg.RECORD_EVERY

    start_pos = np.asarray(start_pos, dtype=np.float32).copy()
    start_orn = np.asarray(start_orn, dtype=np.float32).copy()
    slots = np.asarray(slots, dtype=np.float32).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if assignment is None:
        assignment, assign_info = assign_swarm(start_pos, slots, cfg.digit, device=device)
    else:
        assignment = np.asarray(assignment, dtype=np.int64).copy()
        assign_info = {"messages_sent": 0, "provided": True}
    messages_sent = int(assign_info.get("messages_sent", 0))

    setpoints = setpoints_from_slots(start_orn, slots, assignment)
    if obs_cfg is None:
        obs_cfg = build_obstacles_for_episode(cfg, start_pos, slots, assignment)
    obstacles_lidar = obstacles_to_lidar_array(obs_cfg)
    obstacles_3d = obstacles_to_3d_array(obs_cfg)
    foh = build_digit_one_hot(cfg.digit)

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
    agents = [
        DroneAgent(
            i,
            vcfg.COMM_RADIUS,
            vcfg.VEL_STUCK_HISTORY_LEN,
            INTEGRAL_WINDOW,
        )
        for i in range(n)
    ]
    for i, agent in enumerate(agents):
        agent.my_slot_idx = int(assignment[i])
        agent.setpoint = setpoints[i].copy()

    vel_hists = {i: deque(maxlen=vcfg.VEL_STUCK_HISTORY_LEN) for i in range(n)}
    integral_bufs = [deque(maxlen=INTEGRAL_WINDOW) for _ in range(n)]
    prev_frames = [np.zeros(41, dtype=np.float32) for _ in range(n)]

    converged_count = 0
    frames: List[Dict[str, Any]] = []
    did_shift = False
    steps_since_shift = 10_000
    shift_proposals_total = 0
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
            _alerts_at_altitude(spawn_pos, spawn_yaw, obstacles_3d),
            mode="decentral",
            meta={
                "messages_sent": messages_sent,
                "assignment_phase": True,
            },
        )
    )

    last_positions: List[np.ndarray] = [p.copy() for p in spawn_pos]

    step = -1
    for step in range(step_cap):
        shifted_this_step = False
        shift_blocked_reason = "n/a"
        positions, eulers, gvels, ang_vels, yaws = [], [], [], [], []
        local_vels = []
        for i, di in enumerate(active):
            st = env.drones[di].state
            gp = np.array(st[3], copy=True)
            ge = np.array(st[1], copy=True)
            gv = np.array(st[2], copy=True)
            av = np.array(st[0], copy=True)
            positions.append(gp)
            eulers.append(ge)
            gvels.append(gv)
            ang_vels.append(av)
            yaws.append(float(ge[2]))
            R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(ge))).reshape(3, 3)
            local_vels.append(R.T @ gv)

        stuck_flags, slot_errors = local_stuck_flags_strict(
            env, active, setpoints, obstacles_lidar, vel_hists
        )
        alerts = _alerts_at_altitude(positions, yaws, obstacles_3d)

        proposal_deliveries = exchange_messages(
            agents, positions, yaws, local_vels, alerts, stuck_flags, step
        )
        shift_proposals_total += proposal_deliveries

        run_gnn = (step % vcfg.SETPOINT_CTRL_EVERY) == 0
        shift_prob = 0.0
        goal_repair_ok = False
        inference_turns = sequential_turn_log(n)
        gnn_cmd_mean_step = 0.0
        slot_error_mean = 0.0
        obs_considered_total = 0
        min_surf = float("inf")
        min_lidar = float("inf")

        shift_eligible, shift_blocked_reason = evaluate_shift_eligibility(
            step,
            stuck_flags,
            slot_errors,
            obstacles_lidar.size > 0,
            steps_since_shift,
            did_shift,
        )
        if shift_blocked_reason in shift_block_reasons:
            shift_block_reasons[shift_blocked_reason] += 1

        if run_gnn:
            device_loaded = _device_from_bundle()
            graph, curr_frames, local_vs = build_setpoint_graph(
                positions,
                eulers,
                gvels,
                ang_vels,
                setpoints,
                foh,
                obstacles_lidar,
                integral_bufs,
                prev_frames,
                vcfg.COMM_RADIUS,
                device_loaded,
            )
            pred_phys, shift_prob = forward_setpoint(graph, curr_frames)
            for i in range(n):
                prev_frames[i] = curr_frames[i]

            hybrid_ok = (
                vcfg.ENABLE_HYBRID_SLOT_SHIFT
                and shift_prob > vcfg.SHIFT_TRIGGER_THRESHOLD
                and shift_eligible
                and swarm_has_shift_proposal(agents)
            )
            goal_repair_ok = (
                vcfg.ENABLE_DECENTRAL_GOAL_REPAIR
                and shift_eligible
                and any(stuck_flags)
            )
            shift_fired = hybrid_ok or goal_repair_ok
            if shift_fired:
                slots = shift_slots_z(slots)
                apply_setpoints_from_slots(setpoints, slots, assignment)
                for agent in agents:
                    agent.setpoint = setpoints[agent.id].copy()
                    agent.clear_mailboxes()
                did_shift = True
                shifted_this_step = True
                steps_since_shift = 0
                _reset_stuck_state(vel_hists)
                converged_count = 0
            else:
                steps_since_shift += 1

            mod_sp = pred_to_global_setpoints(
                pred_phys,
                np.array(positions),
                eulers,
                local_vs,
                setpoints,
                step,
            )
            pos_arr = np.asarray(positions, dtype=np.float32)
            cmd_targets = np.column_stack([mod_sp[:, 0], mod_sp[:, 1], mod_sp[:, 3]])
            slot_targets = np.column_stack([setpoints[:, 0], setpoints[:, 1], setpoints[:, 3]])
            gnn_cmd_mean_step = float(np.mean(np.linalg.norm(cmd_targets - pos_arr, axis=1)))
            slot_error_mean = float(np.mean(np.linalg.norm(slot_targets - pos_arr, axis=1)))
            for i, di in enumerate(active):
                gp = positions[i]
                obs_at_alt = filter_obstacles_by_altitude(float(gp[2]), obstacles_3d)
                obs_considered_total += int(obs_at_alt.shape[0]) if obs_at_alt.ndim == 2 else 0
                min_surf = min(min_surf, min_obstacle_surface_distance(gp, obs_at_alt))
                min_lidar = min(min_lidar, min_planar_lidar(gp, yaws[i], obs_at_alt))
                if vcfg.ENABLE_DECENTRAL_APF:
                    others = [positions[j] for j in range(n) if j != i]
                    apf_sp = compute_apf_setpoint_viz(
                        gp, mod_sp[i], obs_at_alt, others, yaw=yaws[i], mode="decentral"
                    )
                    safe_sp = safety_clamp_setpoint(gp, apf_sp, obs_at_alt)
                else:
                    safe_sp = safety_clamp_setpoint(gp, mod_sp[i], obs_at_alt)
                env.set_setpoint(di, safe_sp)
        else:
            steps_since_shift += 1
            for i, di in enumerate(active):
                gp = positions[i]
                obs_at_alt = filter_obstacles_by_altitude(float(gp[2]), obstacles_3d)
                obs_considered_total += int(obs_at_alt.shape[0]) if obs_at_alt.ndim == 2 else 0
                min_surf = min(min_surf, min_obstacle_surface_distance(gp, obs_at_alt))
                min_lidar = min(min_lidar, min_planar_lidar(gp, yaws[i], obs_at_alt))
                if vcfg.ENABLE_DECENTRAL_APF:
                    others = [positions[j] for j in range(n) if j != i]
                    apf_sp = compute_apf_setpoint_viz(
                        gp, setpoints[i], obs_at_alt, others, yaw=yaws[i], mode="decentral"
                    )
                    safe_sp = safety_clamp_setpoint(gp, apf_sp, obs_at_alt)
                else:
                    safe_sp = safety_clamp_setpoint(gp, setpoints[i], obs_at_alt)
                env.set_setpoint(di, safe_sp)

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
                    _alerts_at_altitude(pos_list, yaw_list, obstacles_3d),
                    slots_shifted=shifted_this_step,
                    mode="decentral",
                    meta={
                        "shift_prob": shift_prob,
                        "shift_proposals": proposal_deliveries,
                        "shift_fired": shifted_this_step,
                        "shift_type": "z_altitude" if shifted_this_step else None,
                        "goal_repair": goal_repair_ok if shifted_this_step else False,
                        "shift_once_enabled": vcfg.SLOT_SHIFT_ONCE,
                        "shift_eligible": shift_eligible,
                        "shift_blocked_reason": shift_blocked_reason,
                        "local_stuck_count": int(sum(stuck_flags)),
                        "min_obstacle_surface_distance": finite_or_none(min_surf),
                        "min_lidar": finite_or_none(min_lidar),
                        "max_slot_error": slot_err_frame,
                        "did_shift_total": did_shift,
                        "shift_block_reasons": dict(shift_block_reasons),
                        "inference_turns": inference_turns,
                        "messages_sent": messages_sent,
                        "gnn_cmd_mean_step": gnn_cmd_mean_step,
                        "slot_error_mean": slot_error_mean,
                        "obstacles_considered": obs_considered_total,
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
        "mode": "decentral",
        "messages_sent": messages_sent,
        "shift_proposals_total": shift_proposals_total,
        "shift_block_reasons": shift_block_reasons,
        "assignment": assignment.tolist(),
    }


def _device_from_bundle():
    from website_gnn.setpoint_runner import _load_setpoint_bundle

    _, _, device = _load_setpoint_bundle()
    return device
