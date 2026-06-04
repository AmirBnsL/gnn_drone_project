"""
PyFlyt rollout for digit formations — V3-compatible graphs (41-dim × 2 frames).

- Sensor-gated APF (obstacles only within lidar range)
- Synchronized slot shift when drones stall on blocked slots
- No wind; spherical obstacles from obstacle_scenarios
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p
import torch
from PyFlyt.core import Aviary
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data

from merged_work.models_creation.dataset_pipeline import (
    COMM_RADIUS_DEFAULT,
    EpisodeConfig,
    build_naive_slots,
    build_obstacles_for_episode,
    detect_slot_blocked,
    make_episode_config,
    obstacles_to_lidar_array,
    sample_initial_state,
    synchronized_slot_shift,
)
from merged_work.models_creation.digit_formations import build_digit_one_hot
from merged_work.models_creation.obstacle_scenarios import ObstacleConfig

LIDAR_RAYS = 16
LIDAR_MAX_RANGE = 5.0
RAW_FRAME_DIM = 41  # 35 base - 4 old one-hot + 10 digit one-hot
FORMATION_ONE_HOT_DIM = 10
INTEGRAL_WINDOW = 10
CONV_THRESHOLD = 0.35
CONV_STEPS = 25
VEL_EPS = 0.12
STUCK_POS_ERR = 0.4
STUCK_MAX_ALONG_DIST = 5.0
SHIFT_MAX_ALONG_DIST = 1.5
SHIFT_MAX_SLOT_ERROR = 5.0
SHIFT_MIN_STEPS = 20
VEL_STUCK_HISTORY_LEN = 5
SLOT_SHIFT_DELTA_Z = 2.0


def apf_repulsive_force(
    drone_xy: np.ndarray,
    obj_xy: np.ndarray,
    obj_radius: float,
    influence: float = 3.0,
    gain: float = 2.0,
) -> np.ndarray:
    diff = drone_xy - obj_xy
    dist = float(np.linalg.norm(diff))
    surface = dist - obj_radius
    if surface > influence:
        return np.zeros(2, dtype=np.float32)
    if surface < 0.05:
        surface = 0.05
    mag = gain * (1.0 / surface - 1.0 / influence) / (surface**2)
    return (mag * diff / (dist + 1e-6)).astype(np.float32)


def _obstacle_visible_euclidean(
    drone_xy: np.ndarray, obs: np.ndarray, max_range: float = LIDAR_MAX_RANGE
) -> bool:
    return float(np.linalg.norm(drone_xy - obs[:2])) - obs[2] <= max_range


def _obstacle_surface_distance(global_pos: np.ndarray, obs: np.ndarray) -> float:
    dp = global_pos[:2]
    return float(np.linalg.norm(dp - obs[:2]) - obs[2])


def _obstacle_visible_teacher(
    global_pos: np.ndarray,
    yaw: float,
    obs: np.ndarray,
    max_range: float = LIDAR_MAX_RANGE,
) -> bool:
    """Teacher APF gate: planar lidar hit OR within sensor range by surface distance."""
    if _obstacle_visible_planar_lidar(global_pos, yaw, obs, max_range):
        return True
    surf = _obstacle_surface_distance(global_pos, obs)
    return 0.0 <= surf <= max_range


def _obstacle_visible_planar_lidar(
    global_pos: np.ndarray,
    yaw: float,
    obs: np.ndarray,
    max_range: float = LIDAR_MAX_RANGE,
) -> bool:
    """True if the 16-ray planar lidar would detect this obstacle (matches build_drone_frame)."""
    dp = global_pos[:2]
    oc = np.asarray(obs[:2], dtype=np.float32)
    rh = float(obs[2])
    angles = np.linspace(0, 2 * np.pi, LIDAR_RAYS, endpoint=False) + yaw
    rays = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    w = oc - dp
    for ray in rays:
        t = float(np.dot(w, ray))
        if t > 0:
            dsq = float(np.sum(w**2) - t**2)
            if dsq <= rh**2:
                dd = t - float(np.sqrt(rh**2 - dsq))
                if 0 < dd <= max_range:
                    return True
    return False


def compute_apf_setpoint_sensor_gated(
    drone_pos: np.ndarray,
    original_sp: np.ndarray,
    obstacles: np.ndarray,
    other_positions: List[np.ndarray],
    drone_radius: float = 0.3,
    obs_influence: float = 3.0,
    drone_influence: float = 2.0,
    max_range: float = LIDAR_MAX_RANGE,
    yaw: float = 0.0,
) -> np.ndarray:
    force = np.zeros(2, dtype=np.float32)
    dp = drone_pos[:2]
    for obs in obstacles:
        if not _obstacle_visible_teacher(drone_pos, yaw, obs, max_range):
            continue
        force += apf_repulsive_force(dp, obs[:2], float(obs[2]), obs_influence, 2.0)
    for op in other_positions:
        if float(np.linalg.norm(dp - op[:2])) > max_range:
            continue
        force += apf_repulsive_force(dp, op[:2], drone_radius, drone_influence, 1.5)
    mag = float(np.linalg.norm(force))
    if mag > 2.0:
        force = force / mag * 2.0
    sp = original_sp.copy()
    sp[0] += force[0]
    sp[1] += force[1]
    return sp


def build_digit_setpoints(
    cfg: EpisodeConfig,
    start_pos: np.ndarray,
    start_orn: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """setpoints (N,4), assignment (N,), naive_offsets (N,3)."""
    n = cfg.num_drones
    slots = build_naive_slots(cfg, start_pos)
    dist = np.linalg.norm(start_pos[:, None, :2] - slots[None, :, :2], axis=2)
    _, col_ind = linear_sum_assignment(dist)
    alt = float(np.mean(start_pos[:, 2]))
    offsets = np.zeros((n, 3), dtype=np.float32)
    offsets[:, :2] = slots[:, :2] - np.mean(start_pos[:, :2], axis=0)
    offsets[:, 2] = 0.0

    setpoints = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        si = int(col_ind[i])
        setpoints[i, 0] = slots[si, 0]
        setpoints[i, 1] = slots[si, 1]
        setpoints[i, 2] = start_orn[i, 2]
        setpoints[i, 3] = alt
    return setpoints, col_ind.astype(np.int64), offsets


def planar_lidar(
    global_pos: np.ndarray,
    yaw: float,
    obstacles: np.ndarray,
    num_rays: int = LIDAR_RAYS,
    max_range: float = LIDAR_MAX_RANGE,
) -> np.ndarray:
    lidar = np.full(num_rays, max_range, dtype=np.float32)
    if obstacles.size == 0:
        return lidar
    dp = global_pos[:2]
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False) + yaw
    rays = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    oc = obstacles[:, :2]
    orr = obstacles[:, 2]
    for i, ray in enumerate(rays):
        w = oc - dp
        t = np.dot(w, ray)
        hm = t > 0
        if np.any(hm):
            wh, th, rh = w[hm], t[hm], orr[hm]
            dsq = np.sum(wh**2, axis=1) - th**2
            v = dsq <= rh**2
            if np.any(v):
                dd = th[v] - np.sqrt(rh[v] ** 2 - dsq[v])
                dd = dd[dd > 0]
                if len(dd) > 0:
                    lidar[i] = min(max_range, float(np.min(dd)))
    return lidar


def build_drone_frame(
    global_pos: np.ndarray,
    global_euler: np.ndarray,
    global_lin_vel: np.ndarray,
    global_ang_vel: np.ndarray,
    setpoint: np.ndarray,
    formation_one_hot: np.ndarray,
    obstacles: np.ndarray,
    integral_pos_err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(global_euler))).reshape(
        3, 3
    )
    local_lin_vel = R.T @ global_lin_vel
    local_ang_vel = R.T @ global_ang_vel
    tgt_pos = np.array([setpoint[0], setpoint[1], setpoint[3]], dtype=np.float32)
    tgt_yaw = setpoint[2]
    global_pos_err = tgt_pos - global_pos
    local_pos_err = R.T @ global_pos_err
    yaw_err = (tgt_yaw - global_euler[2] + np.pi) % (2 * np.pi) - np.pi
    lidar = planar_lidar(global_pos, global_euler[2], obstacles)
    frame = np.concatenate(
        [
            local_lin_vel,
            local_ang_vel,
            lidar,
            local_pos_err,
            [yaw_err],
            [global_pos[2], 10.0 - global_pos[2]],
            integral_pos_err,
            formation_one_hot,
        ]
    ).astype(np.float32)
    assert frame.shape[0] == RAW_FRAME_DIM
    return frame, global_pos.copy(), local_pos_err.copy()


def spawn_spheres(obstacles: ObstacleConfig, client_id: int) -> None:
    for pos, r in zip(obstacles.positions, obstacles.radii):
        col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=float(r), physicsClientId=client_id
        )
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=float(r),
            rgbaColor=[1, 0, 0, 0.5],
            physicsClientId=client_id,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[float(pos[0]), float(pos[1]), float(pos[2])],
            physicsClientId=client_id,
        )


def _nearest_obstacle_along(
    drone_pos: np.ndarray,
    goal_dir: np.ndarray,
    obstacles: np.ndarray,
) -> Tuple[float, np.ndarray]:
    if obstacles.size == 0:
        return float("inf"), np.zeros(2, dtype=np.float32)
    best = float("inf")
    best_dir = np.zeros(2, dtype=np.float32)
    for obs in obstacles:
        to_obs = obs[:2] - drone_pos[:2]
        along = float(np.dot(to_obs, goal_dir))
        if 0 < along < best:
            perp = float(np.linalg.norm(to_obs - along * goal_dir))
            if perp < obs[2] + 0.3:
                best = along
                best_dir = to_obs / (np.linalg.norm(to_obs) + 1e-6)
    return best, best_dir


def check_stuck_on_slot(
    local_lin_vel: np.ndarray,
    drone_pos: np.ndarray,
    target_pos: np.ndarray,
    obstacles: np.ndarray,
    vel_history: deque,
    *,
    strict: bool = False,
) -> bool:
    """World-frame obstacle ray; velocity window with optional strict shift gating."""
    vel_history.append(local_lin_vel.copy())
    need = VEL_STUCK_HISTORY_LEN if strict else 2
    if len(vel_history) < need:
        return False
    speeds = [float(np.linalg.norm(v)) for v in vel_history]
    if strict and float(np.max(speeds)) > VEL_EPS:
        return False
    if speeds[-1] >= speeds[-2]:
        return False
    window = np.stack(list(vel_history), axis=0)
    err_w = target_pos[:2] - drone_pos[:2]
    pe = float(np.linalg.norm(err_w))
    if pe < STUCK_POS_ERR or pe < 1e-6:
        return False
    if strict and pe > SHIFT_MAX_SLOT_ERROR:
        return False
    goal_dir = err_w / pe
    along, obs_dir = _nearest_obstacle_along(drone_pos, goal_dir, obstacles)
    desired = np.array([err_w[0], err_w[1], 0.0], dtype=np.float32)
    along_max = SHIFT_MAX_ALONG_DIST if strict else STUCK_MAX_ALONG_DIST
    return detect_slot_blocked(
        window,
        desired,
        np.array([obs_dir[0], obs_dir[1], 0.0], dtype=np.float32),
        along,
        vel_eps=VEL_EPS,
        max_along_dist=along_max,
    )


LIDAR_CRITICAL_THRESH = 4.0


def _thirds_snapshot_keep(T: int, shift_steps: set) -> set:
    """Every 2 steps in first/last third, every 3 in middle; always keep last + shift steps."""
    lo, hi = T // 3, (2 * T) // 3
    keep: set = set()
    for s in range(T):
        if s < lo or s >= hi:
            if s % 2 == 0:
                keep.add(s)
        else:
            if s % 3 == 0:
                keep.add(s)
    if T > 0:
        keep.add(T - 1)
    keep.update(shift_steps)
    return keep


def _lidar_near_snapshot_keep(
    buf: List[dict],
    obstacles_lidar: np.ndarray,
    threshold: float = LIDAR_CRITICAL_THRESH,
) -> set:
    """Always keep timesteps where any drone sees an obstacle within threshold."""
    keep: set = set()
    if obstacles_lidar.size == 0:
        return keep
    for entry in buf:
        for gp, ge, _, _ in entry["pre"]:
            lidar = planar_lidar(gp, float(ge[2]), obstacles_lidar)
            if float(np.min(lidar)) < threshold:
                keep.add(int(entry["step"]))
                break
    return keep


def _build_graph_from_step(
    entry: dict,
    active: List[int],
    foh: np.ndarray,
    obstacles_lidar: np.ndarray,
    communication_radius: float,
    prev_feats: dict,
    cfg: EpisodeConfig,
    ep_idx: int,
) -> Data:
    """Build one PyG graph from a buffered rollout step."""
    setpoints = entry["setpoints"]
    pre = entry["pre"]
    next_pos = entry["next_pos"]
    next_euler = entry["next_euler"]
    n = len(active)

    cur_pos = [pre[i][0] for i in range(n)]
    cur_euler = [pre[i][1] for i in range(n)]
    gvels = [pre[i][2] for i in range(n)]

    labels = []
    for i in range(n):
        gp, ge, _, _ = pre[i]
        np_ = next_pos[i]
        ne_ = next_euler[i]
        disp = np_ - gp
        R = np.array(
            p.getMatrixFromQuaternion(p.getQuaternionFromEuler(ge))
        ).reshape(3, 3)
        ld = R.T @ disp
        dy = (ne_[2] - ge[2] + np.pi) % (2 * np.pi) - np.pi
        labels.append(np.concatenate([ld, [dy]]).astype(np.float32))

    states = []
    for i in range(n):
        gp, ge, gv, av = pre[i]
        ipe = entry["ipe"][i]
        # Raw assigned slot setpoints in features; labels remain APF rollout displacement.
        frame, _, _ = build_drone_frame(
            gp, ge, gv, av, setpoints[i], foh, obstacles_lidar, ipe
        )
        states.append(frame)

    stacked = []
    for i, di in enumerate(active):
        prev = prev_feats.get(di, np.zeros(RAW_FRAME_DIM, dtype=np.float32))
        stacked.append(np.concatenate([states[i], prev]))
        prev_feats[di] = states[i].copy()

    edges, eattrs = _build_edges(
        np.array(cur_pos), np.array(gvels), np.array(cur_euler), communication_radius
    )
    x = torch.as_tensor(np.array(stacked), dtype=torch.float32)
    target = torch.as_tensor(np.array(labels), dtype=torch.float32)
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        ea = torch.as_tensor(np.array(eattrs), dtype=torch.float32)
    else:
        ei = torch.empty((2, 0), dtype=torch.long)
        ea = torch.empty((0, 7), dtype=torch.float32)

    shift_label = torch.tensor([1.0 if entry["shift_fired"] else 0.0], dtype=torch.float32)
    return Data(
        x=x,
        target=target,
        edge_index=ei,
        edge_attr=ea,
        pos=torch.as_tensor(np.array(cur_pos), dtype=torch.float32),
        formation_id=torch.tensor(cfg.digit, dtype=torch.long),
        episode_id=torch.tensor(ep_idx, dtype=torch.long),
        step_idx=torch.tensor(entry["step"], dtype=torch.long),
        num_drones=torch.tensor(n, dtype=torch.long),
        obstacles=torch.as_tensor(obstacles_lidar, dtype=torch.float32),
        shift_label=shift_label,
    )


def simulate_setpoint_episode(
    ep_idx: int,
    split: str,
    cfg: EpisodeConfig,
    max_steps: int = 400,
    save_interval: int = 5,  # unused; kept for API compatibility
    communication_radius: float = COMM_RADIUS_DEFAULT,
    graphical: bool = False,
) -> Tuple[str, List[Data]]:
    start_pos, start_orn = sample_initial_state(cfg)
    setpoints, assignment, _offsets = build_digit_setpoints(cfg, start_pos, start_orn)
    slots = build_naive_slots(cfg, start_pos)
    obs_cfg = build_obstacles_for_episode(cfg, start_pos, slots, assignment)
    obstacles_lidar = obstacles_to_lidar_array(obs_cfg)

    env = Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        drone_type="quadx",
        render=graphical,
    )
    env.set_mode(7)
    client = env._client
    if obstacles_lidar.size > 0:
        spawn_spheres(obs_cfg, client)
        env.register_all_new_bodies()

    n = cfg.num_drones
    active = list(range(n))
    foh = build_digit_one_hot(cfg.digit)
    int_bufs = {i: deque(maxlen=INTEGRAL_WINDOW) for i in range(n)}
    vel_hist = {i: deque(maxlen=VEL_STUCK_HISTORY_LEN) for i in range(n)}
    buf: List[dict] = []
    shift_steps: set = set()
    converged_count = 0
    did_shift = False

    for step in range(max_steps):
        pre: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for i, di in enumerate(active):
            st = env.drones[di].state
            gp = np.array(st[3], copy=True)
            ge = np.array(st[1], copy=True)
            gv = np.array(st[2], copy=True)
            av = np.array(st[0], copy=True)
            pre.append((gp, ge, gv, av))

        stuck_any = False
        shift_fired = False
        for i, di in enumerate(active):
            gp, ge, gv, _av = pre[i]
            tgt = np.array([setpoints[i, 0], setpoints[i, 1], setpoints[i, 3]])
            R = np.array(
                p.getMatrixFromQuaternion(p.getQuaternionFromEuler(ge))
            ).reshape(3, 3)
            if check_stuck_on_slot(
                R.T @ gv, gp, tgt, obstacles_lidar, vel_hist[di], strict=True
            ):
                stuck_any = True

        if (
            step >= SHIFT_MIN_STEPS
            and stuck_any
            and obstacles_lidar.size > 0
            and not did_shift
        ):
            slots = synchronized_slot_shift(slots, obs_cfg, delta_z=SLOT_SHIFT_DELTA_Z)
            shift_fired = True
            did_shift = True
            shift_steps.add(step)
            for i in range(n):
                si = int(assignment[i])
                setpoints[i, 0] = slots[si, 0]
                setpoints[i, 1] = slots[si, 1]
                setpoints[i, 3] = slots[si, 2]
            for hist in vel_hist.values():
                hist.clear()

        others_cache = {di: np.array(env.drones[di].state[3]) for di in active}
        frame_setpoints_step: List[np.ndarray] = []
        for i, di in enumerate(active):
            _gp, ge, _gv, _av = pre[i]
            others = [others_cache[j] for j, dj in enumerate(active) if dj != di]
            mod = compute_apf_setpoint_sensor_gated(
                others_cache[di],
                setpoints[i],
                obstacles_lidar,
                others,
                yaw=float(ge[2]),
            )
            frame_setpoints_step.append(mod.copy())
            env.set_setpoint(di, mod)

        env.step()

        next_pos: List[np.ndarray] = []
        next_euler: List[np.ndarray] = []
        for i, di in enumerate(active):
            st = env.drones[di].state
            next_pos.append(np.array(st[3], copy=True))
            next_euler.append(np.array(st[1], copy=True))

        ipe_list: List[np.ndarray] = []
        for i, di in enumerate(active):
            gp, ge, _, _ = pre[i]
            tgt_p = np.array([setpoints[i, 0], setpoints[i, 1], setpoints[i, 3]])
            R = np.array(
                p.getMatrixFromQuaternion(p.getQuaternionFromEuler(ge))
            ).reshape(3, 3)
            lpe = R.T @ (tgt_p - gp)
            int_bufs[di].append(lpe.copy())
            if len(int_bufs[di]) > 0:
                ipe = np.mean(list(int_bufs[di]), axis=0).astype(np.float32)
            else:
                ipe = np.zeros(3, dtype=np.float32)
            ipe_list.append(ipe)

        buf.append(
            {
                "step": step,
                "shift_fired": shift_fired,
                "setpoints": setpoints.copy(),
                "frame_setpoints": frame_setpoints_step,
                "pre": pre,
                "next_pos": next_pos,
                "next_euler": next_euler,
                "ipe": ipe_list,
            }
        )

        errors = []
        for i, di in enumerate(active):
            gp = next_pos[i]
            tgt = np.array([setpoints[i, 0], setpoints[i, 1], setpoints[i, 3]])
            errors.append(float(np.linalg.norm(gp - tgt)))
        if errors and max(errors) < CONV_THRESHOLD:
            converged_count += 1
        else:
            converged_count = 0

        if converged_count >= CONV_STEPS:
            break

    env.disconnect()

    T = len(buf)
    graphs: List[Data] = []
    if T == 0:
        return split, graphs

    keep = _thirds_snapshot_keep(T, shift_steps)
    keep |= _lidar_near_snapshot_keep(buf, obstacles_lidar)
    prev_feats: dict = {}
    for s in sorted(keep):
        graphs.append(
            _build_graph_from_step(
                buf[s],
                active,
                foh,
                obstacles_lidar,
                communication_radius,
                prev_feats,
                cfg,
                ep_idx,
            )
        )

    return split, graphs


def _build_edges(positions, global_vels, eulers, comm_radius):
    edges, attrs = [], []
    n = len(positions)
    for i in range(n):
        Ri = np.array(
            p.getMatrixFromQuaternion(p.getQuaternionFromEuler(eulers[i]))
        ).reshape(3, 3)
        for j in range(n):
            if i == j:
                continue
            rp = positions[j] - positions[i]
            d = float(np.linalg.norm(rp))
            if d <= comm_radius:
                rv = global_vels[j] - global_vels[i]
                edges.append([i, j])
                attrs.append(np.concatenate([Ri.T @ rp, [d], Ri.T @ rv]))
    return edges, attrs


def rollout_from_seed(
    ep_idx: int,
    split: str,
    seed: int,
    num_drones: int,
    **kwargs,
) -> Tuple[str, List[Data]]:
    cfg = make_episode_config(seed=seed, num_drones=num_drones)
    return simulate_setpoint_episode(ep_idx, split, cfg, **kwargs)
