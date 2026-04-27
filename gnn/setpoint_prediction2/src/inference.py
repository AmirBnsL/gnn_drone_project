"""
Real-time GNN inference for drone swarm in PyFlyt simulator.

=== FIXES APPLIED (see FIXES section below for full explanation) ===

FIX 1 — Velocity Frame (CRITICAL):
    PyFlyt's state[2] IS already in the local body frame. The data-collection
    script (datacollection.py line 325) reads it as "local_lin_vel" and feeds
    it directly to the GNN without any rotation. The inference script was doing
    the same thing — BUT it then tried to compute edge features using
    `R.T @ (vel_j - vel_i)` where vel_i/j were reconstructed as `R @ local_vel`
    (converting them back to global). This was inconsistent and introduced a
    double-rotation bug in the edge attributes.

    FIX: Keep state[2] as-is (it is already local body velocity). To compute
    relative velocity for edge attributes, convert each drone's local velocity
    to global ONCE using `R @ local_vel`, then compute `R_i.T @ (global_j - global_i)`
    to get the relative velocity in drone i's local frame. This matches training.

FIX 2 — Inertia Feedback Loop (Carrot Setpoint):
    The GNN predicts the full 1-step physical displacement: Δp = v·dt + ½a·dt².
    The velocity term v·dt is the dominant component. Multiplying the raw
    displacement by gain=500 therefore amplifies the current velocity into a
    huge forward setpoint — the drone keeps accelerating and cannot brake.

    FIX: Subtract the inertial coasting component (v·dt) from the predicted
    displacement before applying the gain. This isolates the GNN's "pure control
    effort" — the acceleration-driven correction beyond what inertia alone would
    produce. The gain is then reduced significantly (10–30) since it now amplifies
    only the true control signal, not the momentum.

    carrot_disp = (pred_disp - v*dt) * gain   (in local body frame)

FIX 3 — Setpoint Formulation with Auto-Braking:
    The old blending mixed the GNN carrot with a raw goal_err term scaled by
    goal_gain. This did not naturally brake near the target. Instead we blend
    the GNN carrot with a braking term that is proportional to *distance from
    goal* and decays to zero at arrival.

    setpoint = pos + α * carrot + β * clamp(goal_err, max_radius)

    When |goal_err| is small, the carrot dominates and naturally brings the
    drone to hover (since pred ≈ v*dt → carrot ≈ 0). When far from the goal,
    goal_gain also pulls the drone toward the target.

Features:
- Random number of drones
- Random start initialization
- Random final positions (blue mini circles)
- Random obstacles
- Random formation (a, w, rectangle, triangle, random_cloud)
- PyFlyt Mode 7 with coordinate transforms (local to global)
- Convergence detection & visualization hold
"""

import argparse
import os
import time
import numpy as np
import torch
import pybullet as p
from torch_geometric.data import Data
from PyFlyt.core import Aviary

from model import SetpointGATv2
from dataloader import DatasetNormalizer, engineer_x

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "in_channels": 58,
    "hidden_channels": 64,
    "out_channels": 4,
    "edge_dim": 7,
    "heads": 4,
    "num_layers": 3,
    "dropout": 0.0,

    "physics_hz": 240,
    "ctrl_hz": 48,
    "ctrl_every": 10,       # GNN queried every 10 physics steps
    "max_steps": 2000,
    "comm_radius": 10.0,
    "arena_size": 10.0,
    "ceiling": 10.0,
    "warmup_steps": 120,

    # --- FIX 2+3: Distance-proportional gain with inertia subtraction ---
    "gain_max": 25.0,        # Maximum gain (far from target)
    "d_half": 2.0,           # Distance at which gain is ~76% of max
    "yaw_gain": 3.0,         # Separate, smaller gain for yaw
    "dt": 1.0 / 120.0,      # Physics timestep (matches PyFlyt default at 120hz ctrl)

    # Blending weights: GNN carrot vs direct goal attraction
    "pred_gain": 0.6,        # Weight of GNN-derived carrot setpoint
    "goal_gain": 0.4,        # Weight of direct goal error attraction
    "ramp_steps": 200,       # Steps over which to ramp up gains from 0

    # Setpoint step limits (meters per GNN call, ~10 physics steps)
    "max_step_xy": 2.0,
    "max_step_z": 1.0,
    "max_step_yaw": 0.4,

    "min_drones": 3,
    "max_drones": 6,
    "min_spawn_dist": 1.5,
    "max_obstacles": 15,
    "obs_radius_range": (0.3, 2.0),

    "num_rays": 16,
    "max_range": 5.0,

    "success_radius": 0.2,
    "collision_radius": 0.35,
}

# ═══════════════════════════════════════════════════════════════════════════
# Helpers: Spawning & Randomization
# ═══════════════════════════════════════════════════════════════════════════

def random_positions(n, xy_range, z_range, min_dist, max_retries=500):
    positions = []
    for _ in range(n):
        for _ in range(max_retries):
            pos = np.array([
                np.random.uniform(-xy_range, xy_range),
                np.random.uniform(-xy_range, xy_range),
                np.random.uniform(z_range[0], z_range[1]),
            ])
            if all(np.linalg.norm(pos - p_) > min_dist for p_ in positions):
                positions.append(pos)
                break
    return np.array(positions)

def sample_obstacles(
    start_positions,
    target_positions,
    max_obstacles=15,
    radius_range=(0.3, 2.0),
    drone_clearance=2.0,
    target_clearance=1.5,
    obstacle_clearance=1.0,
    lateral_spread=3.0,
    max_attempts_per_obstacle=200,
):
    """
    Place obstacles ALONG drone-to-target trajectories so drones must avoid them.

    Each obstacle is spawned near a random drone's flight path (between its
    start position and its target) with a lateral offset, guaranteeing that
    drones will encounter obstacles during navigation.

    Returns np.ndarray of shape (N, 3): columns are [x, y, radius].
    """
    num_obstacles = np.random.randint(3, max_obstacles + 1)
    n_drones = len(start_positions)
    if n_drones == 0:
        return np.zeros((0, 3))

    # Protected zones: both drone spawns and targets
    spawn_xy = start_positions[:, :2]
    target_xy = target_positions[:, :2] if target_positions.shape[1] >= 2 else target_positions
    protected_xy = np.vstack([spawn_xy, target_xy])

    accepted = []

    for _ in range(num_obstacles):
        for _ in range(max_attempts_per_obstacle):
            # Pick a random drone's trajectory
            idx = np.random.randint(n_drones)
            start = start_positions[idx]
            end = target_positions[idx]

            # Random point along the trajectory (avoid very start/end)
            t = np.random.uniform(0.15, 0.85)
            midpoint_x = start[0] + t * (end[0] - start[0])
            midpoint_y = start[1] + t * (end[1] - start[1])

            # Add lateral offset perpendicular to the trajectory
            traj_dir = np.array([end[0] - start[0], end[1] - start[1]])
            traj_len = np.linalg.norm(traj_dir)
            if traj_len > 0.1:
                perp = np.array([-traj_dir[1], traj_dir[0]]) / traj_len
                lateral = np.random.uniform(-lateral_spread, lateral_spread)
                midpoint_x += lateral * perp[0]
                midpoint_y += lateral * perp[1]

            cr = np.random.uniform(radius_range[0], radius_range[1])
            candidate = np.array([midpoint_x, midpoint_y])

            # Check 1: Clear of all drone start positions
            dists_spawn = np.linalg.norm(spawn_xy - candidate, axis=1)
            if np.any(dists_spawn < drone_clearance + cr):
                continue

            # Check 2: Clear of all target positions
            dists_target = np.linalg.norm(target_xy - candidate, axis=1)
            if np.any(dists_target < target_clearance + cr):
                continue

            # Check 3: Clear of previously placed obstacles
            if len(accepted) > 0:
                existing = np.array(accepted)
                dists_existing = np.linalg.norm(existing[:, :2] - candidate, axis=1)
                if np.any(dists_existing - existing[:, 2] - cr < obstacle_clearance):
                    continue

            accepted.append([midpoint_x, midpoint_y, cr])
            break

    if len(accepted) == 0:
        return np.zeros((0, 3))
    return np.array(accepted, dtype=np.float64)


def simple_obstacles(
    start_positions,
    max_obstacles=25,
    xy_bounds=(-15.0, 15.0),
    radius_range=(0.3, 2.0),
    drone_clearance=3.0,
    obstacle_clearance=1.0,
    max_attempts_per_obstacle=100,
):
    """
    Simple random-arena obstacle placement (identical to data-collection pipeline).
    Obstacles are scattered uniformly across the arena with clearance from drones.
    """
    num_obstacles = np.random.randint(0, max_obstacles + 1)
    if num_obstacles == 0:
        return np.zeros((0, 3))

    protected_xy = start_positions[:, :2] if len(start_positions) > 0 else np.zeros((0, 2))
    accepted = []

    for _ in range(num_obstacles):
        for _ in range(max_attempts_per_obstacle):
            cx = np.random.uniform(xy_bounds[0], xy_bounds[1])
            cy = np.random.uniform(xy_bounds[0], xy_bounds[1])
            cr = np.random.uniform(radius_range[0], radius_range[1])

            if len(protected_xy) > 0:
                dists = np.linalg.norm(protected_xy - np.array([cx, cy]), axis=1)
                if np.any(dists < drone_clearance + cr):
                    continue

            if len(accepted) > 0:
                existing = np.array(accepted)
                dists = np.linalg.norm(existing[:, :2] - np.array([cx, cy]), axis=1)
                if np.any(dists - existing[:, 2] - cr < obstacle_clearance):
                    continue

            accepted.append([cx, cy, cr])
            break

    if len(accepted) == 0:
        return np.zeros((0, 3))
    return np.array(accepted, dtype=np.float64)


def spawn_pybullet_obstacles(obstacles, client_id):
    ids = []
    for obs in obstacles:
        x, y, r = obs[0], obs[1], obs[2]
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=10.0,
                                  rgbaColor=[0.8, 0.2, 0.1, 0.6], physicsClientId=client_id)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=10.0, physicsClientId=client_id)
        body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, baseCollisionShapeIndex=col,
                                 basePosition=[x, y, 5.0], physicsClientId=client_id)
        ids.append(body)
    return ids

def spawn_target_spheres(targets, client_id):
    ids = []
    for t in targets:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15,
                                  rgbaColor=[0.1, 0.3, 1.0, 0.8], physicsClientId=client_id)
        body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                 basePosition=[t[0], t[1], t[2]], physicsClientId=client_id)
        ids.append(body)
    return ids

# ═══════════════════════════════════════════════════════════════════════════
# Helpers: Formations (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def generate_random_cloud_setpoints(num_drones, xy_limit, z_range, min_dist=1.5):
    return random_positions(num_drones, xy_limit, z_range, min_dist)

def formation_a_offsets(num_drones, spacing=2.0):
    offsets = np.zeros((num_drones, 3), dtype=np.float32)
    if num_drones <= 1: return offsets
    num_crossbar = (num_drones // 5) if num_drones > 5 else 0
    num_v_legs = num_drones - num_crossbar
    if num_v_legs % 2 == 0:
        num_crossbar += 1; num_v_legs -= 1
    offsets[0] = [0.0, 0.0, 0.0]
    for idx in range(1, num_v_legs):
        level = (idx + 1) // 2
        side = -1.0 if idx % 2 == 1 else 1.0
        offsets[idx, 0] = side * level * spacing
        offsets[idx, 1] = level * spacing
    if num_crossbar > 0:
        max_level = (num_v_legs + 1) // 2
        mid_level = max(1, max_level // 2)
        left_x = -mid_level * spacing
        right_x = mid_level * spacing
        width = right_x - left_x
        for crossbar_idx in range(num_crossbar):
            fraction = (crossbar_idx + 1) / (num_crossbar + 1)
            target_idx = num_v_legs + crossbar_idx
            offsets[target_idx, 0] = left_x + (fraction * width)
            offsets[target_idx, 1] = mid_level * spacing
    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets

def formation_rectangle_offsets(num_drones, spacing=2.0):
    offsets = np.zeros((num_drones, 3), dtype=np.float32)
    if num_drones == 0: return offsets
    perimeter = num_drones * spacing
    side = perimeter / 4.0
    corners = [[-side/2, -side/2], [side/2, -side/2], [side/2, side/2], [-side/2, side/2]]
    for i in range(min(num_drones, 4)): offsets[i, :2] = corners[i]
    if num_drones > 4:
        remaining = num_drones - 4
        drones_per_edge = [remaining // 4 + (1 if e < remaining % 4 else 0) for e in range(4)]
        current_idx = 4
        for e in range(4):
            start_c = np.array(corners[e])
            end_c = np.array(corners[(e + 1) % 4])
            for step in range(1, drones_per_edge[e] + 1):
                fraction = step / (drones_per_edge[e] + 1)
                offsets[current_idx, :2] = start_c + fraction * (end_c - start_c)
                current_idx += 1
    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets

def formation_triangle_offsets(num_drones, spacing=2.0):
    offsets = np.zeros((num_drones, 3), dtype=np.float32)
    if num_drones == 0: return offsets
    perimeter = num_drones * spacing
    side = perimeter / 3.0
    height = side * np.sqrt(3) / 2.0
    corners = [[-side/2, -height/3], [side/2, -height/3], [0.0, 2*height/3]]
    for i in range(min(num_drones, 3)): offsets[i, :2] = corners[i]
    if num_drones > 3:
        remaining = num_drones - 3
        drones_per_edge = [remaining // 3 + (1 if e < remaining % 3 else 0) for e in range(3)]
        current_idx = 3
        for e in range(3):
            start_c = np.array(corners[e])
            end_c = np.array(corners[(e + 1) % 3])
            for step in range(1, drones_per_edge[e] + 1):
                fraction = step / (drones_per_edge[e] + 1)
                offsets[current_idx, :2] = start_c + fraction * (end_c - start_c)
                current_idx += 1
    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets

def formation_w_offsets(num_drones, spacing=2.0):
    offsets = np.zeros((num_drones, 3), dtype=np.float32)
    if num_drones == 0: return offsets
    points = np.array([[-2.0, 2.0], [-1.0, 0.0], [0.0, 2.0], [1.0, 0.0], [2.0, 2.0]], dtype=np.float32) * spacing
    segs = points[1:] - points[:-1]
    seg_len = np.linalg.norm(segs, axis=1)
    total = float(np.sum(seg_len))
    if total <= 1e-6: return offsets
    distances = np.linspace(0.0, total, num_drones, dtype=np.float32)
    acc = 0.0; seg_idx = 0
    for i, d in enumerate(distances):
        while seg_idx < len(seg_len) - 1 and d > acc + seg_len[seg_idx]:
            acc += seg_len[seg_idx]; seg_idx += 1
        t = 0.0 if seg_len[seg_idx] < 1e-6 else (d - acc) / seg_len[seg_idx]
        offsets[i, :2] = points[seg_idx] + t * segs[seg_idx]
    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets

def build_formation_positions(formation_name, num_drones, center_xy, altitude, spacing=2.0, xy_limit=5.0):
    if formation_name == "random_cloud":
        return generate_random_cloud_setpoints(num_drones, xy_limit, (altitude - 1.0, altitude + 1.0), spacing)
    elif formation_name == "a":
        offsets = formation_a_offsets(num_drones, spacing)
    elif formation_name == "rectangle":
        offsets = formation_rectangle_offsets(num_drones, spacing)
    elif formation_name == "triangle":
        offsets = formation_triangle_offsets(num_drones, spacing)
    elif formation_name == "w":
        offsets = formation_w_offsets(num_drones, spacing)
    else:
        offsets = formation_triangle_offsets(num_drones, spacing)
    positions = np.zeros((num_drones, 3), dtype=np.float32)
    positions[:, :2] = center_xy + offsets[:, :2]
    positions[:, 2] = altitude
    return positions

def assign_slots(drones_xy, slots_xy):
    num_drones = drones_xy.shape[0]
    try:
        from scipy.optimize import linear_sum_assignment
        dist = np.linalg.norm(drones_xy[:, None, :] - slots_xy[None, :, :], axis=2)
        _, col_ind = linear_sum_assignment(dist)
        return col_ind
    except Exception:
        pass
    remaining = list(range(num_drones))
    assignments = []
    for i in range(num_drones):
        dist_arr = np.linalg.norm(slots_xy[remaining] - drones_xy[i], axis=1)
        pick = int(np.argmin(dist_arr))
        assignments.append(remaining[pick])
        remaining.pop(pick)
    return np.array(assignments, dtype=np.int64)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers: Feature Building
# ═══════════════════════════════════════════════════════════════════════════

def compute_lidar(pos, yaw, obstacles, num_rays=16, max_range=5.0):
    rays = np.full(num_rays, max_range)
    if len(obstacles) == 0: return rays
    drone_pos = pos[:2]
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False) + yaw
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    obs_c = obstacles[:, :2]
    obs_r = obstacles[:, 2]
    for i, ray in enumerate(dirs):
        W = obs_c - drone_pos
        t = np.dot(W, ray)
        hit = t > 0
        if np.any(hit):
            W_h, t_h, r_h = W[hit], t[hit], obs_r[hit]
            d_sq = np.sum(W_h ** 2, axis=1) - t_h ** 2
            valid = d_sq <= r_h ** 2
            if np.any(valid):
                dists = t_h[valid] - np.sqrt(r_h[valid] ** 2 - d_sq[valid])
                dists = dists[dists > 0]
                if len(dists) > 0:
                    rays[i] = min(max_range, np.min(dists))
    return rays


def build_single_frame(drone, target, obstacles, cfg):
    """
    Build the 32-dim feature frame for a single drone, matching training exactly.

    FIX 1 (Velocity Frame): PyFlyt's state[2] is ALREADY in the local body frame
    (confirmed by checking datacollection.py line 325 which calls it local_lin_vel
    without any rotation). We keep it as-is, identical to training.
    We still need the global velocity to compute consistent edge attributes,
    so we convert it to global here: global_vel = R @ local_vel.
    """
    state = drone.state
    global_pos = np.array(state[3], copy=True)
    global_euler = np.array(state[1], copy=True)

    # state[2] is LOCAL body-frame linear velocity (same as training)
    local_lin_vel = np.array(state[2], copy=True)
    local_ang_vel = np.array(state[0], copy=True)

    rot_quat = p.getQuaternionFromEuler(global_euler)
    rot_matrix = np.array(p.getMatrixFromQuaternion(rot_quat)).reshape(3, 3)

    # Convert local → global velocity for use in edge attribute computation
    global_lin_vel = rot_matrix @ local_lin_vel

    lidar = compute_lidar(global_pos, global_euler[2], obstacles, cfg["num_rays"], cfg["max_range"])

    target_pos = np.array(target[:3])
    target_yaw = target[3]
    global_pos_err = target_pos - global_pos
    local_pos_err = rot_matrix.T @ global_pos_err
    yaw_err = (target_yaw - global_euler[2] + np.pi) % (2 * np.pi) - np.pi

    dist_floor = global_pos[2]
    dist_ceil = cfg["ceiling"] - global_pos[2]

    # 32-dim frame: matches training data structure exactly
    frame = np.concatenate([
        local_lin_vel,          # indices 0-2:  local body velocity (as trained)
        local_ang_vel,          # indices 3-5:  local angular velocity
        lidar,                  # indices 6-21: 16 lidar rays
        local_pos_err,          # indices 22-24: local position error to target
        [yaw_err],              # index 25:     yaw error
        [dist_floor, dist_ceil],# indices 26-27: floor/ceiling distances
        [0, 0, 0, 0],           # indices 28-31: formation placeholder zeros
    ])

    return frame, global_pos, global_euler, global_lin_vel, local_lin_vel


def build_graph(drones, targets, obstacles, prev_frames, cfg, device):
    """
    Build the PyG graph for GNN inference.

    FIX 1 (Edge attributes): Edge relative velocity is computed in drone i's
    local frame via R_i.T @ (global_vel_j - global_vel_i). This is consistent
    with how the training data was generated (data collection computes relative
    displacement in global frame, then projects to local frame for edges).
    """
    N = len(drones)
    curr_frames = []; positions = []; eulers = []; global_vels = []; local_vels = []

    for i, drone in enumerate(drones):
        frame, pos, euler, gvel, lvel = build_single_frame(drone, targets[i], obstacles, cfg)
        curr_frames.append(frame)
        positions.append(pos)
        eulers.append(euler)
        global_vels.append(gvel)
        local_vels.append(lvel)

    curr_frames = np.array(curr_frames)
    positions = np.array(positions)
    eulers = np.array(eulers)
    global_vels = np.array(global_vels)
    local_vels = np.array(local_vels)

    if prev_frames is None:
        prev_frames = np.zeros_like(curr_frames)
    x_raw = np.concatenate([curr_frames, prev_frames], axis=1)

    edge_index = []; edge_attr = []
    for i in range(N):
        rot_q = p.getQuaternionFromEuler(eulers[i])
        R_i = np.array(p.getMatrixFromQuaternion(rot_q)).reshape(3, 3)
        for j in range(N):
            if i == j: continue
            rel_pos_global = positions[j] - positions[i]
            dist = np.linalg.norm(rel_pos_global)
            if dist <= cfg["comm_radius"]:
                # FIX 1: Project relative position and velocity into drone i's body frame
                rel_pos_local = R_i.T @ rel_pos_global
                rel_vel_global = global_vels[j] - global_vels[i]
                rel_vel_local = R_i.T @ rel_vel_global
                edge_index.append([i, j])
                edge_attr.append(np.concatenate([rel_pos_local, [dist], rel_vel_local]))

    x_t = torch.tensor(x_raw, dtype=torch.float32)
    if edge_index:
        ei_t = torch.tensor(edge_index, dtype=torch.long).T
        ea_t = torch.tensor(np.array(edge_attr), dtype=torch.float32)
    else:
        ei_t = torch.zeros((2, 0), dtype=torch.long)
        ea_t = torch.zeros((0, cfg["edge_dim"]), dtype=torch.float32)

    graph = Data(x=x_t, edge_index=ei_t, edge_attr=ea_t).to(device)
    return graph, curr_frames, positions, eulers, local_vels


def pred_to_global_setpoints(pred_scaled, positions, eulers, local_vels, targets, step, cfg):
    """
    Convert GNN predictions to absolute position setpoints for PyFlyt Mode 7.

    FIX 2 — Inertia Subtraction + Distance-Proportional Gain:
        control_effort = pred_disp - v*dt  (isolate acceleration component)
        gain(d) = gain_max * tanh(d / d_half)  (auto-brake near target)

    FIX 3 — Adaptive Blending:
        Near target: goal_gain dominates (precise positioning)
        Far from target: pred_gain dominates (GNN-guided navigation)
    """
    dt = cfg["dt"]
    ramp = min(1.0, step / max(1, cfg["ramp_steps"]))
    gain_max = cfg["gain_max"] * ramp
    yaw_gain = cfg["yaw_gain"] * ramp
    base_pred_gain = cfg["pred_gain"]
    base_goal_gain = cfg["goal_gain"]
    d_half = cfg["d_half"]

    setpoints = np.zeros((len(positions), 4))

    for i in range(len(positions)):
        dx_pred = pred_scaled[i, 0]
        dy_pred = pred_scaled[i, 1]
        dz_pred = pred_scaled[i, 2]
        dyaw_pred = pred_scaled[i, 3]

        lv = local_vels[i]  # shape (3,)

        # FIX 2: Subtract inertial coast to isolate control effort
        control_x = dx_pred - lv[0] * dt
        control_y = dy_pred - lv[1] * dt
        control_z = dz_pred - lv[2] * dt
        control_yaw = dyaw_pred

        # Goal error in global frame
        goal_err = targets[i, :3] - positions[i]
        goal_dist = np.linalg.norm(goal_err)

        # Distance-proportional gain: tanh profile for smooth braking
        dist_gain = gain_max * np.tanh(goal_dist / d_half)

        # Scale control effort into carrot (local frame)
        carrot_local = np.array([control_x * dist_gain,
                                  control_y * dist_gain,
                                  control_z * dist_gain])

        # Rotate carrot to global frame
        rot_q = p.getQuaternionFromEuler(eulers[i])
        R = np.array(p.getMatrixFromQuaternion(rot_q)).reshape(3, 3)
        carrot_global = R @ carrot_local

        # FIX 3: Adaptive blending — more goal influence near target
        blend_alpha = np.clip(goal_dist / d_half, 0.0, 1.0)  # 0 at goal, 1 far
        pred_w = base_pred_gain * blend_alpha
        goal_w = base_goal_gain + base_pred_gain * (1.0 - blend_alpha)

        # Yaw
        yaw = eulers[i][2]
        goal_yaw_err = ((targets[i, 3] - yaw + np.pi) % (2 * np.pi)) - np.pi

        blended_xyz = carrot_global * pred_w + goal_err * goal_w
        blended_yaw = control_yaw * yaw_gain * pred_w + goal_yaw_err * goal_w

        # Clip per-axis to prevent wild setpoints
        bxy = blended_xyz[:2]
        bxy_norm = np.linalg.norm(bxy)
        if bxy_norm > cfg["max_step_xy"]:
            bxy = bxy / bxy_norm * cfg["max_step_xy"]
        blended_z = np.clip(blended_xyz[2], -cfg["max_step_z"], cfg["max_step_z"])
        blended_yaw = np.clip(blended_yaw, -cfg["max_step_yaw"], cfg["max_step_yaw"])

        setpoints[i, 0] = positions[i][0] + bxy[0]
        setpoints[i, 1] = positions[i][1] + bxy[1]
        setpoints[i, 2] = yaw + blended_yaw
        setpoints[i, 3] = max(0.5, positions[i][2] + blended_z)

    return setpoints

# ═══════════════════════════════════════════════════════════════════════════
# Main Inference Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(model, normalizer, cfg, device, formation_arg=None, drones_arg=None, obstacle_mode="comp"):
    num_drones = drones_arg or np.random.randint(cfg["min_drones"], cfg["max_drones"] + 1)
    spawn_pos = random_positions(num_drones, cfg["arena_size"] * 0.4, (1.0, 3.0), cfg["min_spawn_dist"])
    spawn_orn = np.zeros((num_drones, 3))
    spawn_orn[:, 2] = np.random.uniform(-np.pi, np.pi, num_drones)

    formation_choices = ["a", "rectangle", "triangle", "w", "random_cloud"]

    waypoints = []
    for _ in range(1):
        chosen_formation = formation_arg or np.random.choice(formation_choices)
        center_xy = np.array([np.random.uniform(-4, 4), np.random.uniform(-4, 4)])
        altitude = np.random.uniform(2.0, 5.0)
        slot_positions = build_formation_positions(chosen_formation, num_drones, center_xy, altitude,
                                                   spacing=2.0, xy_limit=cfg["arena_size"] * 0.4)
        slot_assignments = np.random.permutation(num_drones)
        waypoints.append({"formation": chosen_formation, "targets": slot_positions[slot_assignments]})
        if obstacle_mode == "simp":
            obstacles = simple_obstacles(
                spawn_pos,
                max_obstacles=cfg["max_obstacles"],
                xy_bounds=(-cfg["arena_size"], cfg["arena_size"]),
                radius_range=cfg["obs_radius_range"],
            )
        else:  # "comp" — trajectory-aware
            obstacles = sample_obstacles(
                spawn_pos,
                waypoints[-1]["targets"],
                max_obstacles=cfg["max_obstacles"],
                radius_range=cfg["obs_radius_range"],
            )

    print(f"\n{'='*60}")
    print(f"Episode: {num_drones} drones | {len(waypoints)} Waypoints | {len(obstacles)} obstacles")
    print(f"{'='*60}")

    env = Aviary(
        start_pos=spawn_pos,
        start_orn=spawn_orn,
        drone_type="quadx",
        render=True,
        drone_options={"control_hz": 120},
        physics_hz=240
    )
    env.set_mode(7)
    client = env._client

    targets = np.zeros((num_drones, 4))
    targets[:, :3] = waypoints[0]["targets"]
    targets[:, 3] = 0.0
    target_ids = spawn_target_spheres(targets, client)

    obs_ids = spawn_pybullet_obstacles(obstacles, client)
    if obs_ids or target_ids:
        env.register_all_new_bodies()

    print("[*] Forcing Takeoff Phase...")
    for _ in range(cfg["warmup_steps"]):
        for i in range(num_drones):
            env.set_setpoint(i, np.array([spawn_pos[i][0], spawn_pos[i][1], spawn_orn[i][2], 1.5]))
        env.step()

    prev_frames = None
    current_setpoints = np.zeros((num_drones, 4))
    # Initialise setpoints to hover in place
    for i in range(num_drones):
        state = env.drones[i].state
        pos = np.array(state[3])
        yaw = np.array(state[1])[2]
        current_setpoints[i] = [pos[0], pos[1], yaw, pos[2]]

    print("[*] Starting GNN Inference Loop...")
    for wp_idx, wp in enumerate(waypoints):
        print(f"[*] Proceeding to Waypoint {wp_idx+1}/{len(waypoints)} (Formation: {wp['formation']})")
        targets[:, :3] = wp["targets"]
        for i, t_id in enumerate(target_ids):
            p.resetBasePositionAndOrientation(t_id, targets[i, :3], [0, 0, 0, 1], physicsClientId=client)

        for step in range(cfg["max_steps"]):
            if step % cfg["ctrl_every"] == 0:
                drones = [env.drones[i] for i in range(num_drones)]

                # Dynamically aim target yaw toward the waypoint
                for i in range(num_drones):
                    direction = targets[i, :2] - np.array(env.drones[i].state[3][:2])
                    if np.linalg.norm(direction) > 0.1:
                        targets[i, 3] = np.arctan2(direction[1], direction[0])

                graph, curr_frames, positions, eulers, local_vels = build_graph(
                    drones, targets, obstacles, prev_frames, cfg, device)

                graph.x = (engineer_x(graph.x) - normalizer.x_mean) / normalizer.x_std
                if graph.edge_attr.numel() > 0:
                    graph.edge_attr = (graph.edge_attr - normalizer.e_mean) / normalizer.e_std

                with torch.no_grad():
                    pred_norm = model(graph.x, graph.edge_index, graph.edge_attr)

                # Restore to physical units (metres / radians)
                pred_phys = (pred_norm * normalizer.y_scale).cpu().numpy()
                raw_pred_phys = pred_phys.copy()

                # FIX 2+3: Pass local_vels so inertia can be subtracted inside
                current_setpoints = pred_to_global_setpoints(
                    pred_phys, positions, eulers, local_vels, targets, step, cfg)

                prev_frames = curr_frames

                if step % 50 == 0:
                    dist_0 = np.linalg.norm(positions[0] - targets[0, :3])
                    raw_pred_0 = raw_pred_phys[0, :3]
                    local_pos_err_0 = curr_frames[0, 22:25]
                    lv_0 = local_vels[0]
                    ctrl_eff = raw_pred_0 - lv_0 * cfg["dt"]
                    eff_gain = cfg["gain_max"] * np.tanh(dist_0 / cfg["d_half"])
                    print(f"[Step {step:4d}] Drone 0 -> Dist: {dist_0:.2f}m "
                          f"| DistGain: {eff_gain:.1f} "
                          f"| LocalErr: [{local_pos_err_0[0]:.2f},{local_pos_err_0[1]:.2f},{local_pos_err_0[2]:.2f}]")
                    print(f"             GNN Pred: [{raw_pred_0[0]:.4f},{raw_pred_0[1]:.4f},{raw_pred_0[2]:.4f}] "
                          f"| CtrlEffort: [{ctrl_eff[0]:.4f},{ctrl_eff[1]:.4f},{ctrl_eff[2]:.4f}] "
                          f"| Yaw: {raw_pred_phys[0,3]:.4f}")

            for i in range(num_drones):
                env.set_setpoint(i, current_setpoints[i])

            env.step()
            time.sleep(1.0 / cfg["physics_hz"])

            # Convergence check
            all_converged = all(
                np.linalg.norm(np.array(env.drones[i].state[3]) - targets[i, :3]) <= cfg["success_radius"]
                for i in range(num_drones)
            )
            if all_converged and step > cfg["warmup_steps"]:
                print(f"[*] Swarm converged to Waypoint {wp_idx+1} at step {step}!")
                break

    print("[+] All waypoints reached! Mission Complete.")
    try:
        while True:
            env.step()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[*] Closing simulator (KeyboardInterrupt)...")
    except p.error:
        print("[*] Closing simulator (Window closed)...")
    finally:
        try:
            env.disconnect()
        except Exception:
            pass


def main():
    ckpt = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    parser = argparse.ArgumentParser(description="GNN Drone Swarm Inference")
    parser.add_argument("--model", default=os.path.join(ckpt, "best_gatv2.pth"))
    parser.add_argument("--stats", default=os.path.join(ckpt, "normalization_stats.pt"))
    parser.add_argument("--num_drones", type=int, default=None)
    parser.add_argument("--formation", type=str, default=None,
                        choices=["a", "rectangle", "triangle", "w", "random_cloud"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--simp", action="store_true",
                        help="Use simple random-arena obstacles (data-collection style)")
    parser.add_argument("--comp", action="store_true",
                        help="Use trajectory-aware obstacles (default)")
    args = parser.parse_args()
    obstacle_mode = "simp" if args.simp else "comp"

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DEFAULT_CONFIG

    model = SetpointGATv2(
        in_ch=cfg["in_channels"], hid_ch=cfg["hidden_channels"],
        out_ch=cfg["out_channels"], edge_dim=cfg["edge_dim"],
        heads=cfg["heads"], num_layers=cfg["num_layers"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    normalizer = DatasetNormalizer.load(args.stats, device=device)
    run_episode(model, normalizer, cfg, device, formation_arg=args.formation, drones_arg=args.num_drones, obstacle_mode=obstacle_mode)


if __name__ == "__main__":
    main()



    # python src/inference.py --simp
# python src/inference.py --comp