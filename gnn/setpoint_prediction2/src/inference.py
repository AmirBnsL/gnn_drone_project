"""
Real-time GNN inference for drone swarm in PyFlyt simulator.

Usage:
    python inference.py
    python inference.py --model ../checkpoints/best_gatv2.pth --stats ../checkpoints/normalization_stats.pt
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
    # Architecture (must match training)
    "in_channels": 58,
    "hidden_channels": 64,
    "out_channels": 4,
    "edge_dim": 7,
    "heads": 4,
    "num_layers": 3,
    "dropout": 0.0,  # no dropout at inference

    # Simulation
    "physics_hz": 240,
    "ctrl_hz": 48,
    "ctrl_every": 5,       # apply GNN every N ctrl steps (~10Hz)
    "max_steps": 2000,
    "comm_radius": 10.0,
    "arena_size": 10.0,
    "ceiling": 10.0,
    "warmup_steps": 120,   # let drones stabilize before GNN control
    "gain": 50.0,          # amplify predicted deltas

    # Blended control (GNN direction + goal attraction)
    "pred_gain": 0.45,     # weight for GNN prediction
    "goal_gain": 0.35,     # weight for direct goal error
    "ramp_steps": 200,     # linearly ramp gains over this many steps
    "max_step_xy": 0.6,    # max XY step per inference (meters)
    "max_step_z": 0.4,     # max Z step per inference (meters)
    "max_step_yaw": 0.25,  # max yaw step per inference (radians)

    # Domain randomization
    "min_drones": 3,
    "max_drones": 5,
    "min_spawn_dist": 1.5,
    "max_obstacles": 0,    # set to 0 for initial testing, increase later
    "obs_radius_range": (0.3, 2.0),

    # LiDAR
    "num_rays": 16,
    "max_range": 5.0,

    # Metrics
    "success_radius": 0.2,
    "collision_radius": 0.35,
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers: Spawning & Randomization
# ═══════════════════════════════════════════════════════════════════════════

def random_positions(n, xy_range, z_range, min_dist, existing=None, max_retries=500):
    """Rejection-sampled random 3D positions with minimum separation."""
    positions = list(existing) if existing is not None else []
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
    return np.array(positions[-n:])


def random_obstacles(max_n, arena, drone_pos, r_range, min_dist=2.0):
    """Random cylindrical obstacles avoiding drone spawn positions."""
    n = np.random.randint(0, max_n + 1)
    obstacles = []  # each: [x, y, radius]
    for _ in range(n):
        for _ in range(200):
            x = np.random.uniform(-arena, arena)
            y = np.random.uniform(-arena, arena)
            r = np.random.uniform(r_range[0], r_range[1])
            pos2d = np.array([x, y])
            # Check vs drones
            ok = all(np.linalg.norm(pos2d - d[:2]) > r + min_dist for d in drone_pos)
            # Check vs other obstacles
            ok = ok and all(np.linalg.norm(pos2d - np.array(o[:2])) > r + o[2] + 0.5 for o in obstacles)
            if ok:
                obstacles.append([x, y, r])
                break
    return np.array(obstacles).reshape(-1, 3) if obstacles else np.zeros((0, 3))


def spawn_pybullet_obstacles(obstacles, client_id):
    """Render obstacles as red cylinders in PyBullet."""
    ids = []
    for obs in obstacles:
        x, y, r = obs
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=10.0,
                                  rgbaColor=[0.8, 0.2, 0.1, 0.6], physicsClientId=client_id)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=10.0,
                                     physicsClientId=client_id)
        body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                 baseCollisionShapeIndex=col,
                                 basePosition=[x, y, 5.0], physicsClientId=client_id)
        ids.append(body)
    return ids


def spawn_target_spheres(targets, client_id):
    """Render target positions as blue spheres."""
    ids = []
    for t in targets:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15,
                                  rgbaColor=[0.1, 0.3, 1.0, 0.8], physicsClientId=client_id)
        body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                 basePosition=[t[0], t[1], t[2]], physicsClientId=client_id)
        ids.append(body)
    return ids


# ═══════════════════════════════════════════════════════════════════════════
# Helpers: Feature Building (matches data_collection_notebook.ipynb EXACTLY)
# ═══════════════════════════════════════════════════════════════════════════

def compute_lidar(pos, yaw, obstacles, num_rays=16, max_range=5.0):
    """Pseudo-LiDAR raycasting — identical to data collection."""
    rays = np.full(num_rays, max_range)
    if len(obstacles) == 0:
        return rays
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
    Build a single 32-dim feature vector for one drone.
    Layout: [local_lin_vel(3), local_ang_vel(3), lidar(16),
             local_pos_error(3), yaw_error(1), floor(1), ceil(1),
             formation_zeros(4)]
    """
    # ── Read state (PyFlyt QuadX state layout) ──
    state = drone.state
    global_pos = np.array(state[3], copy=True)
    global_euler = np.array(state[1], copy=True)
    local_lin_vel = np.array(state[2], copy=True)
    local_ang_vel = np.array(state[0], copy=True)

    # Rotation matrix for this drone
    rot_quat = p.getQuaternionFromEuler(global_euler)
    rot_matrix = np.array(p.getMatrixFromQuaternion(rot_quat)).reshape(3, 3)

    # ── LiDAR ──
    lidar = compute_lidar(global_pos, global_euler[2], obstacles,
                          cfg["num_rays"], cfg["max_range"])

    # ── Target error in local body frame ──
    target_pos = np.array(target[:3])
    target_yaw = target[3]
    global_pos_err = target_pos - global_pos
    local_pos_err = rot_matrix.T @ global_pos_err
    yaw_err = (target_yaw - global_euler[2] + np.pi) % (2 * np.pi) - np.pi

    # ── Boundary awareness ──
    dist_floor = global_pos[2]
    dist_ceil = cfg["ceiling"] - global_pos[2]

    # ── Assemble 32-dim frame (formation = zeros, will be dropped by engineer_x) ──
    frame = np.concatenate([
        local_lin_vel, local_ang_vel, lidar,
        local_pos_err, [yaw_err],
        [dist_floor, dist_ceil],
        [0, 0, 0, 0],  # formation placeholder
    ])

    # Also return global state for edge building
    global_lin_vel = rot_matrix @ local_lin_vel
    return frame, global_pos, global_euler, global_lin_vel


def build_graph(drones, targets, obstacles, prev_frames, cfg, device):
    """
    Build a full PyG Data object for one timestep.
    Returns the graph + current frames for history stacking.
    """
    N = len(drones)
    curr_frames = []
    positions = []
    eulers = []
    lin_vels = []

    for i, drone in enumerate(drones):
        frame, pos, euler, vel = build_single_frame(drone, targets[i], obstacles, cfg)
        curr_frames.append(frame)
        positions.append(pos)
        eulers.append(euler)
        lin_vels.append(vel)

    curr_frames = np.array(curr_frames)  # (N, 32)
    positions = np.array(positions)
    eulers = np.array(eulers)
    lin_vels = np.array(lin_vels)

    # ── Frame stacking: [t0, t-1] = 64 dims ──
    if prev_frames is None:
        prev_frames = np.zeros_like(curr_frames)
    x_raw = np.concatenate([curr_frames, prev_frames], axis=1)  # (N, 64)

    # ── Build edges (radius graph) ──
    edge_index = []
    edge_attr = []
    for i in range(N):
        rot_q = p.getQuaternionFromEuler(eulers[i])
        R = np.array(p.getMatrixFromQuaternion(rot_q)).reshape(3, 3)
        for j in range(N):
            if i == j:
                continue
            rel_pos = positions[j] - positions[i]
            dist = np.linalg.norm(rel_pos)
            if dist <= cfg["comm_radius"]:
                rel_pos_local = R.T @ rel_pos
                rel_vel_local = R.T @ (lin_vels[j] - lin_vels[i])
                edge_index.append([i, j])
                edge_attr.append(np.concatenate([rel_pos_local, [dist], rel_vel_local]))

    # ── To tensors ──
    x_t = torch.tensor(x_raw, dtype=torch.float32)
    if edge_index:
        ei_t = torch.tensor(edge_index, dtype=torch.long).T
        ea_t = torch.tensor(np.array(edge_attr), dtype=torch.float32)
    else:
        ei_t = torch.zeros((2, 0), dtype=torch.long)
        ea_t = torch.zeros((0, cfg["edge_dim"]), dtype=torch.float32)

    graph = Data(x=x_t, edge_index=ei_t, edge_attr=ea_t).to(device)
    return graph, curr_frames, positions, eulers


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate Transform: Local Prediction → Global Setpoint
# ═══════════════════════════════════════════════════════════════════════════

def pred_to_global_setpoints(pred_scaled, positions, eulers, targets, step, cfg):
    """
    Blended control: GNN direction + direct goal attraction.

    pred_scaled: (N, 4) numpy — [dx, dy, dz, dyaw] in physical local frame
    targets:     (N, 4) numpy — [X, Y, Z, Yaw] final targets
    Returns:     (N, 4) numpy — [X, Y, Yaw, Z] for PyFlyt mode=7
    """
    pred_gain = cfg.get("pred_gain", 0.45)
    goal_gain = cfg.get("goal_gain", 0.35)
    ramp_steps = cfg.get("ramp_steps", 200)
    max_step_xy = cfg.get("max_step_xy", 0.6)
    max_step_z = cfg.get("max_step_z", 0.4)
    max_step_yaw = cfg.get("max_step_yaw", 0.25)

    # Ramp gains up over first N steps to avoid initial instability
    ramp = min(1.0, step / ramp_steps)
    pg = pred_gain * ramp
    gg = goal_gain * ramp

    setpoints = np.zeros((len(positions), 4))
    for i in range(len(positions)):
        dx, dy, dz, dyaw = pred_scaled[i]
        yaw = eulers[i][2]

        # Rotate local GNN delta to global frame
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        gnn_dx = cos_y * dx - sin_y * dy
        gnn_dy = sin_y * dx + cos_y * dy

        # Direct goal error (global frame)
        goal_err = targets[i, :3] - positions[i]
        goal_yaw_err = ((targets[i, 3] - yaw + np.pi) % (2 * np.pi)) - np.pi

        # Blend: GNN direction + goal attraction
        blended_x = gnn_dx * pg + goal_err[0] * gg
        blended_y = gnn_dy * pg + goal_err[1] * gg
        blended_z = dz * pg + goal_err[2] * gg
        blended_yaw = dyaw * pg + goal_yaw_err * gg

        # Clamp step size for stability
        xy = np.array([blended_x, blended_y])
        xy_norm = np.linalg.norm(xy)
        if xy_norm > max_step_xy:
            xy = xy / xy_norm * max_step_xy
        blended_z = np.clip(blended_z, -max_step_z, max_step_z)
        blended_yaw = np.clip(blended_yaw, -max_step_yaw, max_step_yaw)

        # PyFlyt mode=7: [X, Y, Yaw, Z]
        setpoints[i, 0] = positions[i][0] + xy[0]
        setpoints[i, 1] = positions[i][1] + xy[1]
        setpoints[i, 2] = yaw + blended_yaw
        setpoints[i, 3] = max(0.5, positions[i][2] + blended_z)
    return setpoints


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(pos_history, vel_history, targets, obstacles, cfg, dt):
    """Compute success rate, collision rate, and trajectory jerk."""
    N = len(targets)
    final_pos = pos_history[-1]  # (N, 3)

    # ── Success rate ──
    dists = np.linalg.norm(final_pos - targets[:, :3], axis=1)
    success = (dists < cfg["success_radius"]).sum()

    # ── Collision detection (cumulative) ──
    drone_collisions = 0
    obs_collisions = 0
    for step_pos in pos_history:
        for i in range(N):
            # Drone-drone
            for j in range(i + 1, N):
                if np.linalg.norm(step_pos[i] - step_pos[j]) < cfg["collision_radius"]:
                    drone_collisions += 1
            # Drone-obstacle
            for obs in obstacles:
                d2d = np.linalg.norm(step_pos[i, :2] - obs[:2])
                if d2d < obs[2] + cfg["collision_radius"]:
                    obs_collisions += 1

    # ── Jerk (rate of change of acceleration) ──
    vel_arr = np.array(vel_history)  # (T, N, 3)
    if len(vel_arr) >= 3:
        acc = np.diff(vel_arr, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt
        avg_jerk = np.mean(np.linalg.norm(jerk, axis=2))
    else:
        avg_jerk = 0.0

    return {
        "success": f"{success}/{N} ({100 * success / N:.0f}%)",
        "final_dists_m": [f"{d:.3f}" for d in dists],
        "drone_collisions": drone_collisions,
        "obstacle_collisions": obs_collisions,
        "avg_jerk_m_s3": f"{avg_jerk:.3f}",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Inference Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(model, normalizer, cfg, device):
    """Run a single randomized inference episode."""

    # ── 1. Domain randomization ──
    num_drones = np.random.randint(cfg["min_drones"], cfg["max_drones"] + 1)
    spawn_pos = random_positions(num_drones, cfg["arena_size"] * 0.5,
                                 z_range=(1.0, 3.0), min_dist=cfg["min_spawn_dist"])
    spawn_orn = np.zeros((num_drones, 3))
    spawn_orn[:, 2] = np.random.uniform(-np.pi, np.pi, num_drones)

    targets = np.zeros((num_drones, 4))
    target_pos = random_positions(num_drones, cfg["arena_size"] * 0.8,
                                  z_range=(1.0, 9.0), min_dist=1.0)
    targets[:, :3] = target_pos
    targets[:, 3] = np.random.uniform(-np.pi, np.pi, num_drones)

    obstacles = random_obstacles(cfg["max_obstacles"], cfg["arena_size"] * 0.8,
                                 spawn_pos, cfg["obs_radius_range"])

    print(f"\n{'='*60}")
    print(f"Episode: {num_drones} drones, {len(obstacles)} obstacles")
    print(f"{'='*60}")

    # ── 2. Initialize PyFlyt ──
    env = Aviary(
        start_pos=spawn_pos,
        start_orn=spawn_orn,
        drone_type="quadx",
        render=True,
    )
    env.set_mode(7)

    # Visualize targets & obstacles
    client = env._client
    spawn_target_spheres(targets, client)
    obs_ids = spawn_pybullet_obstacles(obstacles, client)
    if obs_ids:
        env.register_all_new_bodies()

    # ── 3. Warm-up: hover at spawn to stabilize ──
    for _ in range(cfg["warmup_steps"]):
        for i in range(num_drones):
            # PyFlyt mode=7: [X, Y, Yaw, Z]
            env.set_setpoint(i, np.array([spawn_pos[i][0], spawn_pos[i][1],
                                          spawn_orn[i][2], spawn_pos[i][2]]))
        env.step()

    # ── 4. Inference loop ──
    prev_frames = None
    pos_history = []
    vel_history = []
    ctrl_dt = 1.0 / cfg["ctrl_hz"]
    gnn_interval = cfg["ctrl_every"]

    model.eval()
    current_setpoints = np.zeros((num_drones, 4))
    for i in range(num_drones):
        # [X, Y, Yaw, Z]
        current_setpoints[i] = [spawn_pos[i][0], spawn_pos[i][1],
                                spawn_orn[i][2], spawn_pos[i][2]]

    for step in range(cfg["max_steps"]):
        # Apply GNN every gnn_interval steps (≈10Hz)
        if step % gnn_interval == 0:
            drones = [env.drones[i] for i in range(num_drones)]
            graph, curr_frames, positions, eulers = build_graph(
                drones, targets, obstacles, prev_frames, cfg, device
            )

            # Feature engineering + normalization (same as training)
            graph.x = (engineer_x(graph.x) - normalizer.x_mean) / normalizer.x_std
            graph.edge_attr = ((graph.edge_attr - normalizer.e_mean) / normalizer.e_std
                               if graph.edge_attr.numel() > 0 else graph.edge_attr)

            # Forward pass
            with torch.no_grad():
                pred_norm = model(graph.x, graph.edge_index, graph.edge_attr)

            # Un-scale predictions to physical units, apply gain
            pred_phys = (pred_norm * normalizer.y_scale).cpu().numpy()
            pred_phys *= cfg["gain"]

            # Convert local displacements → global setpoints (blended with goal)
            current_setpoints = pred_to_global_setpoints(
                pred_phys, positions, eulers, targets, step, cfg
            )
            prev_frames = curr_frames

            # Record metrics
            pos_history.append(positions.copy())
            vels = np.array([env.drones[i].state[2] for i in range(num_drones)])
            vel_history.append(vels)

        # Send setpoints to PyFlyt
        for i in range(num_drones):
            env.set_setpoint(i, current_setpoints[i])
        env.step()

        # Print progress + diagnostics
        if step % 200 == 0:
            dists = [np.linalg.norm(np.array(env.drones[i].state[3]) - targets[i, :3])
                     for i in range(num_drones)]
            print(f"  step {step:4d} | avg dist: {np.mean(dists):.3f} m | alt[0]: {env.drones[0].state[3][2]:.2f} m")
            if step == 0:
                print(f"    y_scale: {normalizer.y_scale.cpu().numpy()}")
                print(f"    pred_norm[0]: {pred_norm[0].cpu().numpy()}")
                print(f"    pred_phys[0] (with gain): {pred_phys[0]}")
                print(f"    setpoint[0]: {current_setpoints[0]}")
                print(f"    drone0 pos: {positions[0]}")
                print(f"    target[0]: {targets[0]}")

    # ── 5. Metrics ──
    dt = ctrl_dt * gnn_interval
    metrics = compute_metrics(pos_history, vel_history, targets, obstacles, cfg, dt)
    print(f"\n--- Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    env.disconnect()
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ckpt = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    parser = argparse.ArgumentParser(description="GNN Drone Swarm Inference")
    parser.add_argument("--model", default=os.path.join(ckpt, "best_gatv2.pth"),
                        help="Path to best_gatv2.pth")
    parser.add_argument("--stats", default=os.path.join(ckpt, "normalization_stats.pt"),
                        help="Path to normalization_stats.pt")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    cfg = DEFAULT_CONFIG
    model = SetpointGATv2(
        in_ch=cfg["in_channels"], hid_ch=cfg["hidden_channels"],
        out_ch=cfg["out_channels"], edge_dim=cfg["edge_dim"],
        heads=cfg["heads"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model: {args.model}")

    # Load normalizer
    normalizer = DatasetNormalizer.load(args.stats, device=device)
    print(f"Loaded normalizer: {args.stats}")

    # Run episodes
    all_metrics = []
    for ep in range(args.episodes):
        metrics = run_episode(model, normalizer, cfg, device)
        all_metrics.append(metrics)

    print(f"\n{'='*60}")
    print(f"Completed {args.episodes} episodes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
