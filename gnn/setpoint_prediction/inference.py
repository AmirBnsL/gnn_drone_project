import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p
import torch
import torch.nn as nn
from PyFlyt.core.aviary import Aviary

# ==============================================================================
# Model Definition
# ==============================================================================
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

class SetpointGATv2(nn.Module):
    def __init__(
        self,
        in_channels:  int = 10,
        edge_dim:     int = 4,
        hidden_dim:   int = 64,
        out_channels: int = 4,
        heads:        int = 4,
        num_layers:   int = 3,
        dropout:      float = 0.05,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout    = dropout
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_dim
            out_per_head = hidden_dim // heads
            self.convs.append(GATv2Conv(
                in_channels=c_in, out_channels=out_per_head, heads=heads,
                edge_dim=edge_dim, concat=True, dropout=dropout, add_self_loops=True
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))
            if c_in != hidden_dim:
                self.skips.append(nn.Linear(c_in, hidden_dim, bias=False))
            else:
                self.skips.append(nn.Identity())

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        h = x
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index, edge_attr)
            h_skip = self.skips[i](h)
            h_new = h_new + h_skip
            h_new = self.norms[i](h_new)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new
        return self.mlp(h)


# ==============================================================================
# Pipeline & Utilities
# ==============================================================================
def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def formation_a_offsets(num_drones: int, spacing: float = 2.0) -> np.ndarray:
    offsets = np.zeros((num_drones, 3), dtype=np.float32)
    if num_drones <= 1:
        return offsets

    num_crossbar = (num_drones // 5) if num_drones > 5 else 0
    num_v_legs = num_drones - num_crossbar

    if num_v_legs % 2 == 0:
        num_crossbar += 1
        num_v_legs -= 1

    offsets[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
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


def formation_rectangle_offsets(num_drones: int, spacing: float = 2.0) -> np.ndarray:
    offsets = np.zeros((num_drones, 3), dtype=np.float32)
    if num_drones == 0:
        return offsets

    perimeter = num_drones * spacing
    side = perimeter / 4.0

    corners = [
        np.array([-side / 2.0, -side / 2.0], dtype=np.float32),
        np.array([ side / 2.0, -side / 2.0], dtype=np.float32),
        np.array([ side / 2.0,  side / 2.0], dtype=np.float32),
        np.array([-side / 2.0,  side / 2.0], dtype=np.float32),
    ]

    for i in range(min(num_drones, 4)):
        offsets[i, :2] = corners[i]

    if num_drones > 4:
        remaining = num_drones - 4
        drones_per_edge = [remaining // 4 + (1 if e < remaining % 4 else 0) for e in range(4)]

        current_idx = 4
        for e in range(4):
            start_c = corners[e]
            end_c = corners[(e + 1) % 4]
            num_edge = drones_per_edge[e]

            for step in range(1, num_edge + 1):
                fraction = step / (num_edge + 1)
                offsets[current_idx, :2] = start_c + fraction * (end_c - start_c)
                current_idx += 1

    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets


def formation_triangle_offsets(num_drones: int, spacing: float = 2.0) -> np.ndarray:
    offsets = np.zeros((num_drones, 3), dtype=np.float32)
    if num_drones == 0:
        return offsets

    perimeter = num_drones * spacing
    side = perimeter / 3.0
    height = side * np.sqrt(3) / 2.0

    corners = [
        np.array([-side / 2.0, -height / 3.0], dtype=np.float32),
        np.array([ side / 2.0, -height / 3.0], dtype=np.float32),
        np.array([ 0.0,         2.0 * height / 3.0], dtype=np.float32),
    ]

    for i in range(min(num_drones, 3)):
        offsets[i, :2] = corners[i]

    if num_drones > 3:
        remaining = num_drones - 3
        drones_per_edge = [remaining // 3 + (1 if e < remaining % 3 else 0) for e in range(3)]

        current_idx = 3
        for e in range(3):
            start_c = corners[e]
            end_c = corners[(e + 1) % 3]
            num_edge = drones_per_edge[e]

            for step in range(1, num_edge + 1):
                fraction = step / (num_edge + 1)
                offsets[current_idx, :2] = start_c + fraction * (end_c - start_c)
                current_idx += 1

    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets


def formation_w_offsets(num_drones: int, spacing: float = 2.0) -> np.ndarray:
    offsets = np.zeros((num_drones, 3), dtype=np.float32)
    if num_drones == 0:
        return offsets

    points = np.array(
        [
            [-2.0,  2.0],
            [-1.0,  0.0],
            [ 0.0,  2.0],
            [ 1.0,  0.0],
            [ 2.0,  2.0],
        ],
        dtype=np.float32,
    ) * spacing

    segs = points[1:] - points[:-1]
    seg_len = np.linalg.norm(segs, axis=1)
    total = float(np.sum(seg_len))
    if total <= 1e-6:
        return offsets

    distances = np.linspace(0.0, total, num_drones, dtype=np.float32)
    acc = 0.0
    seg_idx = 0

    for i, d in enumerate(distances):
        while seg_idx < len(seg_len) - 1 and d > acc + seg_len[seg_idx]:
            acc += seg_len[seg_idx]
            seg_idx += 1
        t = 0.0 if seg_len[seg_idx] < 1e-6 else (d - acc) / seg_len[seg_idx]
        offsets[i, :2] = points[seg_idx] + t * segs[seg_idx]

    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets


def build_formation_positions(
    formation_name: str,
    num_drones: int,
    center_xy: np.ndarray,
    altitude: float,
    spacing: float = 2.0,
) -> np.ndarray:
    if formation_name == "a":
        offsets = formation_a_offsets(num_drones, spacing)
    elif formation_name == "rectangle":
        offsets = formation_rectangle_offsets(num_drones, spacing)
    elif formation_name == "triangle":
        offsets = formation_triangle_offsets(num_drones, spacing)
    elif formation_name == "w":
        offsets = formation_w_offsets(num_drones, spacing)
    else:
        raise ValueError(f"Unknown formation: {formation_name}")

    positions = np.zeros((num_drones, 3), dtype=np.float32)
    positions[:, :2] = center_xy + offsets[:, :2]
    positions[:, 2] = altitude
    return positions


def assign_slots(drones_xy: np.ndarray, slots_xy: np.ndarray) -> np.ndarray:
    num_drones = drones_xy.shape[0]

    try:
        from scipy.optimize import linear_sum_assignment

        dist = np.linalg.norm(drones_xy[:, None, :] - slots_xy[None, :, :], axis=2)
        _, col_ind = linear_sum_assignment(dist)
        return col_ind
    except Exception:
        pass

    if num_drones <= 8:
        best_cost = float("inf")
        best_perm = None
        for perm in itertools.permutations(range(num_drones)):
            perm = np.array(perm, dtype=np.int64)
            cost = np.linalg.norm(drones_xy - slots_xy[perm], axis=1).sum()
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        return best_perm

    remaining = list(range(num_drones))
    assignments = []
    for i in range(num_drones):
        dist = np.linalg.norm(slots_xy[remaining] - drones_xy[i], axis=1)
        pick = int(np.argmin(dist))
        assignments.append(remaining[pick])
        remaining.pop(pick)
    return np.array(assignments, dtype=np.int64)


def spawn_target_markers(
    env,
    positions: np.ndarray,
    radius: float = 0.08,
    color: tuple = (0.2, 0.4, 1.0, 0.9),
):
    client = getattr(env, "_client", None)
    if client is None:
        return []

    vis_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
        physicsClientId=client,
    )

    marker_ids = []
    for pos in positions:
        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis_id,
            basePosition=pos.tolist(),
            physicsClientId=client,
        )
        marker_ids.append(body_id)

    if hasattr(env, "register_all_new_bodies"):
        env.register_all_new_bodies()

    return marker_ids


def sample_random_start_positions(
    num_drones: int,
    rng: np.random.Generator,
    xy_limit: float = 5.0,
    altitude_range: tuple = (0.5, 1.5),
    min_separation: float = 1.5,
    max_attempts: int = 2000,
) -> np.ndarray:
    positions = []
    attempts = 0

    while len(positions) < num_drones and attempts < max_attempts:
        attempts += 1
        candidate = rng.uniform(-xy_limit, xy_limit, size=3).astype(np.float32)
        candidate[2] = rng.uniform(altitude_range[0], altitude_range[1])

        if not positions:
            positions.append(candidate)
            continue

        existing_xy = np.array(positions, dtype=np.float32)[:, :2]
        if np.min(np.linalg.norm(existing_xy - candidate[:2], axis=1)) < min_separation:
            continue

        positions.append(candidate)

    if len(positions) < num_drones:
        remaining = num_drones - len(positions)
        extra = rng.uniform(-xy_limit, xy_limit, size=(remaining, 3)).astype(np.float32)
        extra[:, 2] = rng.uniform(altitude_range[0], altitude_range[1], size=remaining)
        positions = list(positions) + list(extra)

    return np.asarray(positions, dtype=np.float32)

class InferencePipeline:
    def __init__(self, model_path: str, normalizer_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[*] Initializing pipeline on {self.device}")
        
        # 1. Load Normalizer
        norm_path = Path(normalizer_path)
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalizer file not found: {norm_path}")
        with open(norm_path, 'r') as f:
            self.norm_data = json.load(f)
            
        self.x_mean = torch.tensor(self.norm_data["x_mean"], device=self.device)
        self.x_std = torch.tensor(self.norm_data["x_std"], device=self.device)
        self.target_mean = torch.tensor(self.norm_data["target_mean"], device=self.device)
        self.target_std = torch.tensor(self.norm_data["target_std"], device=self.device)
        self.edge_mean = torch.tensor(self.norm_data["edge_mean"], device=self.device)
        self.edge_std = torch.tensor(self.norm_data["edge_std"], device=self.device)
        
        # 2. Load Model
        ckpt_path = Path(model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
            
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # Instantiate model (handling hyperparameters if present)
        hparams = ckpt.get("model_hparams", {"in_channels": 10, "edge_dim": 4})
        self.model = SetpointGATv2(**hparams).to(self.device)
        
        # Handle state dict key
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("[+] Model and scalers loaded successfully.")

    def preprocess(self, states, formations):
        """
        states: numpy array [N, 9]
            Layout: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, ang_x, ang_y, ang_z]
            Matches data collection: local linear velocity + local angular velocity.
        formations: numpy array [N, 3] one-hot encoding for the formation.
        
        Returns: graph data (x, edge_index, edge_attr, yaw) ready for GNN.
        """
        N = states.shape[0]
        
        # We use velocity and angular rates as in the training dataset.
        vel = states[:, 3:6]
        ang = states[:, 6:9]
        pos = states[:, 0:3]
        yaw = ang[:, 2] # ang_z (matches training normalizer)
        
        vel_t = torch.tensor(vel, dtype=torch.float32, device=self.device)
        ang_x_y = torch.tensor(ang[:, 0:2], dtype=torch.float32, device=self.device)
        yaw_t = torch.tensor(yaw, dtype=torch.float32, device=self.device).unsqueeze(1)
        yaw_t = wrap_angle(yaw_t)
        
        form_t = torch.tensor(formations, dtype=torch.float32, device=self.device)
        
        # Construct raw x [N, 9] (before normalizer's sin/cos transformation)
        # Features: vel_x, vel_y, vel_z, ang_x, ang_y, yaw, form_A, form_Rect, form_Tri
        raw_x = torch.cat([vel_t, ang_x_y, yaw_t, form_t], dim=1)
        
        # The model expects 10 features: Convert yaw to sin/cos
        sin_yaw = torch.sin(yaw_t)
        cos_yaw = torch.cos(yaw_t)
        
        # Transformed X: [vel (3), ang_x_y (2), sin_yaw (1), cos_yaw (1), form (3)] = 10
        x_transformed = torch.cat([vel_t, ang_x_y, sin_yaw, cos_yaw, form_t], dim=1)
        
        # Normalize X
        # Note: Depending on training, x_mean might be length 9 (pre-sin/cos) or 10.
        # Handling the user's notes: "The model expects 10 node features", so if normalizer has 10 elements, normalizer maps to x_transformed.
        # If normalizer has 9 elements, it scales the 9 features FIRST, then we do sin/cos. 
        # But based on src/dataset.py logic, it replaces yaw with sin/cos FIRST, the x_mean is 10 long.
        # We will assume x_mean is length 10. If not, pad it.
        if len(self.x_mean) == 9:
            # Fallback if x_mean was generated before sin/cos split:
            x_norm = (raw_x - self.x_mean) / self.x_std
            # then do sin/cos
            sin_yaw = torch.sin(x_norm[:, 5].unsqueeze(1))
            cos_yaw = torch.cos(x_norm[:, 5].unsqueeze(1))
            x_model_input = torch.cat([x_norm[:, :5], sin_yaw, cos_yaw, x_norm[:, 6:]], dim=1)
        else:
            x_model_input = (x_transformed - self.x_mean) / self.x_std
        
        # Prepare Edges (Fully connected graph excluding self)
        adj = torch.ones((N, N), device=self.device) - torch.eye(N, device=self.device)
        edge_index = adj.nonzero().t() # [2, E]
        
        src, dst = edge_index[0], edge_index[1]
        
        # Global position difference
        pos_t = torch.tensor(pos, dtype=torch.float32, device=self.device)
        delta_pos_global = pos_t[dst] - pos_t[src]
        
        delta_x_global = delta_pos_global[:, 0]
        delta_y_global = delta_pos_global[:, 1]
        delta_z = delta_pos_global[:, 2]
        
        distance = torch.norm(delta_pos_global, dim=1)
        
        # Edge Attributes: [rel_x, rel_y, rel_z, distance] in global frame
        raw_e = torch.stack([delta_x_global, delta_y_global, delta_z, distance], dim=1)
        
        # Normalize Edge Attributes
        e_norm = (raw_e - self.edge_mean) / self.edge_std
        
        return x_model_input, edge_index, e_norm, yaw_t

    def postprocess(self, y_norm, yaw_t):
        """
        Denormalizes model output and transforms setpoints back to Global frame (if required by PyFlyt)
        or keeps them in Local frame depending on the flight controller.
        Assuming PyFlyt API expects setpoints in the Global frame (or you pass local to an internal controller).
        By default we will output Global frame delta setpoints.
        """
        y_physical_local = y_norm * self.target_std + self.target_mean
        
        x_local = y_physical_local[:, 0]
        y_local = y_physical_local[:, 1]
        delta_z = y_physical_local[:, 2]
        delta_yaw = y_physical_local[:, 3]
        
        yaw = yaw_t.squeeze(1)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        
        # Inverse transform from Local Body Frame to Global Frame
        delta_x_global = x_local * cos_y - y_local * sin_y
        delta_y_global = x_local * sin_y + y_local * cos_y
        
        setpoints_global = torch.stack([delta_x_global, delta_y_global, delta_z, delta_yaw], dim=1)
        return setpoints_global.detach().cpu().numpy()

    @torch.no_grad()
    def predict(self, states, formations):
        x_model_input, edge_index, e_norm, yaw_t = self.preprocess(states, formations)
        y_norm = self.model(x_model_input, edge_index, e_norm)
        setpoints_global = self.postprocess(y_norm, yaw_t)
        return setpoints_global


# ==============================================================================
# Main Simulation Loop
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="PyFlyt Aviary Inference Script with GATv2")
    parser.add_argument("--model", type=str, default="./checkpoints/best_setpoint_gatv2.pt", help="Path to model checkpoint")
    parser.add_argument("--norm", type=str, default="./results/normalizer.json", help="Path to normalizer JSON")
    parser.add_argument("--num_drones", type=int, default=5, help="Number of drones in swarm")
    parser.add_argument(
        "formation",
        nargs="?",
        default="a",
        choices=["a", "rectangle", "triangle", "w"],
        help="Formation type: a, rectangle, triangle, or w",
    )
    args = parser.parse_args()

    pipeline = InferencePipeline(args.model, args.norm)

    num_drones = args.num_drones
    rng = np.random.default_rng()
    start_pos = sample_random_start_positions(num_drones, rng)

    formation_name = args.formation
    formation_center = np.mean(start_pos[:, :2], axis=0).astype(np.float32)
    formation_altitude = 1.5
    formation_spacing = 2.0

    slot_positions = build_formation_positions(
        formation_name,
        num_drones,
        formation_center,
        formation_altitude,
        spacing=formation_spacing,
    )
    slot_assignments = assign_slots(start_pos[:, :2], slot_positions[:, :2])
    final_positions = slot_positions[slot_assignments]
    
    env = Aviary(
        drone_type="quadx",
        start_pos=start_pos,
        start_orn=np.zeros((num_drones, 3)),
        drone_options={"control_hz": 120},
        render=True,
        physics_hz=240,
    )

    env.reset()
    
    # IMPORTANT: Mode 7 needs a strong initial signal to spin up
    for i in range(num_drones):
        env.drones[i].set_mode(7) 

    spawn_target_markers(env, final_positions)

    print("[*] Forcing Takeoff Phase...")
    # Force drones to 1.5 meters to break ground contact
    for _ in range(50):
        for i in range(num_drones):
            env.set_setpoint(i, np.array([start_pos[i, 0], start_pos[i, 1], 0.0, 1.5], dtype=np.float64))
        env.step()

    formations = np.zeros((num_drones, 3))
    if formation_name == "a":
        formations[:, 0] = 1.0
    elif formation_name == "rectangle":
        formations[:, 1] = 1.0
    elif formation_name == "triangle":
        formations[:, 2] = 1.0
    elif formation_name == "w":
        print("[!] 'w' formation is custom and not in the training one-hot set.")
    else:
        raise ValueError(f"Unknown formation: {formation_name}")

    print("[*] Starting Model-Controlled Flight...")
    base_pred_gain = 0.45
    base_goal_gain = 0.35
    min_goal_gain = 0.12
    ramp_steps = 200
    max_step_xy = 0.6
    max_step_z = 0.4
    max_step_yaw = 0.25
    for step in range(1200):
        states_list = []
        yaw_list = []
        for i in range(num_drones):
            s = env.state(i)
            # PyFlyt order used in data collection: ang_vel, euler, lin_vel, pos
            global_pos = np.asarray(s[3], dtype=np.float32)
            local_lin_vel = np.asarray(s[2], dtype=np.float32)
            local_ang_vel = np.asarray(s[0], dtype=np.float32)
            global_euler = np.asarray(s[1], dtype=np.float32)

            flat_state = np.concatenate((global_pos, local_lin_vel, local_ang_vel))
            states_list.append(flat_state)
            yaw_list.append(global_euler[2])
        states = np.array(states_list)

        # Get GNN predictions
        setpoints_global = pipeline.predict(states, formations)
        
        # Gain ramp to avoid large early excursions
        ramp = min(1.0, step / ramp_steps)
        pred_gain = base_pred_gain * ramp
        goal_gain = min_goal_gain + (base_goal_gain - min_goal_gain) * ramp
        
        for i in range(num_drones):
            # Calculate next waypoint (blend model delta with final target)
            goal_error = final_positions[i] - states[i, 0:3]
            blended = (setpoints_global[i, 0:3] * pred_gain) + (goal_error * goal_gain)

            # Clamp per-step movement to avoid flying away then returning
            xy = blended[0:2]
            xy_norm = np.linalg.norm(xy)
            if xy_norm > max_step_xy:
                xy = (xy / (xy_norm + 1e-6)) * max_step_xy
            dz = float(np.clip(blended[2], -max_step_z, max_step_z))
            dyaw = float(np.clip(setpoints_global[i, 3] * pred_gain, -max_step_yaw, max_step_yaw))

            target_x = states[i, 0] + xy[0]
            target_y = states[i, 1] + xy[1]
            # Ensure Z is always high enough to stay airborne
            target_z = max(1.2, states[i, 2] + dz)
            target_yaw = yaw_list[i] + dyaw

            action = np.array([target_x, target_y, target_yaw, target_z], dtype=np.float64)
            env.set_setpoint(i, action)

        # Step the physics twice to allow for mechanical lag
        env.step()
        env.step()
        
        if step % 100 == 0:
            print(f"Step {step} | Drone 0 Alt: {states[0, 2]:.2f}m | Target Z: {target_z:.2f}m")

    print("[+] Mission Complete. Press Ctrl+C to close.")
    try:
        while True:
            env.step() # Keep running physics so they maintain hover
            time.sleep(0.01) # تقليل استهلاك المعالج
    except KeyboardInterrupt:
        print("[*] Closing simulator...")
        env.close()
if __name__ == "__main__":
    main()
