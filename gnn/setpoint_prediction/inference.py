import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
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
        states: numpy array [N, num_state_features]
            Assuming PyFlyt raw states include [vel_x, vel_y, vel_z, ang_x, ang_y, ang_z, pos_x, pos_y, pos_z]
            We extract the relevant features based on the normalizer input dimensions.
        formations: numpy array [N, 3] one-hot encoding for the formation.
        
        Returns: graph data (x, edge_index, edge_attr, yaw) ready for GNN.
        """
        N = states.shape[0]
        
        # Assume standard extraction (adjust indices based on your specific PyFlyt observation wrapper)
        # Typically in PyFlyt QuadX: pos(0:3), vel(3:6), euler(6:9), body_rates(9:12)
        # We need: vel_x, vel_y, vel_z (3:6 for local/global?), ang_x, ang_y, ang_z (6:9 for euler angles)
        
        vel = states[:, 3:6]
        euler = states[:, 6:9]
        pos = states[:, 0:3]
        yaw = euler[:, 2] # ang_z
        
        vel_t = torch.tensor(vel, dtype=torch.float32, device=self.device)
        ang_x_y = torch.tensor(euler[:, 0:2], dtype=torch.float32, device=self.device)
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
        
        # Coordinate Transformation: Global to Local Body Frame of the SOURCE drone
        src_yaw = yaw_t[src].squeeze(1)
        cos_y = torch.cos(src_yaw)
        sin_y = torch.sin(src_yaw)
        
        delta_x_global = delta_pos_global[:, 0]
        delta_y_global = delta_pos_global[:, 1]
        delta_z = delta_pos_global[:, 2]
        
        # Rotation matrix logic:
        # local_x = global_x * cos(yaw) + global_y * sin(yaw)
        # local_y = -global_x * sin(yaw) + global_y * cos(yaw)
        rel_x_local = delta_x_global * cos_y + delta_y_global * sin_y
        rel_y_local = -delta_x_global * sin_y + delta_y_global * cos_y
        
        distance = torch.norm(delta_pos_global, dim=1)
        
        # Edge Attributes: [rel_x, rel_y, rel_z, distance]
        raw_e = torch.stack([rel_x_local, rel_y_local, delta_z, distance], dim=1)
        
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
    args = parser.parse_args()

    pipeline = InferencePipeline(args.model, args.norm)

    num_drones = args.num_drones
    # Spread drones out so they don't collide on takeoff
    start_pos = np.zeros((num_drones, 3))
    start_pos[:, 0] = np.linspace(-3, 3, num_drones)
    start_pos[:, 2] = 0.1 # Start slightly above the ground plane
    
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

    print("[*] Forcing Takeoff Phase...")
    # Force drones to 1.5 meters to break ground contact
    for _ in range(50):
        for i in range(num_drones):
            env.set_setpoint(i, np.array([start_pos[i, 0], 0.0, 1.5, 0.0], dtype=np.float64))
        env.step()

    formations = np.zeros((num_drones, 3))
    formations[:, 0] = 1.0 

    print("[*] Starting Model-Controlled Flight...")
    for step in range(1200):
        states_list = []
        for i in range(num_drones):
            s = env.state(i)
            # PyFlyt order: pos(0:3), vel(3:6), euler(6:9), ang_vel(9:12)
            flat_state = np.concatenate((s[0], s[2], s[1], s[3]))
            states_list.append(flat_state)
        states = np.array(states_list)

        # Get GNN predictions
        setpoints_global = pipeline.predict(states, formations)
        
        # INCREASED GAIN: Need enough force to move against air resistance
        gain = 0.8 
        
        for i in range(num_drones):
            # Calculate next waypoint
            target_x = states[i, 0] + (setpoints_global[i, 0] * gain)
            target_y = states[i, 1] + (setpoints_global[i, 1] * gain)
            # Ensure Z is always high enough to stay airborne
            target_z = max(1.2, states[i, 2] + (setpoints_global[i, 2] * gain))
            target_yaw = states[i, 8] + (setpoints_global[i, 3] * gain)

            action = np.array([target_x, target_y, target_z, target_yaw], dtype=np.float64)
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
