import json
import os
import time
import shutil
from typing import Literal
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import pybullet as p
import torch
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data, HeteroData, InMemoryDataset

from PyFlyt.core import Aviary

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
FORMATION_NAMES = ("a", "rectangle", "triangle")
FORMATION_TO_ID = {name: idx for idx, name in enumerate(FORMATION_NAMES)}
SPLIT_NAMES = ("train", "val", "test")
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}
TASK_TYPES = (
    "setpoint_prediction",
    "residual_correction",
    "formation_assignment_homo",
    "formation_assignment_hetero",
)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def wind_generator(time: float, position: np.ndarray):
    wind = np.zeros_like(position)
    wind[:, 2] = np.sin(time) * 0.5 + np.random.normal(0, 0.2, size=(len(position),))
    wind[:, 0] = np.cos(time / 2.0) * 0.3
    wind[:, 1] = np.sin(time / 2.0) * 0.3
    return wind

def sample_episode_initial_conditions(
    num_drones: int,
    rng: np.random.Generator,
    xy_limit: float = 10.0,
    altitude_range: tuple[float, float] = (0.5, 5.0),
):
    start_pos = rng.uniform(-xy_limit, xy_limit, size=(num_drones, 3))
    start_pos[:, 2] = rng.uniform(
        altitude_range[0], altitude_range[1], size=(num_drones,)
    )
    start_orn = np.zeros((num_drones, 3))
    start_orn[:, 2] = rng.uniform(-np.pi, np.pi, size=(num_drones,))
    return start_pos, start_orn

def sample_obstacles(rng: np.random.Generator, num_obstacles: int, xy_limit: float):
    if num_obstacles == 0:
        return np.zeros((0, 2))
    return rng.uniform(-xy_limit, xy_limit, size=(num_obstacles, 2))

def resolve_formation_name(dataset_type: str):
    if dataset_type in {"a", "formation_a"}: return "a"
    if dataset_type in {"rectangle", "rectangular", "formation_rectangle"}: return "rectangle"
    if dataset_type in {"triangle", "formation_triangle"}: return "triangle"
    return None

def _formation_a_offsets(num_drones: int, spacing: float = 2.0):
    offsets = np.zeros((num_drones, 3))
    if num_drones == 1: return offsets
    num_crossbar = (num_drones // 5) if num_drones > 5 else 0
    num_v_legs = num_drones - num_crossbar
    if num_v_legs % 2 == 0:
        num_crossbar += 1
        num_v_legs -= 1
    offsets[0] = np.array([0.0, 0.0, 0.0])
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

def _formation_rectangle_offsets(num_drones: int, spacing: float = 2.0):
    offsets = np.zeros((num_drones, 3))
    if num_drones == 0: return offsets
    perimeter = num_drones * spacing
    side = perimeter / 4.0
    corners = [
        np.array([-side / 2, -side / 2]),
        np.array([side / 2, -side / 2]),
        np.array([side / 2, side / 2]),
        np.array([-side / 2, side / 2]),
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

def _formation_triangle_offsets(num_drones: int, spacing: float = 2.0):
    offsets = np.zeros((num_drones, 3))
    if num_drones == 0: return offsets
    perimeter = num_drones * spacing
    side = perimeter / 3.0
    h = side * np.sqrt(3) / 2.0
    corners = [
        np.array([-side / 2, -h / 3]),
        np.array([side / 2, -h / 3]),
        np.array([0, 2 * h / 3]),
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

def apply_obstacle_avoidance(slots, obstacles, obstacle_radius, padding=1.0):
    if len(obstacles) == 0: return slots
    safe_slots = np.copy(slots)
    min_dist = obstacle_radius + padding
    for i in range(len(safe_slots)):
        for obs in obstacles:
            diff = safe_slots[i, :2] - obs
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                direction = diff / (dist + 1e-6)
                safe_slots[i, :2] = obs + direction * min_dist
    return safe_slots

def _build_formation_setpoints(formation_name: str, start_pos: np.ndarray, obstacles=np.empty((0, 2)), obstacle_radius=0.0):
    num_drones = len(start_pos)
    formation_center = np.mean(start_pos[:, :2], axis=0)
    target_altitude = np.mean(start_pos[:, 2])

    if formation_name == "a": offsets = _formation_a_offsets(num_drones)
    elif formation_name == "rectangle": offsets = _formation_rectangle_offsets(num_drones)
    elif formation_name == "triangle": offsets = _formation_triangle_offsets(num_drones)
    else: return None, None, None

    global_slots = np.zeros((num_drones, 3))
    global_slots[:, :2] = formation_center + offsets[:, :2]
    safe_global_slots = apply_obstacle_avoidance(global_slots, obstacles, obstacle_radius, padding=1.0)
    dist_matrix = np.linalg.norm(start_pos[:, None, :2] - safe_global_slots[None, :, :2], axis=2)
    _, assigned_indices = linear_sum_assignment(dist_matrix)

    setpoints = np.zeros((num_drones, 4))
    for i in range(num_drones):
        slot_idx = assigned_indices[i]
        setpoints[i, :2] = safe_global_slots[slot_idx, :2]
        setpoints[i, 2] = 0.0
        setpoints[i, 3] = target_altitude

    return setpoints, assigned_indices, offsets

def create_aviary(start_pos: np.ndarray, start_orn: np.ndarray, environmental_wind: bool, obstacles: np.ndarray = np.empty((0, 2)), obstacle_radius: float = 1.0, graphical: bool = False):
    env = Aviary(start_pos=start_pos, start_orn=start_orn, drone_type="quadx", render=graphical)
    if environmental_wind: env.register_wind_field_function(wind_generator)
    env.set_mode(7)
    if len(obstacles) > 0:
        physics_client = env._client
        for obs in obstacles:
            col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=obstacle_radius, height=10.0, physicsClientId=physics_client)
            vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=obstacle_radius, length=10.0, rgbaColor=[1, 0, 0, 0.5], physicsClientId=physics_client)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[obs[0], obs[1], 5.0], physicsClientId=physics_client)
        env.register_all_new_bodies()
    return env

def build_setpoints(dataset_type: str, start_pos: np.ndarray, start_orn: np.ndarray, rng: np.random.Generator, obstacles: np.ndarray = np.empty((0, 2)), obstacle_radius: float = 1.0):
    num_drones = len(start_pos)
    formation_name = resolve_formation_name(dataset_type)
    if formation_name is not None:
        setpoints, col_ind, offsets = _build_formation_setpoints(formation_name, start_pos, obstacles, obstacle_radius)
        if setpoints is not None: return setpoints, col_ind, offsets

    setpoints = np.zeros((num_drones, 4))
    if dataset_type == "hovering":
        setpoints[:, :2] = start_pos[:, :2]
        setpoints[:, 2] = start_orn[:, 2]
        setpoints[:, 3] = start_pos[:, 2]
    else:
        radius = 10.0 if dataset_type == "aggressive" else 5.0
        setpoints[:, :2] = start_pos[:, :2] + rng.uniform(-radius, radius, size=(num_drones, 2))
        setpoints[:, 2] = rng.uniform(-np.pi, np.pi, size=(num_drones,))
        setpoints[:, 3] = rng.uniform(1.0, radius, size=(num_drones,))
    col_ind = np.arange(num_drones)
    return setpoints, col_ind, np.zeros((num_drones, 3))

def maybe_add_sensor_noise(global_pos, global_euler, local_lin_vel, local_ang_vel, noisy_sensors, noise_variance):
    if not noisy_sensors: return global_pos, global_euler, local_lin_vel, local_ang_vel
    global_pos = global_pos + np.random.normal(0, noise_variance, size=3)
    global_euler = global_euler + np.random.normal(0, noise_variance, size=3)
    local_lin_vel = local_lin_vel + np.random.normal(0, noise_variance, size=3)
    local_ang_vel = local_ang_vel + np.random.normal(0, noise_variance, size=3)
    return global_pos, global_euler, local_lin_vel, local_ang_vel

def compute_lidar_features(global_pos, global_euler, obstacles, obstacle_radius, num_rays=16, max_range=5.0):
    """Simulates 2D LiDAR raycasting to detect obstacles and return distance readings."""
    obs_features = np.full(num_rays, max_range)
    if len(obstacles) == 0:
        return obs_features
        
    drone_pos = global_pos[:2]
    drone_yaw = global_euler[2]
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False) + drone_yaw
    rays = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    W = obstacles - drone_pos # (O, 2)
    t = W @ rays.T # (O, 16)
    hit_mask = t > 0
    
    W_sq = np.sum(W**2, axis=1)[:, None]
    d_sq = W_sq - t**2
    valid_hit = hit_mask & (d_sq <= obstacle_radius**2)
    
    if np.any(valid_hit):
        safe_d_sq = np.clip(d_sq, 0, obstacle_radius**2)
        hit_dists = t - np.sqrt(obstacle_radius**2 - safe_d_sq)
        hit_dists[~valid_hit] = max_range
        hit_dists[hit_dists <= 0] = max_range
        min_dists = np.min(hit_dists, axis=0) # (16,)
        obs_features = np.minimum(obs_features, min_dists)
        
    return obs_features

def build_drone_features(drone, drone_idx, setpoint, assigned_slot_idx, naive_offset, task_type, noisy_sensors, noise_variance, formation_one_hot, obstacles, obstacle_radius, include_formation_in_state, start_pos_center):
    state = drone.state
    global_pos = np.array(state[3], copy=True)
    global_euler = np.array(state[1], copy=True)
    local_lin_vel = np.array(state[2], copy=True)
    local_ang_vel = np.array(state[0], copy=True)

    global_pos, global_euler, local_lin_vel, local_ang_vel = maybe_add_sensor_noise(
        global_pos, global_euler, local_lin_vel, local_ang_vel, noisy_sensors, noise_variance
    )

    obs_features = compute_lidar_features(global_pos, global_euler, obstacles, obstacle_radius)

    gnn_input_state = np.concatenate([local_lin_vel, local_ang_vel, obs_features])
    if include_formation_in_state and formation_one_hot is not None:
        gnn_input_state = np.concatenate([gnn_input_state, formation_one_hot])

    target_global_pos = np.array([setpoint[0], setpoint[1], setpoint[3]])
    target_global_yaw = setpoint[2]

    if task_type == "setpoint_prediction":
        global_pos_error = target_global_pos - global_pos
        yaw_error = target_global_yaw - global_euler[2]
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        rotation_quaternion = p.getQuaternionFromEuler(global_euler)
        rot_matrix = np.array(p.getMatrixFromQuaternion(rotation_quaternion)).reshape(3, 3)
        local_pos_error = rot_matrix.T @ global_pos_error
        y_label = np.concatenate([local_pos_error, [yaw_error]])

    elif task_type == "residual_correction":
        naive_global_pos = np.array([
            start_pos_center[0] + naive_offset[0],
            start_pos_center[1] + naive_offset[1],
            target_global_pos[2],
        ])
        y_label = target_global_pos - naive_global_pos

    elif task_type in ("formation_assignment_homo", "formation_assignment_hetero"):
        y_label = np.array([assigned_slot_idx], dtype=np.float32)

    return gnn_input_state, y_label, drone.pwm, global_pos

def build_edges(global_positions: np.ndarray, communication_radius: float):
    edges = []
    edge_attrs = []
    num_drones = len(global_positions)
    for i in range(num_drones):
        for j in range(num_drones):
            if i == j: continue
            rel_pos = global_positions[j] - global_positions[i]
            dist = np.linalg.norm(rel_pos)
            if dist <= communication_radius:
                edges.append([i, j])
                edge_attrs.append(rel_pos)
    return edges, edge_attrs

def collect_step_data(env, active_drones, setpoints, slot_assignments, naive_offsets, task_type, noisy_sensors, noise_variance, communication_radius, formation_one_hot, obstacles, obstacle_radius, include_formation_in_state, start_pos_center):
    episode_states, episode_targets, episode_labels, global_positions = [], [], [], []
    for i, drone_idx in enumerate(active_drones):
        drone = env.drones[drone_idx]
        slot_idx = slot_assignments[i]
        gnn_input_state, gnn_input_target, motor_pwm_labels, global_pos = build_drone_features(
            drone, drone_idx, setpoints[i], slot_idx, naive_offsets[slot_idx], task_type, noisy_sensors, noise_variance, formation_one_hot, obstacles, obstacle_radius, include_formation_in_state, start_pos_center
        )
        episode_states.append(gnn_input_state)
        episode_targets.append(gnn_input_target)
        episode_labels.append(motor_pwm_labels)
        global_positions.append(global_pos)
    edges, edge_attrs = build_edges(np.array(global_positions), communication_radius)
    return episode_states, episode_targets, episode_labels, edges, edge_attrs, global_positions

def compute_split_episode_counts(num_episodes, split_ratios):
    counts = [int(num_episodes * ratio) for ratio in split_ratios]
    remainder = num_episodes - sum(counts)
    for idx in range(remainder): counts[idx] += 1
    return {split_name: count for split_name, count in zip(SPLIT_NAMES, counts)}

def resolve_split_spread_scale(split_name, validation_spread_scale, test_spread_scale):
    if split_name == "val": return validation_spread_scale
    if split_name == "test": return test_spread_scale
    return 1.0

def should_sample_step(step_idx, max_steps, tapered_sampling, dense_sampling_steps, mid_sampling_steps, mid_step_stride, late_step_stride):
    if not tapered_sampling or step_idx == max_steps - 1: return True
    mid_sampling_steps = max(mid_sampling_steps, dense_sampling_steps)
    if step_idx < dense_sampling_steps: return True
    if step_idx < mid_sampling_steps: return (step_idx - dense_sampling_steps) % mid_step_stride == 0
    return (step_idx - mid_sampling_steps) % late_step_stride == 0

def build_episode_seed(seed, split_name, split_episode_idx):
    if seed is None: return None
    return int(seed + SPLIT_SEED_OFFSETS[split_name] + split_episode_idx)

def write_dataset_metadata(metadata_path, metadata):
    with open(metadata_path, "w", encoding="ascii") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")

def save_dataset(dataset_path, all_graphs, formation_names, split_name):
    data, slices = InMemoryDataset.collate(all_graphs)
    torch.save(
        {"data": data, "slices": slices, "formation_names": formation_names, "split_name": split_name},
        dataset_path,
    )


def convert_history_to_graphs(raw_history, task_type, active_drones, naive_offsets, formation_id, global_episode_id, obstacles):
    """Converts raw numerical numpy histories into PyTorch Geometric graph data structures."""
    graphs = []
    num_active = len(active_drones)
    
    for step_data in raw_history:
        x = torch.as_tensor(np.asarray(step_data["ep_states"]), dtype=torch.float32)
        target = torch.as_tensor(np.asarray(step_data["ep_targets"]), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(step_data["ep_labels"]), dtype=torch.float32)

        if step_data["edges"]:
            edge_index = torch.tensor(step_data["edges"], dtype=torch.long).t().contiguous()
            edge_attr = torch.as_tensor(np.asarray(step_data["edge_attrs"]), dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float32)

        if task_type == "formation_assignment_hetero":
            graph = HeteroData()
            graph["drone"].x = x
            graph["drone"].y = y
            graph["drone"].pos = torch.as_tensor(np.asarray(step_data["glob_pos"]), dtype=torch.float32)
            graph["slot"].x = torch.as_tensor(naive_offsets, dtype=torch.float32)
            graph["drone", "communicates", "drone"].edge_index = edge_index
            graph["drone", "communicates", "drone"].edge_attr = edge_attr
            drones = torch.arange(num_active, dtype=torch.long)
            slots = target.view(-1).long()
            graph["drone", "assigned_to", "slot"].edge_label_index = torch.stack([drones, slots], dim=0)
            graph.formation_id = torch.tensor([formation_id], dtype=torch.long)
            graph.episode_id = torch.tensor([global_episode_id], dtype=torch.long)
            graph.step_idx = torch.tensor([step_data["step_idx"]], dtype=torch.long)
            graph.num_drones = torch.tensor([num_active], dtype=torch.long)
            graph.obstacles = torch.as_tensor(obstacles, dtype=torch.float32)
        else:
            graph = Data(
                x=x, target=target, y=y, edge_index=edge_index, edge_attr=edge_attr,
                pos=torch.as_tensor(np.asarray(step_data["glob_pos"]), dtype=torch.float32),
                obstacles=torch.as_tensor(obstacles, dtype=torch.float32),
                formation_id=torch.tensor([formation_id], dtype=torch.long),
                episode_id=torch.tensor([global_episode_id], dtype=torch.long),
                step_idx=torch.tensor([step_data["step_idx"]], dtype=torch.long),
                num_drones=torch.tensor([num_active], dtype=torch.long),
            )
        
        graphs.append(graph)
    return graphs


# ---------------------------------------------------------
# MULTIPROCESSING WORKER
# ---------------------------------------------------------
def simulate_episode(config: dict):
    """Worker function to simulate a complete episode logic and return captured graphs."""
    
    rng = np.random.default_rng(config["episode_seed"])
    num_drones = int(rng.integers(10, 21))
    
    episode_dataset_type = config["dataset_type"]
    if episode_dataset_type in {"mixed_formations", "mixed", "formations"}:
        episode_dataset_type = str(rng.choice(config["mixed_formation_types"]))

    xy_limit = config["base_xy_limit"] * config["split_spread_scale"]
    start_pos, start_orn = sample_episode_initial_conditions(
        num_drones, rng, xy_limit=xy_limit, altitude_range=config["altitude_range"]
    )

    num_obs_conf = config["num_obstacles"]
    if isinstance(num_obs_conf, tuple) or isinstance(num_obs_conf, list):
        current_num_obstacles = int(rng.integers(num_obs_conf[0], num_obs_conf[1] + 1))
    else:
        current_num_obstacles = num_obs_conf

    obstacles = sample_obstacles(rng, current_num_obstacles, xy_limit=5.0)

    # Graphical MUST be false for parallel processes
    env = create_aviary(start_pos, start_orn, config["environmental_wind"], obstacles, config["obstacle_radius"], False)
    setpoints, col_ind, naive_offsets = build_setpoints(episode_dataset_type, start_pos, start_orn, rng, obstacles, config["obstacle_radius"])
    
    active_drones = list(range(num_drones))
    formation_name = resolve_formation_name(episode_dataset_type)
    formation_id = FORMATION_TO_ID[formation_name] if formation_name else -1

    formation_one_hot = None
    if formation_id >= 0:
        formation_one_hot = np.zeros(len(FORMATION_NAMES), dtype=np.float32)
        formation_one_hot[formation_id] = 1.0

    for i, drone_idx in enumerate(active_drones):
        env.set_setpoint(drone_idx, setpoints[i])

    saved_steps = 0
    steps_since_last_event = 0
    failure_injected_this_episode = False
    already_converged_for_segment = False
    raw_history = []
    count_near_zero = 0
    count_significant = 0

    max_steps = config["max_steps"]
    task_type = config["task_type"]

    for step_idx in range(max_steps):
        steps_since_last_event += 1

        should_inject_failure = (config["inject_failures"] and not failure_injected_this_episode and step_idx >= max_steps // 2)
        if should_inject_failure and len(active_drones) > 2:
            failed_drone = active_drones.pop()
            env.drones[failed_drone].set_mode(0)
            active_start_pos = np.array([env.drones[idx].state[3] for idx in active_drones])
            setpoints, col_ind, naive_offsets = build_setpoints(episode_dataset_type, active_start_pos, start_orn[: len(active_drones)], rng, obstacles, config["obstacle_radius"])
            for i, drone_idx in enumerate(active_drones):
                env.drones[drone_idx].set_mode(7)
                env.set_setpoint(drone_idx, setpoints[i])
            failure_injected_this_episode = True
            steps_since_last_event = 0
            already_converged_for_segment = False

        if config["dynamic_formation"] and step_idx > 0 and step_idx % 100 == 0:
            next_shape = str(rng.choice(tuple(set(FORMATION_NAMES) - {formation_name})))
            formation_name = next_shape
            formation_id = FORMATION_TO_ID[formation_name]
            formation_one_hot = np.zeros(len(FORMATION_NAMES), dtype=np.float32)
            formation_one_hot[formation_id] = 1.0
            active_start_pos = np.array([env.drones[idx].state[3] for idx in active_drones])
            setpoints, col_ind, naive_offsets = build_setpoints(formation_name, active_start_pos, start_orn[: len(active_drones)], rng, obstacles, config["obstacle_radius"])
            for i, drone_idx in enumerate(active_drones): env.set_setpoint(drone_idx, setpoints[i])
            steps_since_last_event = 0
            already_converged_for_segment = False

        is_converged = False
        if config["conv_stopping"] and steps_since_last_event > 50:
            max_pos_error = 0.0
            for i, drone_idx in enumerate(active_drones):
                drone_pos = env.drones[drone_idx].state[3]
                target_pos = np.array([setpoints[i][0], setpoints[i][1], setpoints[i][3]])
                error = np.linalg.norm(drone_pos - target_pos)
                if error > max_pos_error: max_pos_error = error
            is_converged = max_pos_error < config["conv_threshold"]

        if is_converged and not already_converged_for_segment:
            already_converged_for_segment = True

        can_stop_episode = is_converged
        if can_stop_episode:
            if config["inject_failures"] and not failure_injected_this_episode: can_stop_episode = False
            if config["dynamic_formation"]: can_stop_episode = False

        if is_converged or should_sample_step(step_idx, max_steps, config["tapered_sampling"], config["dense_sampling_steps"], config["mid_sampling_steps"], config["mid_step_stride"], config["late_step_stride"]):
            if can_stop_episode or should_sample_step(step_idx, max_steps, config["tapered_sampling"], config["dense_sampling_steps"], config["mid_sampling_steps"], config["mid_step_stride"], config["late_step_stride"]):
                
                ep_states, ep_targets, ep_labels, edges, edge_attrs, glob_pos = collect_step_data(
                    env, active_drones, setpoints, col_ind, naive_offsets, task_type, config["noisy_sensors"], config["noise_variance"], config["communication_radius"], formation_one_hot, obstacles, config["obstacle_radius"], config["include_formation_in_state"], np.mean(start_pos[:, :2], axis=0)
                )

                should_skip_step = False
                is_near_zero = False

                if task_type == "residual_correction":
                    max_residual_norm = np.max(np.linalg.norm(ep_targets, axis=1))
                    is_near_zero = (max_residual_norm < config["residual_dropout_threshold"])
                    if is_near_zero:
                        # Since we simulate globally inside config loop, we track running total locally per episode and extrapolate roughly
                        total_potential = count_near_zero + count_significant + 1
                        if total_potential > 10 and (count_near_zero + 1) / total_potential > config["residual_balance_ratio"]:
                            should_skip_step = True

                if not should_skip_step:
                    raw_history.append({
                        "ep_states": ep_states,
                        "ep_targets": ep_targets,
                        "ep_labels": ep_labels,
                        "edges": edges,
                        "edge_attrs": edge_attrs,
                        "glob_pos": glob_pos,
                        "step_idx": step_idx,
                    })
                    saved_steps += 1
                    
                    if task_type == "residual_correction":
                        if is_near_zero: count_near_zero += 1
                        else: count_significant += 1

        env.step()
        if can_stop_episode: break

    env.disconnect()
    
    # Process collected history into PyTorch graphs at the end to minimize overhead in the simulation loop
    graphs = convert_history_to_graphs(
        raw_history=raw_history,
        task_type=task_type,
        active_drones=active_drones,
        naive_offsets=naive_offsets,
        formation_id=formation_id,
        global_episode_id=config["global_episode_id"],
        obstacles=obstacles,
    )
    
    # Save the generated graphs for this episode to a temporary file on disk (bypassing IPC serialization bottlenecks)
    temp_file = os.path.join(config["temp_dir"], f"ep_{config['global_episode_id']}.pt")
    torch.save(graphs, temp_file)

    episode_center = np.mean(start_pos[:, :2], axis=0)
    initial_xy_radius = float(np.max(np.linalg.norm(start_pos[:, :2] - episode_center, axis=1)))

    episode_record = {
        "episode_id": config["global_episode_id"],
        "split": config["split_name"],
        "split_episode_idx": config["split_episode_idx"],
        "episode_seed": config["episode_seed"],
        "num_drones": num_drones,
        "episode_dataset_type": episode_dataset_type,
        "formation_name": formation_name,
        "initial_xy_limit": xy_limit,
        "initial_xy_radius": initial_xy_radius,
        "total_steps": step_idx + 1,
        "saved_steps": saved_steps,
        "converged": bool(is_converged),
    }

    return config["split_name"], temp_file, count_near_zero, count_significant, episode_record

# ---------------------------------------------------------
# MAIN COORDINATOR
# ---------------------------------------------------------
def generate_dataset_parallel(
    num_workers: int = 4,
    num_episodes: int = 50,
    max_steps: int = 500,
    dataset_name: str = "formation_dataset",
    dataset_type: str = "mixed_formations",
    task_type: Literal[
        "setpoint_prediction",
        "residual_correction",
        "formation_assignment_homo",
        "formation_assignment_hetero",
    ] = "setpoint_prediction",
    num_obstacles: int | tuple[int, int] = 0,
    obstacle_radius: float = 1.0,
    residual_balance_ratio: float = 0.5,
    residual_dropout_threshold: float = 0.1,
    inject_failures: bool = False,
    dynamic_formation: bool = False,
    noisy_sensors: bool = False,
    noise_variance: float = 0.01,
    environmental_wind: bool = False,
    communication_radius: float = np.inf,
    include_formation_in_state: bool = True,
    mixed_formation_types: tuple = FORMATION_NAMES,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 12345,
    base_xy_limit: float = 10.0,
    altitude_range: tuple[float, float] = (0.5, 5.0),
    validation_spread_scale: float = 1.25,
    test_spread_scale: float = 1.5,
    tapered_sampling: bool = True,
    dense_sampling_steps: int = 120,
    mid_sampling_steps: int = 240,
    mid_step_stride: int = 2,
    late_step_stride: int = 5,
    conv_stopping: bool = True,
    conv_threshold: float = 0.2,
) -> tuple[dict[str, str], str]:
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    datasets_dir = os.path.join(repo_root, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {TASK_TYPES}")

    dataset_prefix = os.path.join(datasets_dir, f"{dataset_name}_{dataset_type}")
    split_episode_counts = compute_split_episode_counts(num_episodes, split_ratios)
    
    temp_dir = os.path.join(datasets_dir, f"temp_{dataset_name}_{dataset_type}_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Collect all tasks to compute
    tasks = []
    global_episode_id = 0
    for split_name in SPLIT_NAMES:
        count = split_episode_counts[split_name]
        spread_scale = resolve_split_spread_scale(split_name, validation_spread_scale, test_spread_scale)
        for idx in range(count):
            ep_seed = build_episode_seed(seed, split_name, idx)
            tasks.append({
                "split_name": split_name,
                "split_episode_idx": idx,
                "split_spread_scale": spread_scale,
                "episode_seed": ep_seed,
                "global_episode_id": global_episode_id,
                
                # Include standard kwargs mapping for the worker dict extraction
                "dataset_type": dataset_type,
                "dataset_name": dataset_name,
                "task_type": task_type,
                "num_obstacles": num_obstacles,
                "obstacle_radius": obstacle_radius,
                "residual_balance_ratio": residual_balance_ratio,
                "residual_dropout_threshold": residual_dropout_threshold,
                "inject_failures": inject_failures,
                "dynamic_formation": dynamic_formation,
                "noisy_sensors": noisy_sensors,
                "noise_variance": noise_variance,
                "environmental_wind": environmental_wind,
                "communication_radius": communication_radius,
                "include_formation_in_state": include_formation_in_state,
                "mixed_formation_types": mixed_formation_types,
                "base_xy_limit": base_xy_limit,
                "altitude_range": altitude_range,
                "max_steps": max_steps,
                "tapered_sampling": tapered_sampling,
                "dense_sampling_steps": dense_sampling_steps,
                "mid_sampling_steps": mid_sampling_steps,
                "mid_step_stride": mid_step_stride,
                "late_step_stride": late_step_stride,
                "conv_stopping": conv_stopping,
                "conv_threshold": conv_threshold,
                "temp_dir": temp_dir,
            })
            global_episode_id += 1

    split_temp_files = {s: [] for s in SPLIT_NAMES}
    split_summaries = {
        s: {"num_episodes": split_episode_counts[s], "num_graphs": 0, "spread_scale": resolve_split_spread_scale(s, validation_spread_scale, test_spread_scale), "count_near_zero": 0, "count_significant": 0} 
        for s in SPLIT_NAMES
    }
    episode_records = []

    print(f"Launching {len(tasks)} physical episodes across {num_workers} parallel workers...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(simulate_episode, task): task for task in tasks}
        completed = 0
        with tqdm(total=len(tasks), desc="Episodes", unit="ep", dynamic_ncols=True) as progress:
            for future in as_completed(futures):
                split_name, temp_file_path, c_zero, c_sig, record = future.result()
                
                split_temp_files[split_name].append(temp_file_path)
                split_summaries[split_name]["num_graphs"] += record["saved_steps"]
                split_summaries[split_name]["count_near_zero"] += c_zero
                split_summaries[split_name]["count_significant"] += c_sig
                episode_records.append(record)
                
                completed += 1
                progress.update(1)
                progress.set_postfix({
                    "split": f"{split_name}-{record['split_episode_idx'] + 1}",
                    "conv": record["converged"],
                    "steps": record["total_steps"],
                    "captured": record["saved_steps"],
                })

    print(f"Parallel Simulation finished in {time.time() - start_time:.2f}s. Aggregating datasets...")
    
    generated_files = {}
    for split_name, temp_paths in split_temp_files.items():
        if not temp_paths: continue
        
        all_graphs = []
        for temp_path in temp_paths:
            all_graphs.extend(torch.load(temp_path, weights_only=False))
            os.remove(temp_path)  # Delete temp file as we build the final split
            
        split_dataset_path = f"{dataset_prefix}_{split_name}.pt"
        save_dataset(split_dataset_path, all_graphs, FORMATION_NAMES, split_name)
        generated_files[split_name] = os.path.basename(split_dataset_path)
        print(f"Generated {split_name} dataset -> {split_dataset_path}")
        
    shutil.rmtree(temp_dir, ignore_errors=True)

    metadata_path = f"{dataset_prefix}_metadata.json"
    metadata = {
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "task_type": task_type,
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "num_obstacles": num_obstacles,
            "obstacle_radius": obstacle_radius,
            "inject_failures": inject_failures,
            "dynamic_formation": dynamic_formation,
            "noisy_sensors": noisy_sensors,
            "noise_variance": noise_variance,
            "environmental_wind": environmental_wind,
            "communication_radius": communication_radius,
            "include_formation_in_state": include_formation_in_state,
            "mixed_formation_types": list(mixed_formation_types),
            "split_ratios": list(split_ratios),
            "seed": seed,
            "base_xy_limit": base_xy_limit,
            "altitude_range": list(altitude_range),
            "validation_spread_scale": validation_spread_scale,
            "test_spread_scale": test_spread_scale,
            "tapered_sampling": tapered_sampling,
            "dense_sampling_steps": dense_sampling_steps,
            "mid_sampling_steps": mid_sampling_steps,
            "mid_step_stride": mid_step_stride,
            "late_step_stride": late_step_stride,
            "conv_stopping": conv_stopping,
            "conv_threshold": conv_threshold,
            "residual_balance_ratio": residual_balance_ratio,
            "residual_dropout_threshold": residual_dropout_threshold,
        },
        "split_summary": split_summaries,
        # sort records to keep order consistent
        "episodes": sorted(episode_records, key=lambda x: x["episode_id"]),
    }
    write_dataset_metadata(metadata_path, metadata)
    print(f"Generated dataset metadata -> {metadata_path}")

    return generated_files, metadata_path

if __name__ == "__main__":
    # Test execution
    generated_files, metadata_path = generate_dataset_parallel(
        num_workers=1,
        dataset_name="set_point_prediction_dataset_parallel",
        dataset_type="mixed_formations",
        task_type="setpoint_prediction",
        noisy_sensors=False,
        environmental_wind=False,
        dynamic_formation=False,
        inject_failures=False,
        communication_radius=10.0,
        include_formation_in_state=True,
        max_steps=1200,
        num_episodes=50,
        tapered_sampling=True,
        conv_stopping=True,
        conv_threshold=0.2,
        num_obstacles=(0, 10),  # Random obstacles between 0 and 10
        residual_balance_ratio=0.5,  # Aim for 50/50 split between near-zero and significant residuals
        residual_dropout_threshold=0.1,  # 10cm threshold
        seed=12345,
    )
    print(f"Done. Outputs: {generated_files}")
#python data_collection_parallel.py | grep -v 'argv\\[0\\]='