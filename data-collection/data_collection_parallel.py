import io
import json
import os
import time
import shutil
from typing import Literal
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import numpy as np
import pybullet as p
import torch
from safetensors.torch import load_file, save_file
import gc
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data, HeteroData, InMemoryDataset

from PyFlyt.core import Aviary


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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


def sample_obstacles(
    rng: np.random.Generator,
    num_obstacles: int,
    xy_limit: float,
    z_limit: tuple[float, float],
    obstacle_radius: float = 1.0,
    obstacle_radius_range: tuple[float, float] | None = None,
):
    if num_obstacles == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=np.float32)

    xy = rng.uniform(-xy_limit, xy_limit, size=(num_obstacles, 2))
    z = rng.uniform(z_limit[0], z_limit[1], size=(num_obstacles, 1))
    obstacles = np.hstack((xy, z)).astype(np.float32)

    if obstacle_radius_range is not None:
        low = float(min(obstacle_radius_range))
        high = float(max(obstacle_radius_range))
        obstacle_radii = rng.uniform(low, high, size=(num_obstacles,)).astype(
            np.float32
        )
    else:
        obstacle_radii = np.full(
            (num_obstacles,), float(obstacle_radius), dtype=np.float32
        )

    return obstacles, obstacle_radii


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

def apply_obstacle_avoidance(slots, obstacles, obstacle_radii, padding=1.0):
    if len(obstacles) == 0: return slots
    safe_slots = np.copy(slots)
    obs_xy = np.asarray(obstacles, dtype=np.float32)[:, :2]
    if obstacle_radii is None or len(obstacle_radii) == 0:
        obstacle_radii = np.ones((len(obstacles),), dtype=np.float32)
    for i in range(len(safe_slots)):
        for obs_idx, obs in enumerate(obs_xy):
            min_dist = float(obstacle_radii[obs_idx]) + padding
            diff = safe_slots[i, :2] - obs
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                direction = diff / (dist + 1e-6)
                safe_slots[i, :2] = obs + direction * min_dist
    return safe_slots

def _build_formation_setpoints(
    formation_name: str,
    start_pos: np.ndarray,
    obstacles=np.empty((0, 2)),
    obstacle_radii=np.empty((0,)),
):
    num_drones = len(start_pos)
    formation_center = np.mean(start_pos[:, :2], axis=0)
    target_altitude = np.mean(start_pos[:, 2])

    if formation_name == "a": offsets = _formation_a_offsets(num_drones)
    elif formation_name == "rectangle": offsets = _formation_rectangle_offsets(num_drones)
    elif formation_name == "triangle": offsets = _formation_triangle_offsets(num_drones)
    else: return None, None, None

    global_slots = np.zeros((num_drones, 3))
    global_slots[:, :2] = formation_center + offsets[:, :2]
    safe_global_slots = apply_obstacle_avoidance(global_slots, obstacles, obstacle_radii, padding=1.0)
    dist_matrix = np.linalg.norm(start_pos[:, None, :2] - safe_global_slots[None, :, :2], axis=2)
    _, assigned_indices = linear_sum_assignment(dist_matrix)

    setpoints = np.zeros((num_drones, 4))
    for i in range(num_drones):
        slot_idx = assigned_indices[i]
        setpoints[i, :2] = safe_global_slots[slot_idx, :2]
        setpoints[i, 2] = 0.0
        setpoints[i, 3] = target_altitude

    return setpoints, assigned_indices, offsets


def create_aviary(
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    environmental_wind: bool,
    obstacles: np.ndarray = np.empty((0, 3)), 
    obstacle_radii: np.ndarray | None = None,
    obstacle_radius: float = 1.0,
    graphical: bool = False,
):
    drone_options = dict()
    drone_options["control_hz"] = 60
    env = Aviary(start_pos=start_pos, start_orn=start_orn, drone_type="quadx", render=graphical, physics_hz=240, drone_options=drone_options)
    
    if environmental_wind: 
        env.register_wind_field_function(wind_generator)
        
    env.set_mode(7)
    
    if len(obstacles) > 0:
        physics_client = env._client
        if obstacle_radii is None or len(obstacle_radii) == 0:
            obstacle_radii_arr = np.full((len(obstacles),), obstacle_radius, dtype=np.float32)
        else:
            obstacle_radii_arr = np.asarray(obstacle_radii, dtype=np.float32)
            
        for obs, obs_radius in zip(obstacles, obstacle_radii_arr):
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=float(obs_radius), physicsClientId=physics_client)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=float(obs_radius), rgbaColor=[1, 0, 0, 0.5], physicsClientId=physics_client)
            
            p.createMultiBody(
                baseMass=0, 
                baseCollisionShapeIndex=col_id, 
                baseVisualShapeIndex=vis_id, 
                basePosition=[obs[0], obs[1], obs[2]], 
                physicsClientId=physics_client
            )
            
        env.register_all_new_bodies()
        
    return env

def build_setpoints(
    dataset_type: str,
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    rng: np.random.Generator,
    obstacles: np.ndarray = np.empty((0, 2)),
    obstacle_radii: np.ndarray | None = None,
    obstacle_radius: float = 1.0,
):
    num_drones = len(start_pos)
    if obstacle_radii is None:
        obstacle_radii = np.full((len(obstacles),), obstacle_radius, dtype=np.float32)
    formation_name = resolve_formation_name(dataset_type)
    if formation_name is not None:
        setpoints, col_ind, offsets = _build_formation_setpoints(formation_name, start_pos, obstacles, obstacle_radii)
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


def compute_apf_setpoints(
    current_positions: np.ndarray,
    final_setpoints: np.ndarray,
    obstacles: np.ndarray,
    obstacle_radii: np.ndarray,
    attractive_gain: float,
    repulsive_gain: float,
    repulsion_padding: float,
    max_step_size: float,
    vertical_gain: float,
):
    if len(current_positions) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    final_target_positions = np.column_stack(
        [final_setpoints[:, 0], final_setpoints[:, 1], final_setpoints[:, 3]]
    )

    attractive = final_target_positions - current_positions
    attractive[:, :2] *= attractive_gain
    attractive[:, 2] *= vertical_gain

    repulsive_xyz = np.zeros((len(current_positions), 3), dtype=np.float32)

    if len(obstacles) > 0:
        if obstacle_radii is None or len(obstacle_radii) == 0:
            obstacle_radii = np.ones((len(obstacles),), dtype=np.float32)

        diff = current_positions[:, None, :3] - obstacles[None, :, :3]
        dist = np.linalg.norm(diff, axis=2) + 1e-6

        influence_radius = obstacle_radii[None, :] + repulsion_padding
        inside_influence = dist < influence_radius

        safe_dist = np.maximum(dist, 1e-3)
        repulse_strength = (
            repulsive_gain * (1.0 / safe_dist - 1.0 / influence_radius) / (safe_dist**2)
        )
        repulse_strength = np.where(inside_influence, repulse_strength, 0.0)

        direction = diff / safe_dist[..., None]
        repulsive_xyz = np.sum(repulse_strength[..., None] * direction, axis=1)

    total_delta = np.copy(attractive)
    total_delta[:, :3] += repulsive_xyz

    norm = np.linalg.norm(total_delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_step_size / (norm + 1e-6))
    bounded_delta = total_delta * scale
    intermediate_positions = current_positions + bounded_delta

    intermediate_setpoints = np.zeros_like(final_setpoints)
    intermediate_setpoints[:, 0] = intermediate_positions[:, 0]
    intermediate_setpoints[:, 1] = intermediate_positions[:, 1]
    intermediate_setpoints[:, 2] = final_setpoints[:, 2]
    intermediate_setpoints[:, 3] = intermediate_positions[:, 2]
    return intermediate_setpoints


def compute_lidar_features(
    global_pos: np.ndarray,
    global_euler: np.ndarray,
    obstacles: np.ndarray,
    obstacle_radii: np.ndarray,
    physics_client: int | None = None,
    num_azimuth: int = 16,
    num_elevation: int = 4,
    fov_up_deg: float = 15.0,
    fov_down_deg: float = -15.0,
    max_range: float = 5.0,
):
    num_drones = global_pos.shape[0]
    num_rays = num_azimuth * num_elevation

    if num_drones == 0:
        return np.zeros((0, num_rays), dtype=np.float32)

    azimuths = np.linspace(0.0, 2.0 * np.pi, num_azimuth, endpoint=False, dtype=np.float32)
    if num_elevation > 1:
        elevations = np.linspace(
            np.radians(fov_down_deg),
            np.radians(fov_up_deg),
            num_elevation,
            dtype=np.float32,
        )
    else:
        elevations = np.array([0.0], dtype=np.float32)

    az_grid, el_grid = np.meshgrid(azimuths, elevations, indexing="ij")
    local_rays = np.stack(
        [
            np.cos(el_grid) * np.cos(az_grid),
            np.cos(el_grid) * np.sin(az_grid),
            np.sin(el_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)

    r, pt, y = global_euler[:, 0], global_euler[:, 1], global_euler[:, 2]
    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(pt), np.sin(pt)
    cz, sz = np.cos(y), np.sin(y)

    rot_matrices = np.zeros((num_drones, 3, 3), dtype=np.float32)
    rot_matrices[:, 0, 0] = cy * cz
    rot_matrices[:, 0, 1] = sx * sy * cz - cx * sz
    rot_matrices[:, 0, 2] = cx * sy * cz + sx * sz
    rot_matrices[:, 1, 0] = cy * sz
    rot_matrices[:, 1, 1] = sx * sy * sz + cx * cz
    rot_matrices[:, 1, 2] = cx * sy * sz - sx * cz
    rot_matrices[:, 2, 0] = -sy
    rot_matrices[:, 2, 1] = sx * cy
    rot_matrices[:, 2, 2] = cx * cy

    global_rays = np.einsum("nij,rj->nri", rot_matrices, local_rays, optimize=True)

    if physics_client is not None:
        ray_starts = np.repeat(global_pos[:, None, :], num_rays, axis=1)
        ray_ends = ray_starts + (global_rays * max_range)
        flat_starts = ray_starts.reshape(-1, 3).tolist()
        flat_ends = ray_ends.reshape(-1, 3).tolist()
        results = p.rayTestBatch(
            rayFromPositions=flat_starts,
            rayToPositions=flat_ends,
            physicsClientId=physics_client,
        )
        hit_fractions = np.array([res[2] for res in results], dtype=np.float32)
        return hit_fractions.reshape(num_drones, num_rays) * max_range

    if len(obstacles) == 0:
        return np.full((num_drones, num_rays), max_range, dtype=np.float32)

    obs = np.asarray(obstacles, dtype=np.float32)
    if obstacle_radii is None or len(obstacle_radii) == 0:
        radii = np.ones((len(obs),), dtype=np.float32)
    else:
        radii = np.asarray(obstacle_radii, dtype=np.float32)

    ray_dirs = global_rays[:, None, :, :]  # (N, 1, R, 3)
    oc = global_pos[:, None, None, :] - obs[None, :, None, :]  # (N, O, 1, 3)
    b = 2.0 * np.sum(oc * ray_dirs, axis=-1)  # (N, O, R)
    c = np.sum(oc * oc, axis=-1) - (radii[None, :, None] ** 2)  # (N, O, 1)
    disc = b**2 - 4.0 * c

    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) * 0.5
    t2 = (-b + sqrt_disc) * 0.5
    t_hit = np.where(t1 > 0.0, t1, t2)
    valid_hit = (disc >= 0.0) & (t_hit > 0.0)
    hit_dists = np.where(valid_hit, t_hit, max_range)
    hit_dists = np.where(hit_dists > 0.0, hit_dists, max_range)
    return np.minimum(np.min(hit_dists, axis=1), max_range)


def build_drone_features(drone, drone_idx, setpoint, assigned_slot_idx, naive_offset, task_type, noisy_sensors, noise_variance, formation_one_hot, obstacles, obstacle_radii, include_formation_in_state, start_pos_center, physics_client=None, precomputed_state=None):
    if precomputed_state is None:
        state = drone.state
        global_pos = np.array(state[3], copy=True)
        global_euler = np.array(state[1], copy=True)
        local_lin_vel = np.array(state[2], copy=True)
        local_ang_vel = np.array(state[0], copy=True)

        global_pos, global_euler, local_lin_vel, local_ang_vel = maybe_add_sensor_noise(
            global_pos, global_euler, local_lin_vel, local_ang_vel, noisy_sensors, noise_variance
        )

        obs_features = compute_lidar_features(
            global_pos[None, :],
            global_euler[None, :],
            obstacles,
            obstacle_radii,
            physics_client=physics_client,
        )[0]
    else:
        global_pos = precomputed_state["global_pos"]
        global_euler = precomputed_state["global_euler"]
        local_lin_vel = precomputed_state["local_lin_vel"]
        local_ang_vel = precomputed_state["local_ang_vel"]
        obs_features = precomputed_state["obs_features"]

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
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return gnn_input_state, y_label, drone.pwm, global_pos


def build_edges(
    global_positions: np.ndarray,
    communication_radius: float,
    global_velocities: np.ndarray | None = None,
):
    """Build communication graph edges with rich 3D attributes.
    
    Edge attrs per edge (i->j):
      [0] rel_x
      [1] rel_y
      [2] rel_z
      [3] dist
      [4] rel_vx
      [5] rel_vy
      [6] rel_vz
    """
    pos = np.asarray(global_positions, dtype=np.float32)
    num_drones = pos.shape[0]

    if num_drones < 2:
        return [], []

    # rel[i, j] = pos[j] - pos[i]
    rel = pos[None, :, :] - pos[:, None, :]  # shape: (N, N, 3)

    # Exclude self-edges
    not_self = ~np.eye(num_drones, dtype=bool)

    if np.isinf(communication_radius):
        mask = not_self
    else:
        dist_sq = np.einsum("ijk,ijk->ij", rel, rel)
        radius_sq = float(communication_radius) ** 2
        mask = (dist_sq <= radius_sq) & not_self

    src, dst = np.nonzero(mask)

    edges = np.column_stack((src, dst)).tolist()
    
    # Compute distances
    dists = np.linalg.norm(rel[src, dst], axis=1, keepdims=True)  # (E, 1)
    
    # Compute relative velocities if provided
    if global_velocities is not None and len(global_velocities) == num_drones:
        vel = np.asarray(global_velocities, dtype=np.float32)
        rel_vel = vel[None, :, :] - vel[:, None, :]  # (N, N, 3)
        rel_vel_attrs = rel_vel[src, dst]  # (E, 3)
        edge_attrs = np.hstack([rel[src, dst], dists, rel_vel_attrs]).tolist()
    else:
        # Zero velocities for residual correction (no physics)
        zero_vel = np.zeros((len(src), 3), dtype=np.float32)
        edge_attrs = np.hstack([rel[src, dst], dists, zero_vel]).tolist()
    
    return edges, edge_attrs


def collect_step_data(env, active_drones, setpoints, slot_assignments, naive_offsets, task_type, noisy_sensors, noise_variance, communication_radius, formation_one_hot, obstacles, obstacle_radii, include_formation_in_state, start_pos_center):
    episode_states, episode_targets, episode_labels = [], [], []

    precomputed = []
    for drone_idx in active_drones:
        state = env.drones[drone_idx].state
        global_pos = np.array(state[3], copy=True)
        global_euler = np.array(state[1], copy=True)
        local_lin_vel = np.array(state[2], copy=True)
        local_ang_vel = np.array(state[0], copy=True)
        global_pos, global_euler, local_lin_vel, local_ang_vel = maybe_add_sensor_noise(
            global_pos, global_euler, local_lin_vel, local_ang_vel, noisy_sensors, noise_variance
        )
        precomputed.append({
            "global_pos": global_pos,
            "global_euler": global_euler,
            "local_lin_vel": local_lin_vel,
            "local_ang_vel": local_ang_vel,
        })

    global_positions = np.asarray([entry["global_pos"] for entry in precomputed], dtype=np.float32)
    global_eulers = np.asarray([entry["global_euler"] for entry in precomputed], dtype=np.float32)
    global_velocities = np.asarray([entry["local_lin_vel"] for entry in precomputed], dtype=np.float32)
    obs_features_batch = compute_lidar_features(
        global_positions,
        global_eulers,
        obstacles,
        obstacle_radii,
        physics_client=env._client,
    )

    for i, drone_idx in enumerate(active_drones):
        drone = env.drones[drone_idx]
        slot_idx = slot_assignments[i]
        precomputed[i]["obs_features"] = obs_features_batch[i]

        gnn_input_state, gnn_input_target, motor_pwm_labels, _ = build_drone_features(
            drone,
            drone_idx,
            setpoints[i],
            slot_idx,
            naive_offsets[slot_idx],
            task_type,
            noisy_sensors,
            noise_variance,
            formation_one_hot,
            obstacles,
            obstacle_radii,
            include_formation_in_state,
            start_pos_center,
            physics_client=env._client,
            precomputed_state=precomputed[i],
        )
        episode_states.append(gnn_input_state)
        episode_targets.append(gnn_input_target)
        episode_labels.append(motor_pwm_labels)

    edges, edge_attrs = build_edges(global_positions, communication_radius, global_velocities)
    return episode_states, episode_targets, episode_labels, edges, edge_attrs, global_positions.tolist()

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


def distribute_episodes_to_workers(
    num_workers,
    split_episode_counts,
    seed,
    validation_spread_scale,
    test_spread_scale,
    worker_batch_size: int | None = None,
    **common_kwargs,
):
    """Distribute episodes into tasks.

    - If worker_batch_size is provided (>0), each task gets at most that many episodes.
    - Otherwise, create roughly num_workers tasks per split.
    """
    worker_tasks = []
    worker_id = 0
    next_global_episode_id = 0

    for split_name in SPLIT_NAMES:
        count = split_episode_counts[split_name]
        spread_scale = resolve_split_spread_scale(
            split_name, validation_spread_scale, test_spread_scale
        )

        if worker_batch_size is not None and worker_batch_size > 0:
            chunk_size = worker_batch_size
        else:
            chunk_size = max(1, (count + num_workers - 1) // num_workers)

        start_episode_idx = 0
        while start_episode_idx < count:
            num_episodes_for_worker = min(chunk_size, count - start_episode_idx)

            episode_configs = []
            for local_idx in range(num_episodes_for_worker):
                split_episode_idx = start_episode_idx + local_idx
                ep_seed = build_episode_seed(seed, split_name, split_episode_idx)
                episode_configs.append(
                    {
                        "split_episode_idx": split_episode_idx,
                        "episode_seed": ep_seed,
                        "global_episode_id": next_global_episode_id,
                    }
                )
                next_global_episode_id += 1

            worker_tasks.append(
                {
                    "worker_id": worker_id,
                    "split_name": split_name,
                    "spread_scale": spread_scale,
                    "episode_configs": episode_configs,
                    "num_episodes": num_episodes_for_worker,
                    **common_kwargs,
                }
            )

            worker_id += 1
            start_episode_idx += num_episodes_for_worker

    return worker_tasks


def write_dataset_metadata(metadata_path, metadata):
    with open(metadata_path, "w", encoding="ascii") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")


def save_object_safetensors(file_path: str, payload_obj) -> None:
    """Persist an arbitrary Python object inside a safetensors file."""
    buffer = io.BytesIO()
    torch.save(payload_obj, buffer)
    byte_arr = np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()
    tensor_payload = torch.from_numpy(byte_arr)
    save_file({"payload": tensor_payload}, file_path)



def load_object_safetensors(file_path: str):
    """Memory-safe loader that immediately frees byte references."""
    tensors = load_file(file_path)
    payload = tensors["payload"]
    payload_bytes = payload.cpu().numpy().tobytes()
    del tensors
    del payload
    gc.collect()
    return torch.load(io.BytesIO(payload_bytes), weights_only=False)


    

def save_dataset(dataset_path, all_graphs, formation_names, split_name):
    data, slices = InMemoryDataset.collate(all_graphs)
    save_object_safetensors(
        dataset_path,
        {"data": data, "slices": slices, "formation_names": formation_names, "split_name": split_name},
    )


def convert_history_to_graphs(raw_history, task_type, active_drones, naive_offsets, formation_id, global_episode_id, obstacles, obstacle_radii):
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
            edge_attr = torch.empty((0, 7), dtype=torch.float32)

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
            graph.obstacle_radii = torch.as_tensor(obstacle_radii, dtype=torch.float32)
        else:
            graph = Data(
                x=x, target=target, y=y, edge_index=edge_index, edge_attr=edge_attr,
                pos=torch.as_tensor(np.asarray(step_data["glob_pos"]), dtype=torch.float32),
                obstacles=torch.as_tensor(obstacles, dtype=torch.float32),
                obstacle_radii=torch.as_tensor(obstacle_radii, dtype=torch.float32),
                formation_id=torch.tensor([formation_id], dtype=torch.long),
                episode_id=torch.tensor([global_episode_id], dtype=torch.long),
                step_idx=torch.tensor([step_data["step_idx"]], dtype=torch.long),
                num_drones=torch.tensor([num_active], dtype=torch.long),
            )
        
        graphs.append(graph)
    return graphs


def generate_residual_correction_sample(
    rng: np.random.Generator,
    num_drones: int,
    formation_name: str,
    xy_limit: float = 10.0,
    altitude_range: tuple[float, float] = (0.5, 5.0),
    num_obstacles: int = 0,
    obstacle_radius: float = 1.0,
    obstacle_radius_range: tuple[float, float] | None = None,
    communication_radius: float = 10.0,
    include_formation_in_state: bool = True,
    formation_names: tuple = FORMATION_NAMES,
    noisy_sensors: bool = False,
    noise_variance: float = 0.01,
):
    """
    Generate a single residual correction sample WITHOUT physics simulation.
    Much faster for dataset generation - purely geometric computation.
    """
    # Sample initial positions
    start_pos, start_orn = sample_episode_initial_conditions(
        num_drones, rng, xy_limit=xy_limit, altitude_range=altitude_range
    )
    
    # Sample obstacles
    obstacles, obstacle_radii = sample_obstacles(
        rng, num_obstacles, xy_limit=5.0, z_limit=altitude_range, obstacle_radius=obstacle_radius, obstacle_radius_range=obstacle_radius_range
    )
    
    # Build setpoints with obstacle avoidance
    setpoints, col_ind, naive_offsets = build_setpoints(
        formation_name, start_pos, start_orn, rng, obstacles, obstacle_radii, obstacle_radius
    )
    
    formation_id = FORMATION_TO_ID.get(formation_name, -1)
    start_pos_center = np.mean(start_pos[:, :2], axis=0)
    
    # Formation one-hot
    formation_one_hot = None
    if formation_id >= 0 and include_formation_in_state:
        formation_one_hot = np.zeros(len(formation_names), dtype=np.float32)
        formation_one_hot[formation_id] = 1.0
    
    # Generate state features geometrically (no physics needed)
    ep_states = []
    ep_targets = []
    ep_labels = []
    
    for i in range(num_drones):
        # Synthetic velocity/angular features (zero or small random for realism)
        local_lin_vel = np.zeros(3, dtype=np.float32)
        local_ang_vel = np.zeros(3, dtype=np.float32)
        
        # Add small random noise if desired (mimicking stationary drones with sensor noise)
        if noisy_sensors:
            local_lin_vel = np.random.normal(0, noise_variance, size=3).astype(np.float32)
            local_ang_vel = np.random.normal(0, noise_variance, size=3).astype(np.float32)
        
        # LiDAR features (3D geometric fallback for residual correction)
        obs_features = compute_lidar_features(
            start_pos[i:i+1], start_orn[i:i+1], obstacles, obstacle_radii, physics_client=None
        )[0]
        
        # Build node features
        gnn_input_state = np.concatenate([local_lin_vel, local_ang_vel, obs_features])
        if include_formation_in_state and formation_one_hot is not None:
            gnn_input_state = np.concatenate([gnn_input_state, formation_one_hot])
        
        # Compute residual label
        target_global_pos = np.array([setpoints[i, 0], setpoints[i, 1], setpoints[i, 3]])
        # Handle case where naive_offsets might be empty (use zero offset)
        if naive_offsets is not None and col_ind is not None and len(naive_offsets) > 0:
            naive_offset = naive_offsets[col_ind[i]]
        else:
            naive_offset = np.zeros(3, dtype=np.float32)
        naive_global_pos = np.array([
            start_pos_center[0] + naive_offset[0],
            start_pos_center[1] + naive_offset[1],
            target_global_pos[2],
        ])
        y_label = target_global_pos - naive_global_pos
        
        # PWM labels (zeros for residual correction)
        motor_pwm_labels = np.zeros(4, dtype=np.float32)
        
        ep_states.append(gnn_input_state)
        ep_targets.append(y_label)
        ep_labels.append(motor_pwm_labels)
    
    # Build edges with zero velocities (no physics simulation)
    zero_velocities = np.zeros_like(start_pos, dtype=np.float32)
    edges, edge_attrs = build_edges(start_pos, communication_radius, zero_velocities)
    
    # Create graph
    x = torch.as_tensor(np.asarray(ep_states), dtype=torch.float32)
    target = torch.as_tensor(np.asarray(ep_targets), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(ep_labels), dtype=torch.float32)
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.as_tensor(np.asarray(edge_attrs), dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
    
    graph = Data(
        x=x, target=target, y=y, edge_index=edge_index, edge_attr=edge_attr,
        pos=torch.as_tensor(start_pos, dtype=torch.float32),
        obstacles=torch.as_tensor(obstacles, dtype=torch.float32),
        obstacle_radii=torch.as_tensor(obstacle_radii, dtype=torch.float32),
        formation_id=torch.tensor([formation_id], dtype=torch.long),
        episode_id=torch.tensor([0], dtype=torch.long),
        step_idx=torch.tensor([0], dtype=torch.long),
        num_drones=torch.tensor([num_drones], dtype=torch.long),
    )
    
    return graph, {
        "num_drones": num_drones,
        "formation_name": formation_name,
        "has_obstacles": len(obstacles) > 0,
        "max_residual_norm": float(np.max(np.linalg.norm(ep_targets, axis=1))),
    }


# ---------------------------------------------------------
# MULTIPROCESSING WORKER
# ---------------------------------------------------------
def simulate_episode(config: dict):
    """Worker function to simulate a single episode (legacy, kept for compatibility).
    Note: For batch processing, use simulate_episode_batch instead.
    """
    
    # Use fast geometric generation for residual correction (no physics simulation)
    if config["task_type"] == "residual_correction":
        # Inline residual correction generation (migrated from generate_residual_correction_batch)
        rng = np.random.default_rng(config["episode_seed"])
        graphs = []
        stats = {"count_near_zero": 0, "count_significant": 0}
        samples_per_seed = config.get("residual_samples_per_seed", 10)
        
        for sample_idx in range(samples_per_seed):
            num_drones = int(rng.integers(10, 21))
            episode_dataset_type = config["dataset_type"]
            if episode_dataset_type in {"mixed_formations", "mixed", "formations"}:
                formation_name = str(rng.choice(config["mixed_formation_types"]))
            else:
                formation_name = resolve_formation_name(episode_dataset_type)
                if formation_name is None:
                    formation_name = str(rng.choice(config["mixed_formation_types"]))
            
            num_obs_conf = config["num_obstacles"]
            if isinstance(num_obs_conf, (tuple, list)):
                current_num_obstacles = int(rng.integers(num_obs_conf[0], num_obs_conf[1] + 1))
            else:
                current_num_obstacles = num_obs_conf
            
            xy_limit = config["base_xy_limit"] * config["split_spread_scale"]
            
            graph, sample_stats = generate_residual_correction_sample(
                rng=rng,
                num_drones=num_drones,
                formation_name=formation_name,
                xy_limit=xy_limit,
                altitude_range=config["altitude_range"],
                num_obstacles=current_num_obstacles,
                obstacle_radius=config["obstacle_radius"],
                obstacle_radius_range=config.get("obstacle_radius_range"),
                communication_radius=config["communication_radius"],
                include_formation_in_state=config["include_formation_in_state"],
                noisy_sensors=config["noisy_sensors"],
                noise_variance=config["noise_variance"],
            )
            
            is_near_zero = sample_stats["max_residual_norm"] < config["residual_dropout_threshold"]
            if is_near_zero:
                total_potential = stats["count_near_zero"] + stats["count_significant"] + 1
                if total_potential > 10 and (stats["count_near_zero"] + 1) / total_potential > config["residual_balance_ratio"]:
                    continue
                stats["count_near_zero"] += 1
            else:
                stats["count_significant"] += 1
            
            graphs.append(graph)
        len_graphs = len(graphs)
        temp_file = os.path.join(config["temp_dir"], f"ep_{config['global_episode_id']}.safetensors")
        save_object_safetensors(temp_file, graphs)   # write immediately
        del graphs 
        
        episode_record = {
            "episode_id": config["global_episode_id"],
            "split": config["split_name"],
            "split_episode_idx": config["split_episode_idx"],
            "episode_seed": config["episode_seed"],
            "num_drones": len_graphs,
            "episode_dataset_type": config["dataset_type"],
            "formation_name": "mixed",
            "initial_xy_limit": config["base_xy_limit"] * config["split_spread_scale"],
            "initial_xy_radius": 0.0,
            "total_steps": samples_per_seed,
            "saved_steps": len_graphs,
            "converged": False,
        }
        
        return config["split_name"], temp_file, stats["count_near_zero"], stats["count_significant"], episode_record
    
    # Physics-based simulation for other task types
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

    obstacles, obstacle_radii = sample_obstacles(
        rng,
        current_num_obstacles,
        xy_limit=5.0,
        z_limit=config["altitude_range"],
        obstacle_radius=config["obstacle_radius"],
        obstacle_radius_range=config.get("obstacle_radius_range"),
    )

    # Graphical MUST be false for parallel processes
    env = create_aviary(
        start_pos,
        start_orn,
        config["environmental_wind"],
        obstacles,
        obstacle_radii,
        config["obstacle_radius"],
        False,
    )
    setpoints, col_ind, naive_offsets = build_setpoints(
        episode_dataset_type,
        start_pos,
        start_orn,
        rng,
        obstacles,
        obstacle_radii,
        config["obstacle_radius"],
    )
    
    active_drones = list(range(num_drones))
    formation_name = resolve_formation_name(episode_dataset_type)
    formation_id = FORMATION_TO_ID[formation_name] if formation_name else -1
    start_pos_center = np.mean(start_pos[:, :2], axis=0)

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
    is_converged = False
    steps_taken = 0

    for step_idx in range(max_steps):
        steps_taken = step_idx + 1
        steps_since_last_event += 1

        should_inject_failure = (config["inject_failures"] and not failure_injected_this_episode and step_idx >= max_steps // 2)
        if should_inject_failure and len(active_drones) > 2:
            failed_drone = active_drones.pop()
            env.drones[failed_drone].set_mode(0)
            active_start_pos = np.array([env.drones[idx].state[3] for idx in active_drones])
            setpoints, col_ind, naive_offsets = build_setpoints(
                episode_dataset_type,
                active_start_pos,
                start_orn[: len(active_drones)],
                rng,
                obstacles,
                obstacle_radii,
                config["obstacle_radius"],
            )
            for i, drone_idx in enumerate(active_drones):
                env.drones[drone_idx].set_mode(7)
                env.set_setpoint(drone_idx, setpoints[i])
            failure_injected_this_episode = True
            steps_since_last_event = 0
            already_converged_for_segment = False

        if config["dynamic_formation"] and step_idx > 0 and step_idx % 100 == 0:
            candidate_shapes = tuple(shape for shape in FORMATION_NAMES if shape != formation_name)
            if len(candidate_shapes) == 0:
                candidate_shapes = FORMATION_NAMES
            next_shape = str(rng.choice(candidate_shapes))
            formation_name = next_shape
            formation_id = FORMATION_TO_ID[formation_name]
            formation_one_hot = np.zeros(len(FORMATION_NAMES), dtype=np.float32)
            formation_one_hot[formation_id] = 1.0
            active_start_pos = np.array([env.drones[idx].state[3] for idx in active_drones])
            setpoints, col_ind, naive_offsets = build_setpoints(
                formation_name,
                active_start_pos,
                start_orn[: len(active_drones)],
                rng,
                obstacles,
                obstacle_radii,
                config["obstacle_radius"],
            )
            for i, drone_idx in enumerate(active_drones): env.set_setpoint(drone_idx, setpoints[i])
            steps_since_last_event = 0
            already_converged_for_segment = False

        command_setpoints = setpoints
        if task_type == "setpoint_prediction" and config.get("apf_enabled", True):
            current_positions = np.array([env.drones[idx].state[3] for idx in active_drones], dtype=np.float32)
            command_setpoints = compute_apf_setpoints(
                current_positions=current_positions,
                final_setpoints=setpoints,
                obstacles=obstacles,
                obstacle_radii=obstacle_radii,
                attractive_gain=config.get("apf_attractive_gain", 0.8),
                repulsive_gain=config.get("apf_repulsive_gain", 1.2),
                repulsion_padding=config.get("apf_repulsion_padding", 2.5),
                max_step_size=config.get("apf_max_step_size", 0.35),
                vertical_gain=config.get("apf_vertical_gain", 0.5),
            )
            for i, drone_idx in enumerate(active_drones):
                env.set_setpoint(drone_idx, command_setpoints[i])

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
                label_setpoints = command_setpoints if task_type == "setpoint_prediction" else setpoints
                
                ep_states, ep_targets, ep_labels, edges, edge_attrs, glob_pos = collect_step_data(
                    env,
                    active_drones,
                    label_setpoints,
                    col_ind,
                    naive_offsets,
                    task_type,
                    config["noisy_sensors"],
                    config["noise_variance"],
                    config["communication_radius"],
                    formation_one_hot,
                    obstacles,
                    obstacle_radii,
                    config["include_formation_in_state"],
                    start_pos_center,
                )

                should_skip_step = False

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
        obstacle_radii=obstacle_radii,
    )
    
    # Save the generated graphs for this episode to a temporary file on disk (bypassing IPC serialization bottlenecks)
    temp_file = os.path.join(config["temp_dir"], f"ep_{config['global_episode_id']}.safetensors")
    save_object_safetensors(temp_file, graphs)

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
        "total_steps": steps_taken,
        "saved_steps": saved_steps,
        "converged": bool(is_converged),
    }

    return config["split_name"], temp_file, count_near_zero, count_significant, episode_record



def save_dataset_shard(dataset_path, all_graphs, formation_names, split_name):
    data, slices = InMemoryDataset.collate(all_graphs)
    save_object_safetensors(
        dataset_path,
        {"data": data, "slices": slices, "formation_names": formation_names, "split_name": split_name},
    )

def simulate_worker_chunk(config: dict):
    """Worker function to simulate a batch of episodes and save a dataset shard directly."""
    split_name = config["split_name"]
    episode_configs = config["episode_configs"]
    task_type = config["task_type"]
    chunk_graphs = []
    episode_records = []
    total_count_near_zero = 0
    total_count_significant = 0

    for ep_idx, ep_config in enumerate(episode_configs):
        single_config = {
            **config,
            "split_name": split_name,
            "split_episode_idx": ep_config["split_episode_idx"],
            "episode_seed": ep_config["episode_seed"],
            "global_episode_id": ep_config["global_episode_id"],
            "split_spread_scale": config["spread_scale"],
        }
        if task_type == "residual_correction":
            rng = np.random.default_rng(ep_config["episode_seed"])
            samples_per_seed = config.get("residual_samples_per_seed", 10)
            episode_saved_steps = 0
            for sample_idx in range(samples_per_seed):
                graph, sample_stats = generate_residual_correction_sample(
                    rng=rng,
                    num_drones=int(rng.integers(10, 21)),
                    formation_name=str(rng.choice(config["mixed_formation_types"])) if config["dataset_type"] in {"mixed_formations", "mixed", "formations"} else resolve_formation_name(config["dataset_type"]) or str(rng.choice(config["mixed_formation_types"])),
                    xy_limit=config["base_xy_limit"] * config["spread_scale"],
                    altitude_range=config["altitude_range"],
                    num_obstacles=int(rng.integers(*config["num_obstacles"])) if isinstance(config["num_obstacles"], (tuple, list)) else config["num_obstacles"],
                    obstacle_radius=config["obstacle_radius"],
                    obstacle_radius_range=config.get("obstacle_radius_range"),
                    communication_radius=config["communication_radius"],
                    include_formation_in_state=config["include_formation_in_state"],
                    noisy_sensors=config["noisy_sensors"],
                    noise_variance=config["noise_variance"],
                )
                is_near_zero = sample_stats["max_residual_norm"] < config["residual_dropout_threshold"]
                if is_near_zero:
                    total_potential = total_count_near_zero + total_count_significant + 1
                    if total_potential > 10 and (total_count_near_zero + 1) / total_potential > config["residual_balance_ratio"]:
                        continue
                    total_count_near_zero += 1
                else:
                    total_count_significant += 1
                graph.episode_id = torch.tensor([ep_config["global_episode_id"]], dtype=torch.long)
                chunk_graphs.append(graph)
                episode_saved_steps += 1
            episode_records.append({
                "episode_id": ep_config["global_episode_id"],
                "split": split_name,
                "split_episode_idx": ep_config["split_episode_idx"],
                "episode_seed": ep_config["episode_seed"],
                "num_drones": episode_saved_steps,
                "episode_dataset_type": config["dataset_type"],
                "formation_name": "mixed",
                "initial_xy_limit": config["base_xy_limit"] * config["spread_scale"],
                "initial_xy_radius": 0.0,
                "total_steps": samples_per_seed,
                "saved_steps": episode_saved_steps,
                "converged": False,
            })
        else:
            rng = np.random.default_rng(ep_config["episode_seed"])
            num_drones = int(rng.integers(10, 21))
            episode_dataset_type = config["dataset_type"]
            if episode_dataset_type in {"mixed_formations", "mixed", "formations"}:
                episode_dataset_type = str(rng.choice(config["mixed_formation_types"]))
            xy_limit = config["base_xy_limit"] * config["spread_scale"]
            start_pos, start_orn = sample_episode_initial_conditions(
                num_drones, rng, xy_limit=xy_limit, altitude_range=config["altitude_range"]
            )
            num_obs_conf = config["num_obstacles"]
            if isinstance(num_obs_conf, tuple) or isinstance(num_obs_conf, list):
                current_num_obstacles = int(rng.integers(num_obs_conf[0], num_obs_conf[1] + 1))
            else:
                current_num_obstacles = num_obs_conf
            obstacles, obstacle_radii = sample_obstacles(
                rng,
                current_num_obstacles,
                xy_limit=5.0,
                z_limit=config["altitude_range"],
                obstacle_radius=config["obstacle_radius"],
                obstacle_radius_range=config.get("obstacle_radius_range"),
            )
            env = create_aviary(
                start_pos,
                start_orn,
                config["environmental_wind"],
                obstacles,
                obstacle_radii,
                config["obstacle_radius"],
                False,
            )
            setpoints, col_ind, naive_offsets = build_setpoints(
                episode_dataset_type,
                start_pos,
                start_orn,
                rng,
                obstacles,
                obstacle_radii,
                config["obstacle_radius"],
            )
            active_drones = list(range(num_drones))
            formation_name = resolve_formation_name(episode_dataset_type)
            formation_id = FORMATION_TO_ID[formation_name] if formation_name else -1
            start_pos_center = np.mean(start_pos[:, :2], axis=0)
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
            max_steps = config["max_steps"]
            is_converged = False
            steps_taken = 0
            for step_idx in range(max_steps):
                steps_taken = step_idx + 1
                steps_since_last_event += 1
                should_inject_failure = (config["inject_failures"] and not failure_injected_this_episode and step_idx >= max_steps // 2)
                if should_inject_failure and len(active_drones) > 2:
                    failed_drone = active_drones.pop()
                    env.drones[failed_drone].set_mode(0)
                    active_start_pos = np.array([env.drones[idx].state[3] for idx in active_drones])
                    setpoints, col_ind, naive_offsets = build_setpoints(
                        episode_dataset_type,
                        active_start_pos,
                        start_orn[: len(active_drones)],
                        rng,
                        obstacles,
                        obstacle_radii,
                        config["obstacle_radius"],
                    )
                    for i, drone_idx in enumerate(active_drones):
                        env.drones[drone_idx].set_mode(7)
                        env.set_setpoint(drone_idx, setpoints[i])
                    failure_injected_this_episode = True
                    steps_since_last_event = 0
                    already_converged_for_segment = False
                if config["dynamic_formation"] and step_idx > 0 and step_idx % 100 == 0:
                    candidate_shapes = tuple(shape for shape in FORMATION_NAMES if shape != formation_name)
                    if len(candidate_shapes) == 0:
                        candidate_shapes = FORMATION_NAMES
                    next_shape = str(rng.choice(candidate_shapes))
                    formation_name = next_shape
                    formation_id = FORMATION_TO_ID[formation_name]
                    formation_one_hot = np.zeros(len(FORMATION_NAMES), dtype=np.float32)
                    formation_one_hot[formation_id] = 1.0
                    active_start_pos = np.array([env.drones[idx].state[3] for idx in active_drones])
                    setpoints, col_ind, naive_offsets = build_setpoints(
                        formation_name,
                        active_start_pos,
                        start_orn[: len(active_drones)],
                        rng,
                        obstacles,
                        obstacle_radii,
                        config["obstacle_radius"],
                    )
                    for i, drone_idx in enumerate(active_drones): env.set_setpoint(drone_idx, setpoints[i])
                    steps_since_last_event = 0
                    already_converged_for_segment = False
                command_setpoints = setpoints
                if task_type == "setpoint_prediction" and config.get("apf_enabled", True):
                    current_positions = np.array([env.drones[idx].state[3] for idx in active_drones], dtype=np.float32)
                    command_setpoints = compute_apf_setpoints(
                        current_positions=current_positions,
                        final_setpoints=setpoints,
                        obstacles=obstacles,
                        obstacle_radii=obstacle_radii,
                        attractive_gain=config.get("apf_attractive_gain", 0.8),
                        repulsive_gain=config.get("apf_repulsive_gain", 1.2),
                        repulsion_padding=config.get("apf_repulsion_padding", 2.5),
                        max_step_size=config.get("apf_max_step_size", 0.35),
                        vertical_gain=config.get("apf_vertical_gain", 0.5),
                    )
                    for i, drone_idx in enumerate(active_drones):
                        env.set_setpoint(drone_idx, command_setpoints[i])
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
                        label_setpoints = command_setpoints if task_type == "setpoint_prediction" else setpoints
                        ep_states, ep_targets, ep_labels, edges, edge_attrs, glob_pos = collect_step_data(
                            env,
                            active_drones,
                            label_setpoints,
                            col_ind,
                            naive_offsets,
                            task_type,
                            config["noisy_sensors"],
                            config["noise_variance"],
                            config["communication_radius"],
                            formation_one_hot,
                            obstacles,
                            obstacle_radii,
                            config["include_formation_in_state"],
                            start_pos_center,
                        )
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
                env.step()
                if can_stop_episode: break
            env.disconnect()
            episode_graphs = convert_history_to_graphs(
                raw_history=raw_history,
                task_type=task_type,
                active_drones=active_drones,
                naive_offsets=naive_offsets,
                formation_id=formation_id,
                global_episode_id=ep_config["global_episode_id"],
                obstacles=obstacles,
                obstacle_radii=obstacle_radii,
            )
            chunk_graphs.extend(episode_graphs)
            episode_center = np.mean(start_pos[:, :2], axis=0)
            initial_xy_radius = float(np.max(np.linalg.norm(start_pos[:, :2] - episode_center, axis=1)))
            episode_records.append({
                "episode_id": ep_config["global_episode_id"],
                "split": split_name,
                "split_episode_idx": ep_config["split_episode_idx"],
                "episode_seed": ep_config["episode_seed"],
                "num_drones": num_drones,
                "episode_dataset_type": episode_dataset_type,
                "formation_name": formation_name,
                "initial_xy_limit": xy_limit,
                "initial_xy_radius": initial_xy_radius,
                "total_steps": steps_taken,
                "saved_steps": len(episode_graphs),
                "converged": bool(is_converged),
            })
    # Write the shard immediately and free RAM
    shard_path = os.path.join(config["temp_dir"], f"shard_{config['worker_id']}_split_{split_name}.safetensors")
    save_dataset_shard(shard_path, chunk_graphs, FORMATION_NAMES, split_name)
    del chunk_graphs
    gc.collect()
    return split_name, shard_path, total_count_near_zero, total_count_significant, episode_records

# ---------------------------------------------------------
# MAIN COORDINATOR
# ---------------------------------------------------------
def generate_dataset_parallel(
    worker_batch_size: int = 32,
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
    obstacle_radius_range: tuple[float, float] | None = None,
    residual_balance_ratio: float = 0.5,
    residual_dropout_threshold: float = 0.1,
    residual_samples_per_seed: int = 10,
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
    apf_enabled: bool = True,
    apf_attractive_gain: float = 0.8,
    apf_repulsive_gain: float = 1.2,
    apf_repulsion_padding: float = 2.5,
    apf_max_step_size: float = 0.35,
    apf_vertical_gain: float = 0.5,
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
    
    # Distribute episodes across workers in batches for efficient I/O
    common_kwargs = {
        "dataset_type": dataset_type,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "num_obstacles": num_obstacles,
        "obstacle_radius": obstacle_radius,
        "obstacle_radius_range": obstacle_radius_range,
        "residual_balance_ratio": residual_balance_ratio,
        "residual_dropout_threshold": residual_dropout_threshold,
        "residual_samples_per_seed": residual_samples_per_seed,
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
        "apf_enabled": apf_enabled,
        "apf_attractive_gain": apf_attractive_gain,
        "apf_repulsive_gain": apf_repulsive_gain,
        "apf_repulsion_padding": apf_repulsion_padding,
        "apf_max_step_size": apf_max_step_size,
        "apf_vertical_gain": apf_vertical_gain,
        "temp_dir": temp_dir,
    }
    
    worker_tasks = distribute_episodes_to_workers(
        worker_batch_size=worker_batch_size,
        num_workers=num_workers,
        split_episode_counts=split_episode_counts,
        seed=seed,
        validation_spread_scale=validation_spread_scale,
        test_spread_scale=test_spread_scale,
        **common_kwargs
    )

    split_temp_files = {s: [] for s in SPLIT_NAMES}
    split_summaries = {
        s: {"num_episodes": split_episode_counts[s], "num_graphs": 0, "spread_scale": resolve_split_spread_scale(s, validation_spread_scale, test_spread_scale), "count_near_zero": 0, "count_significant": 0} 
        for s in SPLIT_NAMES
    }
    episode_records = []

    print(f"Launching {len(worker_tasks)} worker tasks across {num_workers} parallel processes...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(simulate_worker_chunk, task): task for task in worker_tasks}
        completed = 0
        with tqdm(total=len(worker_tasks), desc="Worker Tasks", unit="task", dynamic_ncols=True) as progress:
            for future in futures:
                pass  # Progress bar placeholder; results handled below

    print(f"Parallel Simulation finished in {time.time() - start_time:.2f}s. Aggregating datasets...")
    
    generated_files = {}
    for split_name in SPLIT_NAMES:
        shard_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith(f"shard_") and f"split_{split_name}" in f]
        if not shard_files:
            continue
        # No aggregation in main process; just record file names
        generated_files[split_name] = [os.path.basename(f) for f in shard_files]
        print(f"Generated {split_name} dataset shards: {generated_files[split_name]}")
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
            "obstacle_radius_range": list(obstacle_radius_range) if obstacle_radius_range is not None else None,
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
            "residual_samples_per_seed": residual_samples_per_seed,
            "apf_enabled": apf_enabled,
            "apf_attractive_gain": apf_attractive_gain,
            "apf_repulsive_gain": apf_repulsive_gain,
            "apf_repulsion_padding": apf_repulsion_padding,
            "apf_max_step_size": apf_max_step_size,
            "apf_vertical_gain": apf_vertical_gain,
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
        worker_batch_size=60,
        num_workers=6,
        dataset_name="setpoint_prediction_dataset_parallel_apf_40000",
        dataset_type="mixed_formations",
        task_type="setpoint_prediction",
        noisy_sensors=False,
        environmental_wind=False,
        dynamic_formation=False,
        inject_failures=False,
        communication_radius=10.0,
        include_formation_in_state=True,
        num_episodes=600,
        max_steps=1500,  # Generate 50 samples per seed (500 total samples)
        tapered_sampling=True,
        conv_stopping=True,
        conv_threshold=0.2,
        num_obstacles=(0, 10),  # Random obstacles between 0 and 10
        seed=12345,
        obstacle_radius_range=(0.4, 1.8),
        apf_enabled=True,
        apf_attractive_gain=1.0,
        apf_repulsive_gain=1.2,
        apf_repulsion_padding=0.8,
        apf_max_step_size=1.5,
        apf_vertical_gain=1.0,
        residual_balance_ratio=0.5,
        residual_dropout_threshold=0.1,
    )
    print(f"Done. Outputs: {generated_files}")
# python data_collection_parallel.py | grep -v 'argv\\[0\\]='
