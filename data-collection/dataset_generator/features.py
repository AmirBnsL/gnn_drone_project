import numpy as np
import pybullet as p
import torch
from torch_geometric.data import Data, HeteroData

def maybe_add_sensor_noise(global_pos, global_euler, local_lin_vel, local_ang_vel, noisy_sensors, noise_variance):
    if not noisy_sensors: return global_pos, global_euler, local_lin_vel, local_ang_vel
    global_pos = global_pos + np.random.normal(0, noise_variance, size=3)
    global_euler = global_euler + np.random.normal(0, noise_variance, size=3)
    local_lin_vel = local_lin_vel + np.random.normal(0, noise_variance, size=3)
    local_ang_vel = local_ang_vel + np.random.normal(0, noise_variance, size=3)
    return global_pos, global_euler, local_lin_vel, local_ang_vel

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
    pos = np.asarray(global_positions, dtype=np.float32)
    num_drones = pos.shape[0]

    if num_drones < 2:
        return [], []

    rel = pos[None, :, :] - pos[:, None, :]  # shape: (N, N, 3)
    not_self = ~np.eye(num_drones, dtype=bool)

    if np.isinf(communication_radius):
        mask = not_self
    else:
        dist_sq = np.einsum("ijk,ijk->ij", rel, rel)
        radius_sq = float(communication_radius) ** 2
        mask = (dist_sq <= radius_sq) & not_self

    src, dst = np.nonzero(mask)

    edges = np.column_stack((src, dst)).tolist()
    dists = np.linalg.norm(rel[src, dst], axis=1, keepdims=True)  # (E, 1)
    
    if global_velocities is not None and len(global_velocities) == num_drones:
        vel = np.asarray(global_velocities, dtype=np.float32)
        rel_vel = vel[None, :, :] - vel[:, None, :]  # (N, N, 3)
        rel_vel_attrs = rel_vel[src, dst]  # (E, 3)
        edge_attrs = np.hstack([rel[src, dst], dists, rel_vel_attrs]).tolist()
    else:
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

def convert_history_to_graphs(raw_history, task_type, active_drones, naive_offsets, formation_id, global_episode_id, obstacles, obstacle_radii):
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
