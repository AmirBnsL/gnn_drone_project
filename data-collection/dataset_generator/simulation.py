import numpy as np
import torch
from torch_geometric.data import Data

from dataset_generator.constants import FORMATION_NAMES, FORMATION_TO_ID
from dataset_generator.utils import sample_episode_initial_conditions, sample_obstacles, should_sample_step
from dataset_generator.formations import resolve_formation_name
from dataset_generator.environment import create_aviary, build_setpoints, compute_apf_setpoints
from dataset_generator.features import compute_lidar_features, build_edges, collect_step_data, convert_history_to_graphs

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
    """Generate a single residual correction sample WITHOUT physics simulation."""
    start_pos, start_orn = sample_episode_initial_conditions(
        num_drones, rng, xy_limit=xy_limit, altitude_range=altitude_range
    )
    
    obstacles, obstacle_radii = sample_obstacles(
        rng, num_obstacles, xy_limit=5.0, z_limit=altitude_range, obstacle_radius=obstacle_radius, obstacle_radius_range=obstacle_radius_range
    )
    
    setpoints, col_ind, naive_offsets = build_setpoints(
        formation_name, start_pos, start_orn, rng, obstacles, obstacle_radii, obstacle_radius
    )
    
    formation_id = FORMATION_TO_ID.get(formation_name, -1)
    start_pos_center = np.mean(start_pos[:, :2], axis=0)
    
    formation_one_hot = None
    if formation_id >= 0 and include_formation_in_state:
        formation_one_hot = np.zeros(len(formation_names), dtype=np.float32)
        formation_one_hot[formation_id] = 1.0
    
    ep_states, ep_targets, ep_labels = [], [], []
    
    for i in range(num_drones):
        local_lin_vel = np.zeros(3, dtype=np.float32)
        local_ang_vel = np.zeros(3, dtype=np.float32)
        
        if noisy_sensors:
            local_lin_vel = np.random.normal(0, noise_variance, size=3).astype(np.float32)
            local_ang_vel = np.random.normal(0, noise_variance, size=3).astype(np.float32)
        
        obs_features = compute_lidar_features(
            start_pos[i:i+1], start_orn[i:i+1], obstacles, obstacle_radii, physics_client=None
        )[0]
        
        gnn_input_state = np.concatenate([local_lin_vel, local_ang_vel, obs_features])
        if include_formation_in_state and formation_one_hot is not None:
            gnn_input_state = np.concatenate([gnn_input_state, formation_one_hot])
        
        target_global_pos = np.array([setpoints[i, 0], setpoints[i, 1], setpoints[i, 3]])
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
        motor_pwm_labels = np.zeros(4, dtype=np.float32)
        
        ep_states.append(gnn_input_state)
        ep_targets.append(y_label)
        ep_labels.append(motor_pwm_labels)
    
    zero_velocities = np.zeros_like(start_pos, dtype=np.float32)
    edges, edge_attrs = build_edges(start_pos, communication_radius, zero_velocities)
    
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

def run_physics_episode(ep_config: dict, config: dict):
    """Runs a single episode loop for tasks requiring physics (setpoint_prediction, etc)."""
    rng = np.random.default_rng(ep_config["episode_seed"])
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

    env = create_aviary(
        start_pos,
        start_orn,
        config["environmental_wind"],
        obstacles,
        obstacle_radii,
        config["obstacle_radius"],
        graphical=False,
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
                episode_dataset_type, active_start_pos, start_orn[: len(active_drones)],
                rng, obstacles, obstacle_radii, config["obstacle_radius"],
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
                formation_name, active_start_pos, start_orn[: len(active_drones)],
                rng, obstacles, obstacle_radii, config["obstacle_radius"],
            )
            for i, drone_idx in enumerate(active_drones): env.set_setpoint(drone_idx, setpoints[i])
            steps_since_last_event = 0
            already_converged_for_segment = False

        command_setpoints = setpoints
        if task_type == "setpoint_prediction" and config.get("apf_enabled", True):
            current_positions = np.array([env.drones[idx].state[3] for idx in active_drones], dtype=np.float32)
            command_setpoints = compute_apf_setpoints(
                current_positions=current_positions, final_setpoints=setpoints,
                obstacles=obstacles, obstacle_radii=obstacle_radii,
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
                    env, active_drones, label_setpoints, col_ind, naive_offsets, task_type,
                    config["noisy_sensors"], config["noise_variance"], config["communication_radius"],
                    formation_one_hot, obstacles, obstacle_radii, config["include_formation_in_state"], start_pos_center,
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
        raw_history=raw_history, task_type=task_type, active_drones=active_drones,
        naive_offsets=naive_offsets, formation_id=formation_id,
        global_episode_id=ep_config["global_episode_id"],
        obstacles=obstacles, obstacle_radii=obstacle_radii,
    )

    episode_center = np.mean(start_pos[:, :2], axis=0)
    initial_xy_radius = float(np.max(np.linalg.norm(start_pos[:, :2] - episode_center, axis=1)))

    episode_record = {
        "episode_id": ep_config["global_episode_id"],
        "split": config["split_name"],
        "split_episode_idx": ep_config["split_episode_idx"],
        "episode_seed": ep_config["episode_seed"],
        "num_drones": num_drones,
        "episode_dataset_type": episode_dataset_type,
        "formation_name": formation_name,
        "initial_xy_limit": xy_limit,
        "initial_xy_radius": initial_xy_radius,
        "total_steps": steps_taken,
        "saved_steps": saved_steps,
        "converged": bool(is_converged),
    }

    return episode_graphs, episode_record
