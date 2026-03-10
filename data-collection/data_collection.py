import os
import json
import numpy as np
import pybullet as p
import torch
from torch_geometric.data import Data, InMemoryDataset
from PyFlyt.core import Aviary


FORMATION_NAMES = ("a", "rectangle", "triangle")
FORMATION_TO_ID = {name: idx for idx, name in enumerate(FORMATION_NAMES)}
SPLIT_NAMES = ("train", "val", "test")
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}


# Define a simple wind function matching what you see in example 09
def wind_generator(time: float, position: np.ndarray):
    """Generates an upward draft with random turbulence."""
    wind = np.zeros_like(position)
    
    # Give a base updraft force plus randomness based on position
    wind[:, 2] = np.sin(time) * 0.5 + np.random.normal(0, 0.2, size=(len(position),))
    
    # Push slightly horizontally 
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
    start_pos[:, 2] = rng.uniform(altitude_range[0], altitude_range[1], size=(num_drones,))

    start_orn = np.zeros((num_drones, 3))
    start_orn[:, 2] = rng.uniform(-np.pi, np.pi, size=(num_drones,))

    return start_pos, start_orn


def resolve_formation_name(dataset_type: str):
    if dataset_type in {"a", "formation_a"}:
        return "a"
    if dataset_type in {"rectangle", "rectangular", "formation_rectangle"}:
        return "rectangle"
    if dataset_type in {"triangle", "formation_triangle"}:
        return "triangle"
    return None


def _formation_a_offsets(num_drones: int, spacing: float = 2.0):
    offsets = np.zeros((num_drones, 3))
    if num_drones == 1:
        return offsets

    offsets[0] = np.array([0.0, 0.0, 0.0])
    for idx in range(1, num_drones):
        level = (idx + 1) // 2
        side = -1.0 if idx % 2 == 1 else 1.0
        offsets[idx, 0] = side * level * spacing
        offsets[idx, 1] = level * spacing

    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets


def _formation_rectangle_offsets(num_drones: int, spacing: float = 2.0):
    cols = int(np.ceil(np.sqrt(num_drones)))
    rows = int(np.ceil(num_drones / cols))

    offsets = np.zeros((num_drones, 3))
    for idx in range(num_drones):
        row = idx // cols
        col = idx % cols
        offsets[idx, 0] = (col - (cols - 1) / 2.0) * spacing
        offsets[idx, 1] = (row - (rows - 1) / 2.0) * spacing

    return offsets


def _formation_triangle_offsets(num_drones: int, spacing: float = 2.0):
    offsets = np.zeros((num_drones, 3))

    count = 0
    row = 0
    while count < num_drones:
        points_in_row = row + 1
        row_y = row * spacing
        row_start_x = -0.5 * row * spacing

        for col in range(points_in_row):
            if count >= num_drones:
                break
            offsets[count, 0] = row_start_x + col * spacing
            offsets[count, 1] = row_y
            count += 1

        row += 1

    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets


def _build_formation_setpoints(formation_name: str, start_pos: np.ndarray):
    num_drones = len(start_pos)
    formation_center = np.mean(start_pos[:, :2], axis=0)
    target_altitude = np.mean(start_pos[:, 2])

    if formation_name == "a":
        offsets = _formation_a_offsets(num_drones)
    elif formation_name == "rectangle":
        offsets = _formation_rectangle_offsets(num_drones)
    elif formation_name == "triangle":
        offsets = _formation_triangle_offsets(num_drones)
    else:
        return None

    setpoints = np.zeros((num_drones, 4))
    setpoints[:, :2] = formation_center + offsets[:, :2]
    setpoints[:, 2] = 0.0
    setpoints[:, 3] = target_altitude
    return setpoints


def create_aviary(
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    environmental_wind: bool,
    graphical: bool = False,
):
    env = Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        drone_type="quadx",
        render=graphical,
    )

    if environmental_wind:
        env.register_wind_field_function(wind_generator)

    env.set_mode(7)
    return env


def build_setpoints(
    dataset_type: str,
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    rng: np.random.Generator,
):
    num_drones = len(start_pos)

    formation_name = resolve_formation_name(dataset_type)
    formation_setpoints = None
    if formation_name is not None:
        formation_setpoints = _build_formation_setpoints(formation_name, start_pos)
    if formation_setpoints is not None:
        return formation_setpoints

    if dataset_type == "hovering":
        setpoints = np.zeros((num_drones, 4))
        setpoints[:, :2] = start_pos[:, :2]
        setpoints[:, 2] = start_orn[:, 2]
        setpoints[:, 3] = start_pos[:, 2]
        return setpoints

    radius = 10.0 if dataset_type == "aggressive" else 5.0
    setpoints = np.zeros((num_drones, 4))
    setpoints[:, :2] = start_pos[:, :2] + rng.uniform(-radius, radius, size=(num_drones, 2))
    setpoints[:, 2] = rng.uniform(-np.pi, np.pi, size=(num_drones,))
    setpoints[:, 3] = rng.uniform(1.0, radius, size=(num_drones,))
    return setpoints


def compute_split_episode_counts(
    num_episodes: int,
    split_ratios: tuple[float, float, float],
):
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must contain exactly three values for train, val, test.")
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("split_ratios must sum to 1.0.")

    counts = [int(num_episodes * ratio) for ratio in split_ratios]
    remainder = num_episodes - sum(counts)
    for idx in range(remainder):
        counts[idx] += 1

    return {split_name: count for split_name, count in zip(SPLIT_NAMES, counts)}


def resolve_split_spread_scale(
    split_name: str,
    validation_spread_scale: float,
    test_spread_scale: float,
):
    if split_name == "val":
        return validation_spread_scale
    if split_name == "test":
        return test_spread_scale
    return 1.0


def should_sample_step(
    step_idx: int,
    max_steps: int,
    tapered_sampling: bool,
    dense_sampling_steps: int,
    mid_sampling_steps: int,
    mid_step_stride: int,
    late_step_stride: int,
):
    if not tapered_sampling or step_idx == max_steps - 1:
        return True

    mid_sampling_steps = max(mid_sampling_steps, dense_sampling_steps)

    if step_idx < dense_sampling_steps:
        return True
    if step_idx < mid_sampling_steps:
        return (step_idx - dense_sampling_steps) % mid_step_stride == 0
    return (step_idx - mid_sampling_steps) % late_step_stride == 0


def build_episode_seed(seed: int | None, split_name: str, split_episode_idx: int):
    if seed is None:
        return None
    return int(seed + SPLIT_SEED_OFFSETS[split_name] + split_episode_idx)


def write_dataset_metadata(metadata_path: str, metadata: dict):
    with open(metadata_path, "w", encoding="ascii") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")


def maybe_add_sensor_noise(
    global_pos: np.ndarray,
    global_euler: np.ndarray,
    local_lin_vel: np.ndarray,
    local_ang_vel: np.ndarray,
    noisy_sensors: bool,
    noise_variance: float,
):
    if not noisy_sensors:
        return global_pos, global_euler, local_lin_vel, local_ang_vel

    global_pos = global_pos + np.random.normal(0, noise_variance, size=3)
    global_euler = global_euler + np.random.normal(0, noise_variance, size=3)
    local_lin_vel = local_lin_vel + np.random.normal(0, noise_variance, size=3)
    local_ang_vel = local_ang_vel + np.random.normal(0, noise_variance, size=3)
    return global_pos, global_euler, local_lin_vel, local_ang_vel


def build_drone_features(
    drone,
    setpoint: np.ndarray,
    noisy_sensors: bool,
    noise_variance: float,
    formation_one_hot: np.ndarray = None,
    include_formation_in_state: bool = True,
):
    state = drone.state
    global_pos = np.array(state[3], copy=True)
    global_euler = np.array(state[1], copy=True)
    local_lin_vel = np.array(state[2], copy=True)
    local_ang_vel = np.array(state[0], copy=True)

    global_pos, global_euler, local_lin_vel, local_ang_vel = maybe_add_sensor_noise(
        global_pos,
        global_euler,
        local_lin_vel,
        local_ang_vel,
        noisy_sensors,
        noise_variance,
    )

    target_global_pos = np.array([setpoint[0], setpoint[1], setpoint[3]])
    target_global_yaw = setpoint[2]

    global_pos_error = target_global_pos - global_pos
    yaw_error = target_global_yaw - global_euler[2]
    yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

    rotation_quaternion = p.getQuaternionFromEuler(global_euler)
    rot_matrix = np.array(p.getMatrixFromQuaternion(rotation_quaternion)).reshape(3, 3)
    local_pos_error = rot_matrix.T @ global_pos_error

    gnn_input_state = np.concatenate([local_lin_vel, local_ang_vel])
    if include_formation_in_state and formation_one_hot is not None:
        gnn_input_state = np.concatenate([gnn_input_state, formation_one_hot])

    gnn_input_target = np.concatenate([local_pos_error, np.array([yaw_error])])
    motor_pwm_labels = drone.pwm

    return gnn_input_state, gnn_input_target, motor_pwm_labels, global_pos


def build_edges(global_positions: np.ndarray, communication_radius: float):
    edges = []
    num_drones = len(global_positions)

    for i in range(num_drones):
        for j in range(num_drones):
            if i == j:
                continue

            dist = np.linalg.norm(global_positions[i] - global_positions[j])
            if dist <= communication_radius:
                edges.append([i, j])

    return edges


def collect_step_data(
    env,
    setpoints,
    noisy_sensors: bool,
    noise_variance: float,
    communication_radius: float,
    formation_one_hot: np.ndarray = None,
    include_formation_in_state: bool = True,
):
    episode_states = []
    episode_targets = []
    episode_labels = []
    global_positions = []

    for i, drone in enumerate(env.drones):
        gnn_input_state, gnn_input_target, motor_pwm_labels, global_pos = build_drone_features(
            drone,
            setpoints[i],
            noisy_sensors,
            noise_variance,
            formation_one_hot,
            include_formation_in_state,
        )
        episode_states.append(gnn_input_state)
        episode_targets.append(gnn_input_target)
        episode_labels.append(motor_pwm_labels)
        global_positions.append(global_pos)

    edges = build_edges(np.array(global_positions), communication_radius)
    return episode_states, episode_targets, episode_labels, edges


def save_dataset(
    dataset_path: str,
    all_graphs,
    formation_names,
    split_name: str,
):
    data, slices = InMemoryDataset.collate(all_graphs)
    torch.save(
        {
            "data": data,
            "slices": slices,
            "formation_names": formation_names,
            "split_name": split_name,
        },
        dataset_path,
    )


def generate_dataset(
    num_episodes=200,
    max_steps=300,
    dataset_name="formation_dataset",
    dataset_type="mixed_formations",  # 'random', 'hovering', 'aggressive', 'a', 'rectangle', 'triangle', 'mixed_formations'
    noisy_sensors=False,
    noise_variance=0.01,
    environmental_wind=False,
    graphical=False,
    communication_radius=np.inf,  # For edge formation
    include_formation_in_state=True,
    mixed_formation_types=FORMATION_NAMES,
    split_ratios=(0.8, 0.1, 0.1),
    seed=12345,
    base_xy_limit=10.0,
    altitude_range=(0.5, 5.0),
    validation_spread_scale=1.25,
    test_spread_scale=1.5,
    tapered_sampling=True,
    dense_sampling_steps=120,
    mid_sampling_steps=240,
    mid_step_stride=2,
    late_step_stride=5,
):
    """
    Generate a multi-drone imitation-learning dataset from the PyFlyt PID controller.

    The simulator runs at the controller loop rate (120 Hz), and each collected sample
    contains node features, local target errors, PWM labels, and graph edges.
    Data is represented in local body-frame conventions (ENU-aligned state usage).

    Parameters
    ----------
    num_episodes : int, default=200
        Number of episodes to simulate.
    max_steps : int, default=300
        Number of simulation steps collected per episode.
    dataset_name : str, default="formation_dataset"
        Base name used in the output file path:
        ``datasets/{dataset_name}_{dataset_type}_{split}.pt``.
    dataset_type : str, default="mixed_formations"
        Task/setpoint mode for each episode.
        Supported values include:
        - ``random``: random nearby targets.
        - ``hovering``: hold current position/yaw.
        - ``aggressive``: wider random target range.
        - ``a``, ``rectangle``, ``triangle``: fixed geometric formations.
        - ``mixed_formations`` (also ``mixed``, ``formations``): choose one
          formation type randomly per episode from ``mixed_formation_types``.
    noisy_sensors : bool, default=False
        If True, adds Gaussian noise to position, Euler angle, linear velocity,
        and angular velocity before building model inputs.
    noise_variance : float, default=0.01
        Standard deviation used for Gaussian sensor noise when
        ``noisy_sensors=True``.
    environmental_wind : bool, default=False
        If True, registers ``wind_generator`` as an external wind field in Aviary.
    graphical : bool, default=False
        If True, runs episodes with the PyBullet GUI enabled (rendering visible).
        If False, runs headless for faster data collection.
    communication_radius : float, default=np.inf
        Maximum inter-drone distance for adding directed communication edges.
        Use ``np.inf`` for fully connected (excluding self-loops).
    include_formation_in_state : bool, default=True
        If True, concatenates a formation one-hot vector to each node state when
        the active episode uses a recognized formation.
    mixed_formation_types : tuple[str, ...], default=FORMATION_NAMES
        Candidate formation names sampled in mixed mode. Typical values:
        ``("a", "rectangle", "triangle")``.
    split_ratios : tuple[float, float, float], default=(0.8, 0.1, 0.1)
        Episode-level split ratios for train, validation, and test datasets.
        Splitting is done by episode, never by individual graph.
    seed : int | None, default=12345
        Base seed used to derive per-split unseen episode seeds.
    base_xy_limit : float, default=10.0
        Base XY spawn range for training episodes.
    altitude_range : tuple[float, float], default=(0.5, 5.0)
        Inclusive altitude sampling range for episode initialization.
    validation_spread_scale : float, default=1.25
        Multiplier applied to ``base_xy_limit`` for validation episodes.
        This creates harder validation subsets with larger initial spreads.
    test_spread_scale : float, default=1.5
        Multiplier applied to ``base_xy_limit`` for test episodes.
    tapered_sampling : bool, default=True
        If True, save more steps early in the rollout and fewer after convergence.
    dense_sampling_steps : int, default=120
        Number of initial steps kept at full resolution when tapered sampling is enabled.
    mid_sampling_steps : int, default=240
        Absolute step index where sampling switches from the mid stride to the late stride.
    mid_step_stride : int, default=2
        Save every ``mid_step_stride`` step between ``dense_sampling_steps`` and
        ``mid_sampling_steps``.
    late_step_stride : int, default=5
        Save every ``late_step_stride`` step after ``mid_sampling_steps``.

    Returns
    -------
    None
        Writes PyTorch Geometric ``.pt`` datasets to disk with collated
        ``Data`` objects for train, validation, and test splits. Each graph contains:
        - ``x``: node features,
        - ``target``: local target errors,
        - ``y``: PWM labels,
        - ``edge_index``: graph connectivity,
        - ``formation_id``: graph-level formation class id,
        - ``episode_id``: global rollout identifier,
        - ``step_idx``: step index within the rollout,
        - ``num_drones``: number of drones in the rollout.
        A JSON sidecar is also written to describe the exact generation recipe.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    datasets_dir = os.path.join(repo_root, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    dataset_prefix = os.path.join(datasets_dir, f"{dataset_name}_{dataset_type}")
    split_episode_counts = compute_split_episode_counts(num_episodes, split_ratios)
    split_graphs = {split_name: [] for split_name in SPLIT_NAMES}
    split_summaries = dict()
    episode_records = []

    global_episode_id = 0
    for split_name in SPLIT_NAMES:
        split_episode_count = split_episode_counts[split_name]
        split_spread_scale = resolve_split_spread_scale(
            split_name,
            validation_spread_scale,
            test_spread_scale,
        )
        split_summaries[split_name] = {
            "num_episodes": split_episode_count,
            "num_graphs": 0,
            "spread_scale": split_spread_scale,
        }

        for split_episode_idx in range(split_episode_count):
            episode_seed = build_episode_seed(seed, split_name, split_episode_idx)
            rng = np.random.default_rng(episode_seed)

            print(
                f"[{dataset_name}] Starting {split_name} episode {split_episode_idx + 1}/{split_episode_count} ..."
            )

            num_drones = int(rng.integers(10, 21))
            episode_dataset_type = dataset_type
            if dataset_type in {"mixed_formations", "mixed", "formations"}:
                episode_dataset_type = str(rng.choice(mixed_formation_types))

            xy_limit = base_xy_limit * split_spread_scale
            start_pos, start_orn = sample_episode_initial_conditions(
                num_drones,
                rng,
                xy_limit=xy_limit,
                altitude_range=altitude_range,
            )
            env = create_aviary(start_pos, start_orn, environmental_wind, graphical)
            setpoints = build_setpoints(episode_dataset_type, start_pos, start_orn, rng)

            formation_name = resolve_formation_name(episode_dataset_type)
            formation_id = -1
            if formation_name is not None:
                formation_id = FORMATION_TO_ID[formation_name]
            formation_one_hot = None
            if formation_id >= 0:
                formation_one_hot = np.zeros(len(FORMATION_NAMES), dtype=np.float32)
                formation_one_hot[formation_id] = 1.0

            print(
                f"[{dataset_name}] {split_name} episode {split_episode_idx + 1} formation = {formation_name if formation_name is not None else episode_dataset_type}"
            )

            env.set_all_setpoints(setpoints)

            saved_steps = 0
            for step_idx in range(max_steps):
                if should_sample_step(
                    step_idx,
                    max_steps,
                    tapered_sampling,
                    dense_sampling_steps,
                    mid_sampling_steps,
                    mid_step_stride,
                    late_step_stride,
                ):
                    episode_states, episode_targets, episode_labels, edges = collect_step_data(
                        env,
                        setpoints,
                        noisy_sensors,
                        noise_variance,
                        communication_radius,
                        formation_one_hot,
                        include_formation_in_state,
                    )

                    x = torch.as_tensor(np.asarray(episode_states), dtype=torch.float32)
                    target = torch.as_tensor(np.asarray(episode_targets), dtype=torch.float32)
                    y = torch.as_tensor(np.asarray(episode_labels), dtype=torch.float32)

                    if edges:
                        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    else:
                        edge_index = torch.empty((2, 0), dtype=torch.long)

                    graph = Data(
                        x=x,
                        target=target,
                        y=y,
                        edge_index=edge_index,
                        formation_id=torch.tensor([formation_id], dtype=torch.long),
                        episode_id=torch.tensor([global_episode_id], dtype=torch.long),
                        step_idx=torch.tensor([step_idx], dtype=torch.long),
                        num_drones=torch.tensor([num_drones], dtype=torch.long),
                    )
                    split_graphs[split_name].append(graph)
                    saved_steps += 1

                env.step()

            env.disconnect()

            split_summaries[split_name]["num_graphs"] += saved_steps
            episode_center = np.mean(start_pos[:, :2], axis=0)
            initial_xy_radius = float(
                np.max(np.linalg.norm(start_pos[:, :2] - episode_center, axis=1))
            )
            episode_records.append(
                {
                    "episode_id": global_episode_id,
                    "split": split_name,
                    "split_episode_idx": split_episode_idx,
                    "episode_seed": episode_seed,
                    "num_drones": num_drones,
                    "episode_dataset_type": episode_dataset_type,
                    "formation_name": formation_name,
                    "initial_xy_limit": xy_limit,
                    "initial_xy_radius": initial_xy_radius,
                    "saved_steps": saved_steps,
                }
            )
            global_episode_id += 1

    generated_files = dict()
    for split_name, graphs in split_graphs.items():
        if not graphs:
            continue

        split_dataset_path = f"{dataset_prefix}_{split_name}.pt"
        save_dataset(
            split_dataset_path,
            graphs,
            FORMATION_NAMES,
            split_name,
        )
        generated_files[split_name] = os.path.basename(split_dataset_path)
        print(f"✅ Generated {split_name} dataset -> {split_dataset_path}")

    metadata_path = f"{dataset_prefix}_metadata.json"
    metadata = {
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "generated_files": generated_files,
        "formation_names": list(FORMATION_NAMES),
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "noisy_sensors": noisy_sensors,
            "noise_variance": noise_variance,
            "environmental_wind": environmental_wind,
            "graphical": graphical,
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
        },
        "split_summary": split_summaries,
        "episodes": episode_records,
    }
    write_dataset_metadata(metadata_path, metadata)
    print(f"✅ Generated dataset metadata -> {metadata_path}")


if __name__ == "__main__":
    # Generate a mixed-formation dataset with episode-level splits and tapered sampling.
    generate_dataset(
        num_episodes=240,
        max_steps=300,
        dataset_name="formation_mixed_comm_10m",
        dataset_type="mixed_formations",
        communication_radius=10.0,
        include_formation_in_state=True,
        tapered_sampling=True,
    )
