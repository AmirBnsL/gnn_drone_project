import io
import json
import gc
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch_geometric.data import InMemoryDataset

from dataset_generator.constants import SPLIT_NAMES

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

def save_dataset_shard(dataset_path, all_graphs, formation_names, split_name):
    data, slices = InMemoryDataset.collate(all_graphs)
    save_object_safetensors(
        dataset_path,
        {"data": data, "slices": slices, "formation_names": formation_names, "split_name": split_name},
    )

def write_dataset_metadata(metadata_path, metadata):
    with open(metadata_path, "w", encoding="ascii") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")

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
