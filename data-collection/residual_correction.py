
import json
import os
import time
from typing import Literal

import numpy as np
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset

from data_collection_parallel import (
    FORMATION_NAMES,
    SPLIT_NAMES,
    SPLIT_SEED_OFFSETS,
    compute_split_episode_counts,
    generate_residual_correction_sample,
    resolve_formation_name,
    resolve_split_spread_scale,
    save_object_safetensors,
    write_dataset_metadata,
)


def build_episode_seed(seed: int | None, split_name: str, split_episode_idx: int):
    if seed is None:
        return None
    return int(seed + SPLIT_SEED_OFFSETS[split_name] + split_episode_idx)


def generate_residual_correction_batch_in_memory(config: dict):
    """
    Same residual batch logic as parallel worker version, but returns graphs directly
    (no temp shard writing).
    """
    rng = np.random.default_rng(config["episode_seed"])

    graphs = []
    stats = {"count_near_zero": 0, "count_significant": 0}
    samples_per_seed = config.get("residual_samples_per_seed", 10)

    for _ in range(samples_per_seed):
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
            current_num_obstacles = int(num_obs_conf)

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
            if (
                total_potential > 10
                and (stats["count_near_zero"] + 1) / total_potential > config["residual_balance_ratio"]
            ):
                continue
            stats["count_near_zero"] += 1
        else:
            stats["count_significant"] += 1

        graphs.append(graph)

    episode_record = {
        "episode_id": config["global_episode_id"],
        "split": config["split_name"],
        "split_episode_idx": config["split_episode_idx"],
        "episode_seed": config["episode_seed"],
        "num_drones": len(graphs),
        "episode_dataset_type": config["dataset_type"],
        "formation_name": "mixed",
        "initial_xy_limit": config["base_xy_limit"] * config["split_spread_scale"],
        "initial_xy_radius": 0.0,
        "total_steps": samples_per_seed,
        "saved_steps": len(graphs),
        "converged": False,
    }

    return graphs, stats["count_near_zero"], stats["count_significant"], episode_record


def generate_residual_dataset_singlepass(
    num_episodes: int = 50,
    dataset_name: str = "residual_correction_singlepass",
    dataset_type: Literal["mixed_formations", "mixed", "formations", "a", "rectangle", "triangle"] = "mixed_formations",
    num_obstacles: int | tuple[int, int] = 0,
    obstacle_radius: float = 1.0,
    obstacle_radius_range: tuple[float, float] | None = None,
    residual_balance_ratio: float = 0.5,
    residual_dropout_threshold: float = 0.1,
    residual_samples_per_seed: int = 10,
    noisy_sensors: bool = False,
    noise_variance: float = 0.01,
    communication_radius: float = np.inf,
    include_formation_in_state: bool = True,
    mixed_formation_types: tuple = FORMATION_NAMES,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 12345,
    base_xy_limit: float = 10.0,
    altitude_range: tuple[float, float] = (0.5, 5.0),
    validation_spread_scale: float = 1.25,
    test_spread_scale: float = 1.5,
):
    """
    Non-parallel residual-correction-only generation.
    No per-episode temp files; one final .safetensors save at end.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    datasets_dir = os.path.join(repo_root, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    split_episode_counts = compute_split_episode_counts(num_episodes, split_ratios)

    split_graphs = {s: [] for s in SPLIT_NAMES}
    split_summaries = {
        s: {
            "num_episodes": split_episode_counts[s],
            "num_graphs": 0,
            "spread_scale": resolve_split_spread_scale(s, validation_spread_scale, test_spread_scale),
            "count_near_zero": 0,
            "count_significant": 0,
        }
        for s in SPLIT_NAMES
    }
    episode_records = []

    tasks = []
    global_episode_id = 0
    for split_name in SPLIT_NAMES:
        count = split_episode_counts[split_name]
        spread_scale = resolve_split_spread_scale(split_name, validation_spread_scale, test_spread_scale)
        for idx in range(count):
            tasks.append(
                {
                    "split_name": split_name,
                    "split_episode_idx": idx,
                    "split_spread_scale": spread_scale,
                    "episode_seed": build_episode_seed(seed, split_name, idx),
                    "global_episode_id": global_episode_id,
                    "dataset_type": dataset_type,
                    "num_obstacles": num_obstacles,
                    "obstacle_radius": obstacle_radius,
                    "obstacle_radius_range": obstacle_radius_range,
                    "residual_balance_ratio": residual_balance_ratio,
                    "residual_dropout_threshold": residual_dropout_threshold,
                    "residual_samples_per_seed": residual_samples_per_seed,
                    "noisy_sensors": noisy_sensors,
                    "noise_variance": noise_variance,
                    "communication_radius": communication_radius,
                    "include_formation_in_state": include_formation_in_state,
                    "mixed_formation_types": mixed_formation_types,
                    "base_xy_limit": base_xy_limit,
                    "altitude_range": altitude_range,
                }
            )
            global_episode_id += 1

    start_time = time.time()
    print(f"Generating {len(tasks)} residual episodes sequentially (single-pass, no temp shards)...")

    with tqdm(tasks, desc="Residual Episodes", unit="ep", dynamic_ncols=True) as pbar:
        for _, task in enumerate(pbar, start=1):
            graphs, c_zero, c_sig, record = generate_residual_correction_batch_in_memory(task)
            split = task["split_name"]

            split_graphs[split].extend(graphs)
            split_summaries[split]["num_graphs"] += record["saved_steps"]
            split_summaries[split]["count_near_zero"] += c_zero
            split_summaries[split]["count_significant"] += c_sig
            episode_records.append(record)

            pbar.set_postfix(
                split=split,
                kept=record["saved_steps"],
                near0=split_summaries[split]["count_near_zero"],
                sig=split_summaries[split]["count_significant"],
            )

    # One final save only
    collated = {}
    for split_name in tqdm(SPLIT_NAMES, desc="Collating Splits", unit="split", dynamic_ncols=True):
        graphs = split_graphs[split_name]
        if len(graphs) == 0:
            collated[split_name] = None
            continue
        data, slices = InMemoryDataset.collate(graphs)
        collated[split_name] = {"data": data, "slices": slices}

    dataset_path = os.path.join(datasets_dir, f"{dataset_name}_{dataset_type}.safetensors")
    payload = {
        "task_type": "residual_correction",
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "formation_names": FORMATION_NAMES,
        "splits": collated,
    }
    save_object_safetensors(dataset_path, payload)

    metadata_path = os.path.join(datasets_dir, f"{dataset_name}_{dataset_type}_metadata.json")
    metadata = {
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "task_type": "residual_correction",
        "config": {
            "num_episodes": num_episodes,
            "num_obstacles": num_obstacles,
            "obstacle_radius": obstacle_radius,
            "obstacle_radius_range": list(obstacle_radius_range) if obstacle_radius_range is not None else None,
            "residual_balance_ratio": residual_balance_ratio,
            "residual_dropout_threshold": residual_dropout_threshold,
            "residual_samples_per_seed": residual_samples_per_seed,
            "noisy_sensors": noisy_sensors,
            "noise_variance": noise_variance,
            "communication_radius": communication_radius,
            "include_formation_in_state": include_formation_in_state,
            "mixed_formation_types": list(mixed_formation_types),
            "split_ratios": list(split_ratios),
            "seed": seed,
            "base_xy_limit": base_xy_limit,
            "altitude_range": list(altitude_range),
            "validation_spread_scale": validation_spread_scale,
            "test_spread_scale": test_spread_scale,
        },
        "split_summary": split_summaries,
        "episodes": sorted(episode_records, key=lambda x: x["episode_id"]),
        "single_tensor_output": os.path.basename(dataset_path),
    }
    write_dataset_metadata(metadata_path, metadata)

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f}s")
    print(f"Single dataset tensor: {dataset_path}")
    print(f"Metadata: {metadata_path}")
    return dataset_path, metadata_path


if __name__ == "__main__":
    generate_residual_dataset_singlepass(
        dataset_name="residual_correction_singlepass_apf_40000",
        dataset_type="mixed_formations",
        num_episodes=40000,
        residual_samples_per_seed=50,
        num_obstacles=(0, 10),
        obstacle_radius_range=(0.4, 1.8),
        communication_radius=10.0,
        include_formation_in_state=True,
        noisy_sensors=False,
        seed=12345,
        residual_balance_ratio=0.5,
    )
