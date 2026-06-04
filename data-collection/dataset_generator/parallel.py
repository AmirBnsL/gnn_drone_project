import os
import time
import shutil
import gc
from typing import Literal
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import torch

from dataset_generator.constants import FORMATION_NAMES, SPLIT_NAMES, TASK_TYPES
from dataset_generator.utils import compute_split_episode_counts, resolve_split_spread_scale, save_dataset_shard, write_dataset_metadata
from dataset_generator.formations import resolve_formation_name
from dataset_generator.simulation import generate_residual_correction_sample, run_physics_episode

def build_episode_seed(seed, split_name, split_episode_idx):
    from dataset_generator.constants import SPLIT_SEED_OFFSETS
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
                episode_configs.append({
                    "split_episode_idx": split_episode_idx,
                    "episode_seed": ep_seed,
                    "global_episode_id": next_global_episode_id,
                })
                next_global_episode_id += 1

            worker_tasks.append({
                "worker_id": worker_id,
                "split_name": split_name,
                "spread_scale": spread_scale,
                "episode_configs": episode_configs,
                "num_episodes": num_episodes_for_worker,
                **common_kwargs,
            })
            worker_id += 1
            start_episode_idx += num_episodes_for_worker

    return worker_tasks

def simulate_worker_chunk(config: dict):
    split_name = config["split_name"]
    episode_configs = config["episode_configs"]
    task_type = config["task_type"]
    chunk_graphs = []
    episode_records = []
    total_count_near_zero = 0
    total_count_significant = 0
    
    # We will flush the chunk_graphs to disk periodically to prevent OOM
    MAX_GRAPHS_PER_SHARD = 2000
    shard_files = []
    shard_counter = 0

    def flush_shard():
        nonlocal chunk_graphs, shard_counter
        if not chunk_graphs:
            return
        shard_path = os.path.join(config["temp_dir"], f"shard_{config['worker_id']}_{shard_counter}_split_{split_name}.safetensors")
        save_dataset_shard(shard_path, chunk_graphs, FORMATION_NAMES, split_name)
        shard_files.append(shard_path)
        del chunk_graphs[:]
        gc.collect()
        shard_counter += 1

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
            ep_graphs, ep_record = run_physics_episode(ep_config, single_config)
            chunk_graphs.extend(ep_graphs)
            episode_records.append(ep_record)
            
        # Flush if getting too large
        if len(chunk_graphs) >= MAX_GRAPHS_PER_SHARD:
            flush_shard()

    flush_shard() # flush remaining

    return split_name, shard_files, total_count_near_zero, total_count_significant, episode_records

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
    repo_root = os.path.dirname(os.path.dirname(script_dir)) # Since it's now in gnn_drone_project/data-collection/dataset_generator
    datasets_dir = os.path.join(repo_root, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {TASK_TYPES}")

    dataset_prefix = os.path.join(datasets_dir, f"{dataset_name}_{dataset_type}")
    split_episode_counts = compute_split_episode_counts(num_episodes, split_ratios)
    
    temp_dir = os.path.join(datasets_dir, f"temp_{dataset_name}_{dataset_type}_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
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

    split_summaries = {
        s: {"num_episodes": split_episode_counts[s], "num_graphs": 0, "spread_scale": resolve_split_spread_scale(s, validation_spread_scale, test_spread_scale), "count_near_zero": 0, "count_significant": 0} 
        for s in SPLIT_NAMES
    }
    episode_records = []

    print(f"Launching {len(worker_tasks)} worker tasks across {num_workers} parallel processes...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(simulate_worker_chunk, task): task for task in worker_tasks}
        with tqdm(total=len(worker_tasks), desc="Worker Tasks", unit="task", dynamic_ncols=True) as progress:
            for future in futures:
                # Need as_completed actually if we want to iterate over them safely, but this works to just await
                pass

    print(f"Parallel Simulation finished in {time.time() - start_time:.2f}s. Aggregating datasets...")
    
    generated_files = {}
    for split_name in SPLIT_NAMES:
        shard_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith(f"shard_") and f"split_{split_name}" in f]
        if not shard_files:
            continue
        generated_files[split_name] = [os.path.basename(f) for f in shard_files]
        print(f"Generated {split_name} dataset shards: {generated_files[split_name]}")
    
    # We will leave the temp dir so the files exist. If we wanted, we could move them. Let's move them to datasets_dir.
    for split_name, files in generated_files.items():
        for f in files:
            shutil.move(os.path.join(temp_dir, f), os.path.join(datasets_dir, f))
            
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
        "episodes": sorted(episode_records, key=lambda x: x["episode_id"]) if len(episode_records) > 0 else [],
    }
    write_dataset_metadata(metadata_path, metadata)
    print(f"Generated dataset metadata -> {metadata_path}")

    return generated_files, metadata_path
