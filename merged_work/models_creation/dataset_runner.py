"""
Dataset generation for assignment (negotiator) and setpoint (V3) tasks.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from merged_work.models_creation.dataset_pipeline import (
    assign_drones_to_slots,
    build_assignment_graph,
    build_naive_slots,
    make_episode_config,
    sample_initial_state,
)
from merged_work.models_creation.setpoint_rollout import rollout_from_seed

SPLIT_NAMES = ("train", "val", "test")
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}


def _split_counts(num_episodes: int, ratios: Tuple[float, float, float]) -> Dict[str, int]:
    counts = [int(num_episodes * r) for r in ratios]
    rem = num_episodes - sum(counts)
    for i in range(rem):
        counts[i % 3] += 1
    return dict(zip(SPLIT_NAMES, counts))


def generate_assignment_dataset_simple(
    dataset_root: Path,
    num_episodes: int = 3000,
    num_drones_range: Tuple[int, int] = (10, 20),
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Path:
    """Generate assignment graphs and a combined negotiator .pt file."""
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    negotiator_path = dataset_root / "negotiator_dataset_digits.pt"
    if negotiator_path.exists():
        print(f"[skip] {negotiator_path}")
        return negotiator_path

    counts = _split_counts(num_episodes, split_ratios)
    rng = np.random.default_rng(seed)
    all_graphs: List[Data] = []
    split_graphs: Dict[str, List[Data]] = {s: [] for s in SPLIT_NAMES}

    for split in SPLIT_NAMES:
        for local in range(counts[split]):
            ep_seed = seed + SPLIT_SEED_OFFSETS[split] + local
            n = int(rng.integers(num_drones_range[0], num_drones_range[1] + 1))
            cfg = make_episode_config(seed=ep_seed, num_drones=n)
            start_pos, _ = sample_initial_state(cfg)
            slots = build_naive_slots(cfg, start_pos)
            assignment = assign_drones_to_slots(start_pos, slots)
            g = build_assignment_graph(cfg, start_pos, slots, assignment)
            all_graphs.append(g)
            split_graphs[split].append(g)

        p = dataset_root / f"assignment_digits_{split}.pt"
        if split_graphs[split]:
            data, slices = InMemoryDataset.collate(split_graphs[split])
            torch.save(
                {
                    "data": data,
                    "slices": slices,
                    "split_name": split,
                    "num_graphs": len(split_graphs[split]),
                },
                p,
            )
            print(f"Saved {split}: {len(split_graphs[split])} → {p}")

    torch.save(all_graphs, negotiator_path)
    print(f"Saved negotiator bundle: {len(all_graphs)} → {negotiator_path}")
    meta = {
        "num_episodes": num_episodes,
        "num_formations": 10,
        "splits": {s: len(split_graphs[s]) for s in SPLIT_NAMES},
    }
    with open(dataset_root / "assignment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return negotiator_path


def _setpoint_worker(task: dict) -> Tuple[str, List[Data]]:
    try:
        return rollout_from_seed(
            task["ep_idx"],
            task["split"],
            task["seed"],
            task["num_drones"],
            max_steps=task.get("max_steps", 400),
            save_interval=task.get("save_interval", 5),
            graphical=False,
        )
    except Exception as exc:
        print(
            f"[setpoint FAILED] ep={task['ep_idx']} seed={task['seed']}: {type(exc).__name__}: {exc}"
        )
        return task["split"], []


def generate_setpoint_dataset(
    dataset_root: Path,
    num_episodes: int = 200,
    num_drones_range: Tuple[int, int] = (10, 20),
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    num_workers: int = 1,
    seed: int = 12345,
    max_steps: int = 400,
    save_interval: int = 5,
) -> Dict[str, Path]:
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    counts = _split_counts(num_episodes, split_ratios)
    rng = np.random.default_rng(seed)
    paths: Dict[str, Path] = {s: dataset_root / f"setpoint_digits_{s}.pt" for s in SPLIT_NAMES}

    if all(p.exists() for p in paths.values()):
        print("[skip] all setpoint splits exist")
        return paths

    tasks = []
    global_idx = 0
    for split in SPLIT_NAMES:
        if paths[split].exists():
            continue
        for local in range(counts[split]):
            ep_seed = seed + SPLIT_SEED_OFFSETS[split] + local
            n = int(rng.integers(num_drones_range[0], num_drones_range[1] + 1))
            tasks.append(
                {
                    "ep_idx": global_idx,
                    "split": split,
                    "seed": ep_seed,
                    "num_drones": n,
                    "max_steps": max_steps,
                    "save_interval": save_interval,
                }
            )
            global_idx += 1

    split_graphs: Dict[str, List[Data]] = {s: [] for s in SPLIT_NAMES}
    worker_fn = partial(_setpoint_worker)
    ok_count = 0
    if num_workers <= 1:
        for t in tasks:
            sp, gs = worker_fn(t)
            split_graphs[sp].extend(gs)
            if gs:
                ok_count += 1
                print(f"Episode {t['ep_idx']} ({sp}): OK — {len(gs)} frames")
            else:
                print(f"Episode {t['ep_idx']} ({sp}): FAILED (0 frames)")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futs = {ex.submit(worker_fn, t): t for t in tasks}
            for fut in as_completed(futs):
                t = futs[fut]
                sp, gs = fut.result()
                split_graphs[sp].extend(gs)
                if gs:
                    ok_count += 1
                    print(f"Episode {t['ep_idx']} ({sp}): OK — {len(gs)} frames")
                else:
                    print(f"Episode {t['ep_idx']} ({sp}): FAILED (0 frames)")

    print(f"Setpoint rollouts: {ok_count} / {len(tasks)} episodes produced data")

    for split in SPLIT_NAMES:
        out_path = paths[split]
        if out_path.exists():
            continue
        graphs = split_graphs[split]
        if not graphs:
            continue
        data, slices = InMemoryDataset.collate(graphs)
        torch.save(
            {"data": data, "slices": slices, "split_name": split, "num_graphs": len(graphs)},
            out_path,
        )
        print(f"Saved setpoint {split}: {len(graphs)} graphs → {out_path}")

    with open(dataset_root / "setpoint_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {"num_episodes": num_episodes, "raw_frame_dim": 41, "splits": counts},
            f,
            indent=2,
        )
    return paths
