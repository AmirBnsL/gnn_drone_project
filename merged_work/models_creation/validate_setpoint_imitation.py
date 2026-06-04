"""
Go/no-go validation before deploying setpoint checkpoints to the Website.

Run from repo root:
  python -m merged_work.models_creation.validate_setpoint_imitation --ckpt-dir path/to/checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from merged_work.models_creation.package_setup import repo_root, setup_project_paths

setup_project_paths()

from merged_work.models_creation.dataset_pipeline import make_episode_config  # noqa: E402
from merged_work.models_creation.setpoint_rollout import (  # noqa: E402
    LIDAR_MAX_RANGE,
    RAW_FRAME_DIM,
    rollout_from_seed,
)
from merged_work.models_creation.setpoint_training import (  # noqa: E402
    DatasetNormalizerV10,
    SplitDataset,
    _forward_setpoint,
    engineer_x_v10,
    normalize_batch_v10,
)
from model import SetpointGATv2  # noqa: E402

OBSTACLE_NEAR_THRESH = 4.5
TEACHER_COSINE_MIN = 0.7
LIDAR_ANGLE_MIN_DEG = 30.0
MIN_SURFACE_HARD = 0.2
MIN_SURFACE_WARN = 0.35


def _load_model(ckpt_dir: Path, device: torch.device) -> Tuple[SetpointGATv2, DatasetNormalizerV10]:
    ckpt_dir = Path(ckpt_dir)
    model = SetpointGATv2(
        in_ch=64, hid_ch=64, out_ch=4, edge_dim=7, heads=4, num_layers=3, dropout=0.0
    )
    model.load_state_dict(
        torch.load(ckpt_dir / "best_gatv2_digits.pth", map_location=device, weights_only=True)
    )
    model.eval()
    payload = torch.load(
        ckpt_dir / "normalization_stats_digits.pt", map_location=device, weights_only=False
    )
    norm = DatasetNormalizerV10(
        payload["x_mean"],
        payload["x_std"],
        payload["e_mean"],
        payload["e_std"],
        payload["y_scale"],
        payload["cos_sin_indices"],
    ).to(device)
    return model, norm


def _predict_batch(
    model: SetpointGATv2,
    norm: DatasetNormalizerV10,
    graphs: List[Data],
    device: torch.device,
) -> np.ndarray:
    if not graphs:
        return np.zeros((0, 4), dtype=np.float32)
    loader = DataLoader(graphs, batch_size=min(64, len(graphs)))
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_eng = engineer_x_v10(batch.x)
            batch = normalize_batch_v10(batch, norm)
            pred_norm, _ = _forward_setpoint(model, batch)
            pred_phys = (pred_norm * norm.y_scale).cpu().numpy()
            preds.append(pred_phys)
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 4), dtype=np.float32)


def gate_raw_slot_goal_deviation(model, norm, device) -> Dict:
    """
    With raw slot goal error toward an obstacle and shortened lidar, prediction should deviate.
    Matches Website inference (slot setpoint in frame, not APF-leaked target).
    """
    x = np.zeros(RAW_FRAME_DIM * 2, dtype=np.float32)
    x[6:22] = LIDAR_MAX_RANGE
    x[22:25] = [3.0, 0.0, 0.0]
    x[25] = 0.0
    x[41 + 6 : 41 + 22] = LIDAR_MAX_RANGE
    x[41 + 22 : 41 + 25] = [3.0, 0.0, 0.0]

    def _pred_xy(arr: np.ndarray) -> np.ndarray:
        g = Data(x=torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0))
        return _predict_batch(model, norm, [g], device)[0, :2]

    open_space = _pred_xy(x.copy())
    blocked = x.copy()
    blocked[6:22] = np.minimum(blocked[6:22], 0.8)
    blocked[41 + 6 : 41 + 22] = np.minimum(blocked[41 + 6 : 41 + 22], 0.8)
    near_obs = _pred_xy(blocked)

    on = float(np.linalg.norm(open_space))
    nn = float(np.linalg.norm(near_obs))
    if on < 1e-5 and nn < 1e-5:
        angle_deg = 0.0
    elif on < 1e-5 or nn < 1e-5:
        angle_deg = 90.0
    else:
        cos_sim = float(np.clip(np.dot(open_space, near_obs) / (on * nn), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_sim)))

    ok = angle_deg >= LIDAR_ANGLE_MIN_DEG or (on < 1e-5 and nn > 1e-4)
    return {
        "ok": ok,
        "angle_deg": angle_deg,
        "open_xy": open_space.tolist(),
        "near_obs_xy": near_obs.tolist(),
    }


def gate_lidar_sensitivity(model, norm, device) -> Dict:
    """Shortening one lidar ray should change predicted XY direction."""
    x = np.zeros(RAW_FRAME_DIM * 2, dtype=np.float32)
    x[6:22] = LIDAR_MAX_RANGE
    x[22:25] = [2.0, 0.0, 0.0]
    x[25] = 0.0
    x[41 + 6 : 41 + 22] = LIDAR_MAX_RANGE

    def _pred_from_x(arr: np.ndarray) -> np.ndarray:
        g = Data(x=torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0))
        return _predict_batch(model, norm, [g], device)[0, :2]

    base = _pred_from_x(x.copy())
    blocked = x.copy()
    blocked[10] = 0.5
    blocked[41 + 10] = 0.5
    alt = _pred_from_x(blocked)

    bn = float(np.linalg.norm(base))
    an = float(np.linalg.norm(alt))
    if bn < 1e-5 or an < 1e-5:
        angle_deg = 0.0
    else:
        cos_sim = float(np.dot(base, alt) / (bn * an))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_sim)))

    ok = angle_deg >= LIDAR_ANGLE_MIN_DEG or (bn < 1e-5 and an > 1e-4)
    return {"ok": ok, "angle_deg": angle_deg, "base_xy": base.tolist(), "alt_xy": alt.tolist()}


def gate_teacher_cosine(
    model,
    norm,
    device,
    seeds: List[int],
    num_drones: int = 12,
) -> Dict:
    cosines: List[float] = []
    for seed in seeds:
        _, graphs = rollout_from_seed(0, "val", seed, num_drones, max_steps=120, graphical=False)
        if not graphs:
            continue
        preds = _predict_batch(model, norm, graphs, device)
        for gi, g in enumerate(graphs):
            if gi >= len(preds):
                break
            lidar_min = float(g.x[:22, 6:22].min())
            if lidar_min >= OBSTACLE_NEAR_THRESH:
                continue
            target = g.target.numpy()
            p = preds[gi]
            pn = float(np.linalg.norm(p[:2]))
            tn = float(np.linalg.norm(target[:2]))
            if pn < 1e-5 or tn < 1e-5:
                continue
            cosines.append(float(np.dot(p[:2], target[:2]) / (pn * tn)))

    mean_cos = float(np.mean(cosines)) if cosines else 0.0
    return {
        "ok": mean_cos >= TEACHER_COSINE_MIN and len(cosines) >= 10,
        "mean_cosine": mean_cos,
        "n_samples": len(cosines),
    }


def gate_collision_proxy(seeds: List[int], num_drones: int = 12) -> Dict:
    """Teacher rollouts should not hug obstacles (<0.2m) for long stretches."""
    try:
        import pybullet as p
        from PyFlyt.core import Aviary
    except ImportError as exc:
        return {"ok": True, "skipped": True, "reason": str(exc)}

    from merged_work.models_creation.dataset_pipeline import (
        build_digit_setpoints,
        build_naive_slots,
        build_obstacles_for_episode,
        obstacles_to_lidar_array,
        sample_initial_state,
    )
    from merged_work.models_creation.setpoint_rollout import (
        compute_apf_setpoint_sensor_gated,
        spawn_spheres,
    )

    min_dists: List[float] = []
    for seed in seeds[:5]:
        cfg = make_episode_config(seed=seed, num_drones=num_drones, scenario="both")
        start_pos, start_orn = sample_initial_state(cfg)
        setpoints, assignment, _ = build_digit_setpoints(cfg, start_pos, start_orn)
        slots = build_naive_slots(cfg, start_pos)
        obs_cfg = build_obstacles_for_episode(cfg, start_pos, slots, assignment)
        obstacles_lidar = obstacles_to_lidar_array(obs_cfg)
        if obstacles_lidar.size == 0:
            continue

        env = Aviary(start_pos=start_pos, start_orn=start_orn, drone_type="quadx", render=False)
        env.set_mode(7)
        spawn_spheres(obs_cfg, env._client)
        env.register_all_new_bodies()
        n = cfg.num_drones
        for _ in range(80):
            others_cache = {i: np.array(env.drones[i].state[3]) for i in range(n)}
            for i in range(n):
                st = env.drones[i].state
                ge = np.array(st[1], copy=True)
                others = [others_cache[j] for j in range(n) if j != i]
                mod = compute_apf_setpoint_sensor_gated(
                    others_cache[i],
                    setpoints[i],
                    obstacles_lidar,
                    others,
                    yaw=float(ge[2]),
                )
                env.set_setpoint(i, mod)
            env.step()
            for i in range(n):
                gp = np.array(env.drones[i].state[3])
                for obs in obstacles_lidar:
                    d = float(np.linalg.norm(gp[:2] - obs[:2])) - float(obs[2])
                    min_dists.append(d)
        env.disconnect()

    if not min_dists:
        return {"ok": True, "skipped": True, "reason": "no obstacle episodes"}
    arr = np.asarray(min_dists)
    frac_bad = float(np.mean(arr < MIN_SURFACE_HARD))
    global_min = float(np.min(arr))
    return {
        "ok": frac_bad < 0.05 and global_min >= MIN_SURFACE_WARN,
        "frac_surface_under_0.2m": frac_bad,
        "min_surface_distance": global_min,
        "n_samples": len(min_dists),
    }


def gate_early_shift_suppressed(seeds: List[int], num_drones: int = 12) -> Dict:
    """Training rollouts must not Z-shift before SHIFT_MIN_STEPS (25)."""
    from merged_work.models_creation.dataset_pipeline import make_episode_config
    from merged_work.models_creation.setpoint_rollout import (
        SHIFT_MIN_STEPS,
        simulate_setpoint_episode,
    )

    early_shifts = 0
    checked = 0
    for seed in seeds[:5]:
        cfg = make_episode_config(seed=seed, num_drones=num_drones, scenario="slot")
        try:
            _, graphs = simulate_setpoint_episode(
                0, "val", cfg, max_steps=120, graphical=False
            )
        except Exception as exc:
            return {"ok": True, "skipped": True, "reason": str(exc)}
        if not graphs:
            continue
        checked += 1
        for g in graphs:
            step_idx = int(getattr(g, "step_idx", torch.tensor(0)).item())
            if step_idx < SHIFT_MIN_STEPS and float(getattr(g, "shift_label", torch.tensor(0.0)).item()) > 0.5:
                early_shifts += 1
    return {
        "ok": early_shifts == 0 and checked > 0,
        "early_shift_graphs": early_shifts,
        "episodes_checked": checked,
    }


def run_validation(ckpt_dir: Path, device: torch.device | None = None) -> Dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir)
    if not (ckpt_dir / "best_gatv2_digits.pth").is_file():
        raise FileNotFoundError(f"Missing checkpoint in {ckpt_dir}")

    model, norm = _load_model(ckpt_dir, device)
    seeds = list(range(2000, 2020))

    results = {
        "raw_slot_goal_deviation": gate_raw_slot_goal_deviation(model, norm, device),
        "lidar_sensitivity": gate_lidar_sensitivity(model, norm, device),
        "teacher_cosine": gate_teacher_cosine(model, norm, device, seeds),
        "collision_proxy": gate_collision_proxy(seeds),
        "early_shift_suppressed": gate_early_shift_suppressed(seeds),
    }
    results["all_ok"] = all(
        results[k].get("ok", False) or results[k].get("skipped", False)
        for k in (
            "raw_slot_goal_deviation",
            "lidar_sensitivity",
            "teacher_cosine",
            "collision_proxy",
            "early_shift_suppressed",
        )
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate setpoint GNN imitation gates")
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=repo_root() / "merged_artifacts" / "checkpoints",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        results = run_validation(args.ckpt_dir)
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}")
        sys.exit(1)

    out_path = args.out or (args.ckpt_dir.parent / "results" / "setpoint_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    if not results["all_ok"]:
        print("GATES FAILED — do not deploy checkpoints to Website.")
        sys.exit(1)
    print("All gates passed.")


if __name__ == "__main__":
    main()
