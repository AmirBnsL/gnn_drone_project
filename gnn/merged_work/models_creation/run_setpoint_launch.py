"""
Regenerate setpoint data, train SetpointGATv2, validate gates.

From repo root:
  python -m merged_work.models_creation.run_setpoint_launch
  python -m merged_work.models_creation.run_setpoint_launch --quick
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

from merged_work.models_creation.dataset_runner import generate_setpoint_dataset
from merged_work.models_creation.finals_pipeline import ensure_deps, preflight
from merged_work.models_creation.package_setup import repo_root, setup_project_paths
from merged_work.models_creation.setpoint_training import TRAIN_CFG, train_setpoint_v3
from merged_work.models_creation.validate_setpoint_imitation import run_validation

setup_project_paths()


def _purge_stale(work_dir: Path) -> None:
    for name in (
        "setpoint_digits_train.pt",
        "setpoint_digits_val.pt",
        "setpoint_digits_test.pt",
    ):
        p = work_dir / name
        if p.exists():
            p.unlink()
            print(f"Removed {p}")
    ckpt = work_dir / "checkpoints"
    for name in ("best_gatv2_digits.pth", "normalization_stats_digits.pt"):
        p = ckpt / name
        if p.exists():
            p.unlink()
            print(f"Removed {p}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=repo_root() / "merged_artifacts")
    parser.add_argument("--quick", action="store_true", help="30 episodes, 25 epochs (smoke)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    episodes = args.episodes or (30 if args.quick else 500)
    epochs = args.epochs or (25 if args.quick else 100)

    try:
        ensure_deps()
    except Exception as exc:
        print(f"ensure_deps warning: {exc}")

    preflight()
    _purge_stale(work_dir)

    print(f"Generating setpoint dataset: {episodes} episodes ...")
    try:
        generate_setpoint_dataset(
            work_dir,
            num_episodes=episodes,
            num_workers=args.workers,
            max_steps=2000,
        )
    except Exception as exc:
        print(f"Dataset generation failed: {exc}")
        sys.exit(1)

    train_paths = {
        "train": work_dir / "setpoint_digits_train.pt",
        "val": work_dir / "setpoint_digits_val.pt",
        "test": work_dir / "setpoint_digits_test.pt",
    }
    if not all(p.exists() for p in train_paths.values()):
        print("Setpoint splits missing after generation.")
        sys.exit(1)

    cfg = dict(TRAIN_CFG)
    cfg["epochs"] = epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training setpoint GNN on {device} for {epochs} epochs ...")
    results = train_setpoint_v3(
        train_paths["train"],
        train_paths["val"],
        train_paths["test"],
        ckpt_dir,
        device,
        cfg=cfg,
        use_data_parallel=torch.cuda.device_count() > 1,
    )
    print("Training results:", results)

    if args.skip_validate:
        return

    print("Running validation gates ...")
    val = run_validation(ckpt_dir, device=device)
    out = work_dir / "results" / "setpoint_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(out, "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2)
    if not val.get("all_ok"):
        print("Validation gates FAILED — see", out)
        sys.exit(1)
    print("Validation passed:", out)


if __name__ == "__main__":
    main()
