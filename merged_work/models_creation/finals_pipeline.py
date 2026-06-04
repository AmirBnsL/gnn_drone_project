"""
End-to-end Kaggle pipeline: datasets → LocalNeg → Bertsekas → Setpoint → zip.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional

import torch

from merged_work.models_creation.assignment_training import (
    pretrain_local_negotiator,
    train_bertsekas_assignment,
)
from merged_work.models_creation.dataset_runner import (
    generate_assignment_dataset_simple,
    generate_setpoint_dataset,
)
from merged_work.models_creation.package_setup import repo_root, setup_project_paths
from merged_work.models_creation.setpoint_training import TRAIN_CFG, train_setpoint_v3

setup_project_paths()


def preflight() -> None:
    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} {p.total_memory / 1024**3:.1f} GiB")
    try:
        import torch_geometric  # noqa: F401

        print("PyG:", torch_geometric.__version__)
    except ImportError as e:
        raise RuntimeError("torch_geometric not installed") from e


def ensure_deps() -> None:
    """Idempotent pip installs for Kaggle."""
    pkgs = [
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        "pyflyt",
        "pybullet",
        "safetensors",
        "tqdm",
        "scipy",
    ]
    for pkg in pkgs:
        try:
            __import__(pkg.replace("-", "_").split("[")[0])
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
            )


def run_pipeline(
    work_dir: Path,
    *,
    assignment_episodes: int = 3000,
    setpoint_episodes: int = 500,
    setpoint_max_steps: int = 2000,
    setpoint_workers: int = 1,
    localneg_epochs: int = 60,
    bertsekas_epochs: int = 80,
    setpoint_epochs: int = 100,
    setpoint_cfg: Optional[Dict] = None,
    skip_setpoint_data: bool = False,
    has_pyflyt: bool = True,
) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    preflight()

    negotiator_path = generate_assignment_dataset_simple(
        work_dir,
        num_episodes=assignment_episodes,
    )

    if skip_setpoint_data or not has_pyflyt:
        if not has_pyflyt:
            print(
                "WARNING: PyFlyt unavailable — skipping setpoint dataset generation "
                "and setpoint GNN training. Assignment models will still train."
            )
    else:
        generate_setpoint_dataset(
            work_dir,
            num_episodes=setpoint_episodes,
            num_workers=setpoint_workers,
            max_steps=setpoint_max_steps,
        )

    base_ckpt = ckpt_dir / "strict_local_negotiator_best_v1.pt"
    pretrain_local_negotiator(
        negotiator_path,
        base_ckpt,
        device,
        epochs=localneg_epochs,
        max_train=min(5000, assignment_episodes),
        max_val=500,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bert_ckpt = ckpt_dir / "bertsekas_best_digits.pt"
    train_bertsekas_assignment(
        negotiator_path,
        base_ckpt,
        bert_ckpt,
        device,
        epochs=bertsekas_epochs,
        max_train=min(5000, assignment_episodes),
        max_val=500,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_paths = {
        "train": work_dir / "setpoint_digits_train.pt",
        "val": work_dir / "setpoint_digits_val.pt",
        "test": work_dir / "setpoint_digits_test.pt",
    }
    if all(p.exists() for p in train_paths.values()):
        _sp_cfg = dict(TRAIN_CFG)
        if setpoint_cfg:
            _sp_cfg.update(setpoint_cfg)
        train_setpoint_v3(
            train_paths["train"],
            train_paths["val"],
            train_paths["test"],
            ckpt_dir,
            device,
            cfg=_sp_cfg,
            use_data_parallel=torch.cuda.device_count() > 1,
        )
    else:
        print(
            "WARNING: Setpoint splits missing — skipping setpoint GNN training. "
            "Check PyFlyt install or partial setpoint rollouts above."
        )

    zip_path = work_dir / "swarm_artifact.zip"
    _build_zip(work_dir, zip_path, repo_root())

    elapsed = time.time() - t0
    meta = {
        "elapsed_sec": elapsed,
        "assignment_episodes": assignment_episodes,
        "setpoint_episodes": setpoint_episodes,
        "setpoint_max_steps": setpoint_max_steps,
        "device": str(device),
        "cuda_devices": torch.cuda.device_count(),
    }
    with open(work_dir / "pipeline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Done in {elapsed / 60:.1f} min. Artifact: {zip_path}")
    return zip_path


def _build_zip(work_dir: Path, zip_path: Path, root: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    include_suffixes = {".pt", ".json", ".pth"}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in work_dir.rglob("*"):
            if p.is_file() and p.suffix in include_suffixes:
                zf.write(p, p.relative_to(work_dir))
        for rel in (
            "merged_work/models_creation/digit_formations.py",
            "merged_work/models_creation/dataset_pipeline.py",
            "merged_work/models_creation/setpoint_rollout.py",
        ):
            src = root / rel
            if src.exists():
                zf.write(src, rel)


def smoke_test(work_dir: Path) -> None:
    from merged_work.models_creation.digit_formations import sample_digit_offsets
    from merged_work.models_creation.dataset_pipeline import make_episode_config, sample_initial_state, build_naive_slots

    cfg = make_episode_config(seed=0, num_drones=12, digit=3, scenario="both")
    pos, _ = sample_initial_state(cfg)
    slots = build_naive_slots(cfg, pos)
    off = sample_digit_offsets(3, 12)
    assert off.shape == (12, 3)
    assert slots.shape == (12, 3)
    print("Smoke test OK:", cfg.digit, slots[:, 2].std())
