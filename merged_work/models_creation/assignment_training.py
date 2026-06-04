"""
LocalNegotiator pretrain + Learned Bertsekas training (10 digit formations).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from merged_work.models_creation.package_setup import setup_project_paths

setup_project_paths()

from local_negotiator import (  # noqa: E402
    LocalNegotiatorGNN,
    evaluate_strict_decentralized,
    load_negotiator_dataset,
    prepare_dataset,
)
from bertsekas_auction_v2 import (  # noqa: E402
    LearnedBertsekaModel,
    evaluate_bertsekas,
    train_bertsekas_model,
    train_strict_decentralized_model,
)

NUM_DIGIT_FORMATIONS = 10


def load_and_prepare_negotiator(
    dataset_path: Path,
    seed: int = 42,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Tuple[LocalNegotiatorGNN, List, List, List]:
    model = LocalNegotiatorGNN(num_formations=NUM_DIGIT_FORMATIONS)
    raw = load_negotiator_dataset(str(dataset_path))
    train_data, val_data, test_data = prepare_dataset(
        raw,
        model.formation_embedding.weight.detach().cpu(),
        seed=seed,
        force_gt_visibility=False,
    )
    if max_train is not None:
        train_data = train_data[:max_train]
    if max_val is not None:
        val_data = val_data[:max_val]
    if max_test is not None:
        test_data = test_data[:max_test]
    return model, train_data, val_data, test_data


def pretrain_local_negotiator(
    dataset_path: Path,
    ckpt_path: Path,
    device: torch.device,
    epochs: int = 60,
    lr: float = 1e-3,
    max_train: int = 5000,
    max_val: int = 500,
) -> Dict:
    if ckpt_path.exists():
        print(f"[skip] LocalNegotiator checkpoint exists: {ckpt_path}")
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model, train_data, val_data, _ = load_and_prepare_negotiator(
        dataset_path, max_train=max_train, max_val=max_val, max_test=100
    )
    model.to(device)
    history = train_strict_decentralized_model(
        model,
        train_data,
        val_data,
        device,
        epochs=epochs,
        lr=lr,
        ckpt_path=str(ckpt_path),
    )
    metrics = evaluate_strict_decentralized(model, val_data[:200], device)
    payload = {
        "model_state_dict": model.state_dict(),
        "metrics": {"val": metrics},
        "history": history,
        "num_formations": NUM_DIGIT_FORMATIONS,
    }
    torch.save(payload, ckpt_path)
    print(f"Saved LocalNegotiator → {ckpt_path}")
    return payload


def train_bertsekas_assignment(
    dataset_path: Path,
    base_ckpt: Path,
    out_ckpt: Path,
    device: torch.device,
    epochs: int = 80,
    lr: float = 3e-4,
    freeze_epochs: int = 10,
    max_train: int = 5000,
    max_val: int = 500,
) -> Dict:
    if out_ckpt.exists():
        print(f"[skip] Bertsekas checkpoint exists: {out_ckpt}")
        return torch.load(out_ckpt, map_location="cpu", weights_only=False)

    base = LocalNegotiatorGNN(num_formations=NUM_DIGIT_FORMATIONS).to(device)
    if base_ckpt.exists():
        ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        base.load_state_dict(state, strict=False)
        print(f"Loaded base from {base_ckpt}")
    else:
        print("Warning: no base checkpoint — Bertsekas trains from scratch encoder")

    model = LearnedBertsekaModel(base).to(device)
    _, train_data, val_data, test_data = load_and_prepare_negotiator(
        dataset_path, max_train=max_train, max_val=max_val, max_test=200
    )
    history = train_bertsekas_model(
        model,
        train_data,
        val_data,
        device,
        epochs=epochs,
        lr=lr,
        freeze_epochs=freeze_epochs,
        force_gt_train=True,
    )
    final = evaluate_bertsekas(model, test_data, device)
    payload = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "metrics": {"bertsekas": final},
        "num_formations": NUM_DIGIT_FORMATIONS,
    }
    torch.save(payload, out_ckpt)
    print(f"Saved Bertsekas → {out_ckpt}")
    return payload
