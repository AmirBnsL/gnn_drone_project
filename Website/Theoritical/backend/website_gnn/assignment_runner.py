"""Load Bertsekas assignment GNN and run decentralized assign()."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from path_setup import ensure_repo_path

ensure_repo_path()

from bertsekas_auction_v2 import LearnedBertsekaModel  # noqa: E402
from local_negotiator import LocalNegotiatorGNN  # noqa: E402

from resources import checkpoint_paths, ensure_checkpoints
from assignment_repair import repair_assignment_bijection

import viz_sim_config as vcfg

NUM_FORMATIONS = 10


@lru_cache(maxsize=1)
def _load_assignment_model() -> LearnedBertsekaModel:
    paths = ensure_checkpoints()
    device = torch.device("cpu")
    base = LocalNegotiatorGNN(num_formations=NUM_FORMATIONS)
    base_payload = torch.load(paths["strict_local_negotiator_best_v1.pt"], map_location=device, weights_only=False)
    base.load_state_dict(base_payload["model_state_dict"], strict=False)
    model = LearnedBertsekaModel(base)
    bert_payload = torch.load(paths["bertsekas_best_digits.pt"], map_location=device, weights_only=False)
    model.load_state_dict(bert_payload["model_state_dict"])
    model.eval()
    return model


def assign_swarm(
    start_pos: np.ndarray,
    slots: np.ndarray,
    formation_id: int,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, Dict]:
    if device is None:
        device = torch.device("cpu")
    model = _load_assignment_model().to(device)
    data = Data(
        drone_pos=torch.tensor(start_pos[:, :2], dtype=torch.float32),
        slots=torch.tensor(slots[:, :2], dtype=torch.float32),
        formation_id=torch.tensor(int(formation_id), dtype=torch.long),
    )
    assignment_t, info = model.assign(
        data, device, slot_radius=vcfg.SLOT_VISIBILITY_RADIUS
    )
    assignment = assignment_t.detach().cpu().numpy().astype(np.int64)
    assignment, repair_info = repair_assignment_bijection(assignment, start_pos, slots)
    info = dict(info)
    info.update(repair_info)
    info["assignment_repaired"] = float(repair_info.get("repaired", False))
    info["assignment_fix_count"] = float(repair_info.get("num_fixed", 0))
    return assignment, info
