"""Run central and decentral simulations on identical initial conditions."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from merged_work.models_creation.dataset_pipeline import (
    EpisodeConfig,
    build_obstacles_for_episode,
)
from resources import ensure_checkpoints
from sim_recorder_central import record_episode_frames as record_central
from sim_recorder_decentral import record_episode_frames_decentral
from website_gnn.assignment_runner import assign_swarm


def run_dual(
    cfg: EpisodeConfig,
    *,
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    slots: np.ndarray,
    assignment_central: np.ndarray,
    max_steps: int | None = None,
    record_every: int | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ensure_checkpoints()

    sp = np.asarray(start_pos, dtype=np.float32).copy()
    so = np.asarray(start_orn, dtype=np.float32).copy()
    sl = np.asarray(slots, dtype=np.float32).copy()
    asn_central = np.asarray(assignment_central, dtype=np.int64).copy()

    obs_cfg_central = build_obstacles_for_episode(cfg, sp, sl, asn_central)

    central = record_central(
        cfg,
        start_pos=sp.copy(),
        start_orn=so.copy(),
        slots=sl.copy(),
        assignment=asn_central.copy(),
        obs_cfg=obs_cfg_central,
        max_steps=max_steps,
        record_every=record_every,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asn_decentral, assign_info = assign_swarm(sp, sl, cfg.digit, device=device)

    decentral = record_episode_frames_decentral(
        cfg,
        start_pos=sp.copy(),
        start_orn=so.copy(),
        slots=sl.copy(),
        assignment=asn_decentral.copy(),
        obs_cfg=obs_cfg_central,
        max_steps=max_steps,
        record_every=record_every,
    )

    meta = {
        "seed": cfg.seed,
        "digit": cfg.digit,
        "scenario": cfg.scenario,
        "num_drones": cfg.num_drones,
        "assignment": asn_central.tolist(),
        "assignment_central": asn_central.tolist(),
        "assignment_decentral": decentral.get("assignment", asn_decentral.tolist()),
        "assignment_repaired": bool(assign_info.get("repaired", False)),
        "assignment_fix_count": int(assign_info.get("num_fixed", 0)),
        "central_steps": central.get("steps"),
        "decentral_steps": decentral.get("steps"),
        "central_converged": central.get("converged"),
        "decentral_converged": decentral.get("converged"),
        "central_stopped_reason": central.get("stopped_reason"),
        "decentral_stopped_reason": decentral.get("stopped_reason"),
        "central_max_steps_cap": central.get("max_steps_cap"),
        "decentral_max_steps_cap": decentral.get("max_steps_cap"),
    }
    return central, decentral, meta
