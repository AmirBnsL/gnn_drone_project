"""Setpoint GNN inference (41-dim digit frames, dual-head)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Tuple

import numpy as np
import pybullet as p
import torch
from torch_geometric.data import Data

import sys
from pathlib import Path

from path_setup import ensure_repo_path

ensure_repo_path()


def _ensure_setpoint_model_path() -> None:
    """Append setpoint src so backend/website_gnn is not shadowed by inference.py."""
    root = Path(__file__).resolve().parents[4]
    src = root / "gnn" / "setpoint_prediction3" / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.append(s)


from merged_work.models_creation.digit_formations import build_digit_one_hot
from merged_work.models_creation.setpoint_rollout import (
    RAW_FRAME_DIM,
    build_drone_frame,
)
from merged_work.models_creation.setpoint_rollout import _build_edges as build_edges
from merged_work.models_creation.setpoint_training import (
    DatasetNormalizerV10,
    engineer_x_v10,
)

from resources import ensure_checkpoints
from sim_frame_utils import DT
import viz_sim_config as vcfg

ENGINEERED_DIM = 64
EDGE_DIM = 7


@lru_cache(maxsize=1)
def _load_setpoint_bundle() -> Tuple[Any, DatasetNormalizerV10, torch.device]:
    _ensure_setpoint_model_path()
    from model import SetpointGATv2  # noqa: E402

    paths = ensure_checkpoints()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SetpointGATv2(
        in_ch=ENGINEERED_DIM,
        hid_ch=64,
        out_ch=4,
        edge_dim=EDGE_DIM,
        heads=4,
        num_layers=3,
        dropout=0.0,
    )
    model.load_state_dict(
        torch.load(paths["best_gatv2_digits.pth"], map_location=device, weights_only=True)
    )
    model.eval()
    payload = torch.load(paths["normalization_stats_digits.pt"], map_location=device, weights_only=False)
    norm = DatasetNormalizerV10(
        payload["x_mean"],
        payload["x_std"],
        payload["e_mean"],
        payload["e_std"],
        payload["y_scale"],
        payload["cos_sin_indices"],
    ).to(device)
    return model, norm, device


def pred_to_global_setpoints(
    pred_phys: np.ndarray,
    positions: np.ndarray,
    eulers: np.ndarray,
    local_vels: List[np.ndarray],
    targets: np.ndarray,
    step: int,
) -> np.ndarray:
    """Map GNN prediction + pull toward slot setpoints for PyFlyt (hybrid controller)."""
    positions = np.asarray(positions, dtype=np.float32)
    eulers = np.asarray(eulers, dtype=np.float32)
    local_vels = np.asarray(local_vels, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    n = len(positions)
    setpoints = np.zeros((n, 4), dtype=np.float32)

    ramp = min(1.0, float(step) / max(1.0, float(vcfg.SETPOINT_RAMP_STEPS)))
    gain_max = float(vcfg.SETPOINT_GAIN_MAX) * ramp
    yaw_gain = float(vcfg.SETPOINT_YAW_GAIN) * ramp
    pred_gain = float(vcfg.SETPOINT_PRED_GAIN)
    goal_gain = float(vcfg.SETPOINT_GOAL_GAIN)
    d_half = float(vcfg.SETPOINT_D_HALF)

    for i in range(n):
        R = np.array(
            p.getMatrixFromQuaternion(p.getQuaternionFromEuler(eulers[i]))
        ).reshape(3, 3)
        goal_pos = np.array([targets[i, 0], targets[i, 1], targets[i, 3]], dtype=np.float32)
        goal_err = goal_pos - positions[i]
        goal_dist = float(np.linalg.norm(goal_err))

        distance_gain = gain_max * np.tanh(goal_dist / max(d_half, 1e-6))
        carrot_local = (pred_phys[i, :3] - local_vels[i] * DT) * distance_gain
        carrot_global = R @ carrot_local

        blend_alpha = float(np.clip(goal_dist / max(d_half, 1e-6), 0.0, 1.0))
        pred_weight = pred_gain * blend_alpha
        goal_weight = goal_gain + pred_gain * (1.0 - blend_alpha)
        if goal_dist < float(vcfg.SETPOINT_GOAL_BOOST_DIST):
            goal_weight += float(vcfg.SETPOINT_GOAL_BOOST)
        delta_xyz = carrot_global * pred_weight + goal_err * goal_weight

        delta_xy = delta_xyz[:2]
        delta_xy_norm = float(np.linalg.norm(delta_xy))
        if delta_xy_norm > vcfg.SETPOINT_MAX_STEP_XY:
            delta_xy = delta_xy / delta_xy_norm * float(vcfg.SETPOINT_MAX_STEP_XY)
        delta_z = float(np.clip(delta_xyz[2], -vcfg.SETPOINT_MAX_STEP_Z, vcfg.SETPOINT_MAX_STEP_Z))

        goal_yaw = float(targets[i, 2])
        yaw_err = (goal_yaw - float(eulers[i, 2]) + np.pi) % (2 * np.pi) - np.pi
        delta_yaw = float(pred_phys[i, 3]) * yaw_gain * pred_weight + yaw_err * goal_weight
        delta_yaw = float(
            np.clip(delta_yaw, -vcfg.SETPOINT_MAX_STEP_YAW, vcfg.SETPOINT_MAX_STEP_YAW)
        )

        setpoints[i, 0] = positions[i, 0] + float(delta_xy[0])
        setpoints[i, 1] = positions[i, 1] + float(delta_xy[1])
        setpoints[i, 2] = float(eulers[i, 2]) + delta_yaw
        setpoints[i, 3] = max(0.5, positions[i, 2] + delta_z)
    return setpoints


def build_setpoint_graph(
    positions: List[np.ndarray],
    eulers: List[np.ndarray],
    global_vels: List[np.ndarray],
    ang_vels: List[np.ndarray],
    setpoints: np.ndarray,
    formation_one_hot: np.ndarray,
    obstacles_lidar: np.ndarray,
    integral_bufs: List[deque],
    prev_frames: List[np.ndarray],
    comm_radius: float,
    device: torch.device,
) -> Tuple[Data, List[np.ndarray], List[np.ndarray]]:
    n = len(positions)
    states = []
    local_vels_out = []
    for i in range(n):
        if len(integral_bufs[i]) > 0:
            ipe = np.mean(list(integral_bufs[i]), axis=0).astype(np.float32)
        else:
            ipe = np.zeros(3, dtype=np.float32)
        frame, _, lpe = build_drone_frame(
            positions[i],
            eulers[i],
            global_vels[i],
            ang_vels[i],
            setpoints[i],
            formation_one_hot,
            obstacles_lidar,
            ipe,
        )
        states.append(frame)
        integral_bufs[i].append(lpe.copy())
        lv = np.array(
            p.getMatrixFromQuaternion(p.getQuaternionFromEuler(eulers[i]))
        ).reshape(3, 3).T @ global_vels[i]
        local_vels_out.append(lv)

    stacked = []
    for i in range(n):
        stacked.append(np.concatenate([states[i], prev_frames[i]]))

    edges, eattrs = build_edges(
        np.array(positions),
        np.array(global_vels),
        np.array(eulers),
        comm_radius,
    )
    x = torch.as_tensor(np.array(stacked), dtype=torch.float32)
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        ea = torch.as_tensor(np.array(eattrs), dtype=torch.float32)
    else:
        ei = torch.empty((2, 0), dtype=torch.long)
        ea = torch.empty((0, EDGE_DIM), dtype=torch.float32)

    batch_vec = torch.zeros(n, dtype=torch.long, device=device)
    graph = Data(x=x, edge_index=ei, edge_attr=ea, batch=batch_vec)
    return graph.to(device), states, local_vels_out


def forward_setpoint(
    graph: Data,
    curr_frames: List[np.ndarray],
) -> Tuple[np.ndarray, float]:
    model, norm, device = _load_setpoint_bundle()
    x_eng = engineer_x_v10(graph.x)
    x_norm = (x_eng - norm.x_mean) / norm.x_std
    ea_norm = (graph.edge_attr - norm.e_mean) / norm.e_std
    batch_vec = graph.batch if graph.batch is not None else torch.zeros(
        graph.x.size(0), dtype=torch.long, device=device
    )
    with torch.no_grad():
        pred_norm, shift_logit = model(x_norm, graph.edge_index, ea_norm, batch=batch_vec)
    pred_phys = (pred_norm * norm.y_scale).cpu().numpy()
    shift_prob = float(torch.sigmoid(shift_logit.view(-1)[0]).item())
    return pred_phys, shift_prob


