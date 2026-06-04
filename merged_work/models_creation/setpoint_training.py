"""
V3 SetpointGATv2 training with 41-dim raw frames (10-digit one-hot).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from merged_work.models_creation.package_setup import setup_project_paths

setup_project_paths()

from model import SetpointGATv2  # noqa: E402

RAW_FRAME_DIM = 41
YAW_IDX = 25
FORMATION_ONE_HOT_DIM = 10
ENGINEERED_DIM = 64


def process_frame_v10(frame: torch.Tensor, yaw_idx: int, frame_dim: int) -> torch.Tensor:
    """41-dim frame → 32 engineered dims (drop 10-dim digit one-hot)."""
    before = frame[:, :yaw_idx]
    yaw = frame[:, yaw_idx]
    after = frame[:, yaw_idx + 1 : frame_dim - FORMATION_ONE_HOT_DIM]
    return torch.cat(
        [before, torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1), after],
        dim=1,
    )


def engineer_x_v10(x: torch.Tensor, raw_frame_dim: int = RAW_FRAME_DIM, yaw_idx: int = YAW_IDX) -> torch.Tensor:
    return torch.cat(
        [
            process_frame_v10(x[:, :raw_frame_dim], yaw_idx, raw_frame_dim),
            process_frame_v10(x[:, raw_frame_dim:], yaw_idx, raw_frame_dim),
        ],
        dim=1,
    )


class DatasetNormalizerV10:
    def __init__(self, x_mean, x_std, e_mean, e_std, y_scale, cos_sin_indices):
        self.x_mean = x_mean
        self.x_std = x_std
        self.e_mean = e_mean
        self.e_std = e_std
        self.y_scale = y_scale
        self.cos_sin_indices = cos_sin_indices
        self.raw_frame_dim = RAW_FRAME_DIM
        self.yaw_idx = YAW_IDX

    @classmethod
    def fit(cls, train_ds, yaw_quantile: float = 0.99):
        all_x = engineer_x_v10(train_ds.data.x)
        x_mean, x_std = all_x.mean(0), all_x.std(0).clamp(min=1e-6)
        cos_sin = [25, 26, 57, 58]
        for i in cos_sin:
            x_mean[i], x_std[i] = 0.0, 1.0
        all_edge = train_ds.data.edge_attr
        e_mean, e_std = all_edge.mean(0), all_edge.std(0).clamp(min=1e-6)
        all_y = train_ds.data.target
        y_scale = all_y.abs().max(0).values.clamp(min=1e-6)
        y_scale[3] = torch.quantile(all_y[:, 3].abs(), yaw_quantile)
        return cls(x_mean, x_std, e_mean, e_std, y_scale, cos_sin)

    def to(self, device):
        self.x_mean = self.x_mean.to(device)
        self.x_std = self.x_std.to(device)
        self.e_mean = self.e_mean.to(device)
        self.e_std = self.e_std.to(device)
        self.y_scale = self.y_scale.to(device)
        return self

    def save(self, path):
        torch.save(
            {
                "x_mean": self.x_mean.cpu(),
                "x_std": self.x_std.cpu(),
                "e_mean": self.e_mean.cpu(),
                "e_std": self.e_std.cpu(),
                "y_scale": self.y_scale.cpu(),
                "cos_sin_indices": self.cos_sin_indices,
                "raw_frame_dim": RAW_FRAME_DIM,
                "yaw_idx": YAW_IDX,
            },
            path,
        )


def normalize_batch_v10(batch, norm: DatasetNormalizerV10):
    batch.x = (engineer_x_v10(batch.x) - norm.x_mean) / norm.x_std
    batch.edge_attr = (batch.edge_attr - norm.e_mean) / norm.e_std
    batch.target = torch.clamp(batch.target / norm.y_scale, -1.0, 1.0)
    return batch


class SplitDataset(InMemoryDataset):
    def __init__(self, path: Path):
        payload = torch.load(path.resolve(), weights_only=False)
        super().__init__(root="")
        self.data, self.slices = payload["data"], payload["slices"]


LIDAR_SLICE = slice(6, 22)
OBSTACLE_NEAR_THRESH = 4.5
OBSTACLE_MSE_WEIGHT = 3.0

TRAIN_CFG: Dict = {
    "in_channels": ENGINEERED_DIM,
    "hidden_channels": 64,
    "out_channels": 4,
    "edge_dim": 7,
    "heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 100,
    "patience": 15,
    "stillness_weight": 0.2,
    "direction_weight": 0.0,
    "stillness_decay": 2.0,
    "direction_min_dist": 0.5,
    "yaw_quantile": 0.99,
    "shift_weight": 0.0,
    "shift_pos_weight": 50.0,
    "obstacle_near_thresh": OBSTACLE_NEAR_THRESH,
    "obstacle_mse_weight": OBSTACLE_MSE_WEIGHT,
}


def _forward_setpoint(model, batch):
    batch_vec = getattr(batch, "batch", None)
    return model(batch.x, batch.edge_index, batch.edge_attr, batch=batch_vec)


def compute_loss(
    pred,
    target,
    x_eng,
    cfg,
    shift_logit=None,
    shift_label=None,
    raw_x: Optional[torch.Tensor] = None,
):
    if raw_x is not None:
        lidar_min = raw_x[:, LIDAR_SLICE].min(dim=1).values
        near = lidar_min < float(cfg.get("obstacle_near_thresh", OBSTACLE_NEAR_THRESH))
        w = torch.where(
            near,
            torch.tensor(cfg.get("obstacle_mse_weight", OBSTACLE_MSE_WEIGHT), device=pred.device),
            torch.tensor(1.0, device=pred.device),
        ).view(-1, 1)
        mse = (w * (pred - target) ** 2).mean()
    else:
        mse = F.mse_loss(pred, target)

    local_pos_err = x_eng[:, 22:25]
    dist = torch.norm(local_pos_err, dim=1)
    stillness = (pred[:, :3].norm(dim=1) * torch.exp(-dist * cfg["stillness_decay"])).mean()

    direction = torch.tensor(0.0, device=pred.device)
    if float(cfg.get("direction_weight", 0.0)) > 0.0:
        far = dist > cfg["direction_min_dist"]
        if far.any():
            pn = pred[far, :2].norm(dim=1, keepdim=True).clamp(min=1e-4)
            gn = local_pos_err[far, :2].norm(dim=1, keepdim=True).clamp(min=1e-4)
            direction = 1.0 - F.cosine_similarity(
                pred[far, :2] / pn, local_pos_err[far, :2] / gn, dim=1
            ).mean()

    shift_loss = torch.tensor(0.0, device=pred.device)
    if (
        shift_logit is not None
        and shift_label is not None
        and cfg.get("shift_weight", 0.0) > 0
    ):
        shift_target = shift_label.view(-1).to(shift_logit.dtype)
        pos_weight = torch.tensor(
            [cfg["shift_pos_weight"]], device=shift_logit.device, dtype=shift_logit.dtype
        )
        shift_loss = F.binary_cross_entropy_with_logits(
            shift_logit.view(-1),
            shift_target,
            pos_weight=pos_weight,
        )

    total = (
        mse
        + cfg["stillness_weight"] * stillness
        + cfg["direction_weight"] * direction
        + cfg.get("shift_weight", 0.0) * shift_loss
    )
    return total, mse.item(), stillness.item(), direction.item(), shift_loss.item()


def train_setpoint_v3(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    ckpt_dir: Path,
    device: torch.device,
    cfg: Optional[Dict] = None,
    use_data_parallel: bool = True,
) -> Dict:
    cfg = dict(TRAIN_CFG if cfg is None else cfg)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_gatv2_digits.pth"
    norm_path = ckpt_dir / "normalization_stats_digits.pt"
    metrics_path = ckpt_dir.parent / "results" / "setpoint_eval_metrics.json"

    train_ds = SplitDataset(train_path)
    val_ds = SplitDataset(val_path)
    test_ds = SplitDataset(test_path)
    normalizer = DatasetNormalizerV10.fit(train_ds, cfg["yaw_quantile"]).to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    model = SetpointGATv2(
        in_ch=cfg["in_channels"],
        hid_ch=cfg["hidden_channels"],
        out_ch=cfg["out_channels"],
        edge_dim=cfg["edge_dim"],
        heads=cfg["heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    if use_data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_val, patience_ctr = float("inf"), 0

    def _core(m):
        return m.module if isinstance(m, nn.DataParallel) else m

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t_loss, t_shift, n = 0.0, 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            raw_x = batch.x.clone()
            x_eng = engineer_x_v10(batch.x)
            batch = normalize_batch_v10(batch, normalizer)
            pred, shift_logit = _forward_setpoint(model, batch)
            shift_label = getattr(batch, "shift_label", None)
            loss, _, _, _, sl = compute_loss(
                pred, batch.target, x_eng, cfg, shift_logit, shift_label, raw_x=raw_x
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = batch.num_graphs
            t_loss += loss.item() * bs
            t_shift += sl * bs
            n += bs
        t_loss /= max(n, 1)
        t_shift /= max(n, 1)

        model.eval()
        v_loss, v_shift, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                raw_x = batch.x.clone()
                x_eng = engineer_x_v10(batch.x)
                batch = normalize_batch_v10(batch, normalizer)
                pred, shift_logit = _forward_setpoint(model, batch)
                shift_label = getattr(batch, "shift_label", None)
                loss, _, _, _, sl = compute_loss(
                    pred, batch.target, x_eng, cfg, shift_logit, shift_label, raw_x=raw_x
                )
                bs = batch.num_graphs
                v_loss += loss.item() * bs
                v_shift += sl * bs
                vn += bs
        v_loss /= max(vn, 1)
        v_shift /= max(vn, 1)
        scheduler.step(v_loss)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Setpoint ep {epoch}: train={t_loss:.5f} val={v_loss:.5f} "
                f"shift_train={t_shift:.5f} shift_val={v_shift:.5f}"
            )

        if v_loss < best_val:
            best_val = v_loss
            patience_ctr = 0
            torch.save(_core(model).state_dict(), best_path)
            normalizer.save(norm_path)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(f"Early stop setpoint @ {epoch}")
                break

    _core(model).load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    test_loss, tn = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            raw_x = batch.x.clone()
            x_eng = engineer_x_v10(batch.x)
            batch = normalize_batch_v10(batch, normalizer)
            pred, shift_logit = _forward_setpoint(model, batch)
            shift_label = getattr(batch, "shift_label", None)
            loss, _, _, _, _ = compute_loss(
                pred, batch.target, x_eng, cfg, shift_logit, shift_label, raw_x=raw_x
            )
            bs = batch.num_graphs
            test_loss += loss.item() * bs
            tn += bs
    test_loss /= max(tn, 1)
    results = {"best_val": best_val, "test_loss": test_loss, "config": cfg}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Setpoint test loss: {test_loss:.5f}")
    return results
