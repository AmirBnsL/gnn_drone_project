"""
Feature engineering + normalization pipeline for the V5 drone swarm dataset.

Usage:
    from dataloader import load_splits, DatasetNormalizer, normalize_batch

    train_ds, val_ds, test_ds = load_splits(train_path, val_path, test_path)
    normalizer = DatasetNormalizer.fit(train_ds, CONFIG)
    normalizer.save(path)

    # In training loop:
    batch = normalize_batch(batch, normalizer)
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader


# ═══════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════

class SplitDataset(InMemoryDataset):
    def __init__(self, path):
        payload = torch.load(Path(path).resolve(), weights_only=False)
        super().__init__(root="")
        self.data, self.slices = payload["data"], payload["slices"]


def load_splits(train_path, val_path, test_path):
    return SplitDataset(train_path), SplitDataset(val_path), SplitDataset(test_path)


# ═══════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════════════════

def process_frame(frame, yaw_idx):
    """Drop formation one-hot, replace yaw with cos/sin. (N,32) → (N,29)"""
    before = frame[:, :yaw_idx]
    yaw = frame[:, yaw_idx]
    after = frame[:, yaw_idx + 1:28]
    return torch.cat([before, torch.cos(yaw).unsqueeze(1),
                      torch.sin(yaw).unsqueeze(1), after], dim=1)


def engineer_x(x, raw_frame_dim=32, yaw_idx=25):
    """Process both stacked frames. (N,64) → (N,58)"""
    return torch.cat([process_frame(x[:, :raw_frame_dim], yaw_idx),
                      process_frame(x[:, raw_frame_dim:], yaw_idx)], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# Normalizer
# ═══════════════════════════════════════════════════════════════════════════

class DatasetNormalizer:
    """
    Computes and applies:
    - x: Z-score (excluding cos/sin features)
    - edge_attr: Z-score
    - target: max-abs scaling to [-1, 1] (99th percentile for yaw)
    """

    def __init__(self, x_mean, x_std, e_mean, e_std, y_scale,
                 cos_sin_indices, raw_frame_dim, yaw_idx):
        self.x_mean = x_mean
        self.x_std = x_std
        self.e_mean = e_mean
        self.e_std = e_std
        self.y_scale = y_scale
        self.cos_sin_indices = cos_sin_indices
        self.raw_frame_dim = raw_frame_dim
        self.yaw_idx = yaw_idx

    @classmethod
    def fit(cls, train_ds, config):
        """Compute normalization stats from the training split only."""
        raw_dim = config["raw_frame_dim"]
        yaw_idx = config["yaw_idx_in_frame"]
        cos_sin = config["cos_sin_indices"]

        all_x = engineer_x(train_ds.data.x, raw_dim, yaw_idx)
        x_mean, x_std = all_x.mean(0), all_x.std(0).clamp(min=1e-6)
        for i in cos_sin:
            x_mean[i], x_std[i] = 0.0, 1.0

        all_edge = train_ds.data.edge_attr
        e_mean, e_std = all_edge.mean(0), all_edge.std(0).clamp(min=1e-6)

        all_y = train_ds.data.target
        y_scale = all_y.abs().max(0).values.clamp(min=1e-6)
        y_scale[3] = torch.quantile(all_y[:, 3].abs(), config["yaw_quantile"])

        return cls(x_mean, x_std, e_mean, e_std, y_scale, cos_sin, raw_dim, yaw_idx)

    def to(self, device):
        """Move all tensors to device."""
        self.x_mean = self.x_mean.to(device)
        self.x_std = self.x_std.to(device)
        self.e_mean = self.e_mean.to(device)
        self.e_std = self.e_std.to(device)
        self.y_scale = self.y_scale.to(device)
        return self

    def save(self, path):
        torch.save({
            "x_mean": self.x_mean.cpu(), "x_std": self.x_std.cpu(),
            "e_mean": self.e_mean.cpu(), "e_std": self.e_std.cpu(),
            "y_scale": self.y_scale.cpu(),
            "cos_sin_indices": self.cos_sin_indices,
            "raw_frame_dim": self.raw_frame_dim,
            "yaw_idx": self.yaw_idx,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        d = torch.load(path, weights_only=False, map_location=device)
        return cls(d["x_mean"], d["x_std"], d["e_mean"], d["e_std"],
                   d["y_scale"], d["cos_sin_indices"],
                   d["raw_frame_dim"], d["yaw_idx"])


def normalize_batch(batch, norm):
    """Apply feature engineering + normalization to a batch (in-place)."""
    batch.x = (engineer_x(batch.x, norm.raw_frame_dim, norm.yaw_idx) - norm.x_mean) / norm.x_std
    batch.edge_attr = (batch.edge_attr - norm.e_mean) / norm.e_std
    batch.target = torch.clamp(batch.target / norm.y_scale, -1.0, 1.0)
    return batch
