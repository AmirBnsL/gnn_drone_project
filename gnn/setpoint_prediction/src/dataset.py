import math
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader


# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_NAMES = [
    "vel_x", "vel_y", "vel_z",
    "ang_x", "ang_y",
    "sin_yaw", "cos_yaw",
    "form_a", "form_rect", "form_tri",
]
LABEL_NAMES = ["Δx_local", "Δy_local", "Δz", "Δyaw"]
EDGE_NAMES  = ["rel_x", "rel_y", "rel_z", "distance"]


# ── Helper Functions ───────────────────────────────────────────────────────────
def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ── Dataset ────────────────────────────────────────────────────────────────────
class DroneSwarmDataset(InMemoryDataset):
    """
    Loader for pre-built .pt split files from datacollection.py.

    Fields per graph:
      x            [N, 25]  – Raw node features
      target       [N, 4]   – Expert setpoint error [Δx, Δy, Δz, Δyaw] → LABEL
      y            [N, 4]   – Motor PWM (not used in setpoint task)
      edge_index   [2, E]   – Communication graph
      edge_attr    [E, 3]   – Relative 3D position (pⱼ - pᵢ)
      pos          [N, 3]   – World position (visualization only)
    """

    def __init__(self, split_path: str | Path):
        split_path = Path(split_path)
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {split_path}")

        payload = torch.load(split_path, weights_only=False)

        self.formation_names = payload.get("formation_names", ["a", "rectangle", "triangle"])
        self.split_name      = payload.get("split_name", "unknown")

        super().__init__(root="")

        self.data   = payload["data"]
        self.slices = payload["slices"]


# ── Normalizer ─────────────────────────────────────────────────────────────────
class SetpointNormalizer:
    """
    Feature engineering for setpoint prediction.

    - Selects useful x columns (drops constant obstacle placeholders)
    - Replaces yaw with sin(yaw) and cos(yaw)
    - Z-score normalizes velocity & angular features
    - Leaves formation one-hot at {0, 1}
    - Transforms Δx, Δy labels to local body frame
    - Z-score normalizes setpoint labels
    - Adds Euclidean distance to edges → 4D
    - Z-score normalizes edge features

    At inference: real_setpoint = denormalize_target(model_output, original_yaw)
    """

    X_KEEP_COLS = [0, 1, 2, 3, 4, 5, 22, 23, 24]  # ang_z is at index 5
    N_ONEHOT    = 3

    def __init__(self, x_mean, x_std, target_mean, target_std, edge_mean, edge_std):
        self.x_mean      = x_mean       # [10]
        self.x_std       = x_std        # [10]
        self.target_mean = target_mean  # [4]
        self.target_std  = target_std   # [4]
        self.edge_mean   = edge_mean    # [4]
        self.edge_std    = edge_std     # [4]

    @classmethod
    def fit(cls, train_dataset: DroneSwarmDataset) -> "SetpointNormalizer":
        """Compute stats from train set. Call ONCE."""
        # Process inputs (x)
        raw_x_selected = train_dataset._data.x[:, cls.X_KEEP_COLS]
        yaw = wrap_angle(raw_x_selected[:, 5])
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        # Replace yaw with sin/cos representation
        raw_x_transformed = torch.cat([raw_x_selected[:, :5], sin_yaw, cos_yaw, raw_x_selected[:, 6:]], dim=1)

        xm = raw_x_transformed.mean(0); xs = raw_x_transformed.std(0).clamp(min=1e-6)
        xm[-cls.N_ONEHOT:] = 0.0  # one-hot pass-through
        xs[-cls.N_ONEHOT:] = 1.0

        # Process targets (y)
        raw_t = train_dataset._data.target
        delta_x, delta_y = raw_t[:, 0], raw_t[:, 1]
        
        # Note: yaw is from the *input* features, representing the drone's current state
        current_yaw = wrap_angle(train_dataset._data.x[:, 5])
        x_local = delta_x * torch.cos(current_yaw) + delta_y * torch.sin(current_yaw)
        y_local = -delta_x * torch.sin(current_yaw) + delta_y * torch.cos(current_yaw)
        
        raw_t_local = torch.stack([x_local, y_local, raw_t[:, 2], wrap_angle(raw_t[:, 3])], dim=1)
        
        tm = raw_t_local.mean(0); ts = raw_t_local.std(0).clamp(min=1e-6)

        # Process edges (e)
        raw_e = train_dataset._data.edge_attr
        raw_dist = torch.norm(raw_e, dim=1, keepdim=True)
        raw_e_full = torch.cat([raw_e, raw_dist], dim=1)
        em = torch.zeros(4); es = raw_e_full.std(0).clamp(min=1e-6)

        return cls(xm, xs, tm, ts, em, es)

    def transform_graph(self, graph: Data) -> Data:
        """Raw graph → GNN-ready graph."""
        # --- Input Features (x) ---
        x_sel = graph.x[:, self.X_KEEP_COLS]
        
        # Angle Wrapping and Sin/Cos Representation for Yaw
        yaw = wrap_angle(x_sel[:, 5])
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        x_transformed = torch.cat([x_sel[:, :5], sin_yaw, cos_yaw, x_sel[:, 6:]], dim=1)
        
        x_norm = (x_transformed - self.x_mean) / self.x_std

        # --- Target Labels (y) ---
        delta_x, delta_y = graph.target[:, 0], graph.target[:, 1]
        
        # Local Body Frame Transformation using the same current yaw
        current_yaw = wrap_angle(graph.x[:, 5])
        x_local = delta_x * torch.cos(current_yaw) + delta_y * torch.sin(current_yaw)
        y_local = -delta_x * torch.sin(current_yaw) + delta_y * torch.cos(current_yaw)
        
        # Assemble local targets and wrap delta_yaw
        y_local_transformed = torch.stack([x_local, y_local, graph.target[:, 2], wrap_angle(graph.target[:, 3])], dim=1)
        
        y_norm = (y_local_transformed - self.target_mean) / self.target_std

        # --- Edge Features (edge_attr) ---
        dist = torch.norm(graph.edge_attr, dim=1, keepdim=True)
        edge_full = torch.cat([graph.edge_attr, dist], dim=1)
        e_norm = (edge_full - self.edge_mean) / self.edge_std

        return Data(
            x          = x_norm,
            y          = y_norm,
            edge_index = graph.edge_index,
            edge_attr  = e_norm,
            pos        = graph.pos,
            # Pass original yaw for inverse transform during evaluation
            original_yaw = current_yaw.unsqueeze(1),
            formation_id = graph.formation_id,
            episode_id   = graph.episode_id,
            step_idx     = graph.step_idx,
            num_drones   = graph.num_drones,
        )

    def denormalize_target(self, y_norm: torch.Tensor, original_yaw: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized predictions → physical setpoints (metres/radians).
        Performs the inverse transform from local to global frame.
        """
        y_physical_local = y_norm * self.target_std + self.target_mean
        
        # Inverse transform from Local Body Frame to Global Frame
        x_local, y_local = y_physical_local[:, 0], y_physical_local[:, 1]
        
        # Ensure yaw is in the correct shape
        if original_yaw.dim() == 2 and original_yaw.shape[1] == 1:
            original_yaw = original_yaw.squeeze(1)

        cos_yaw = torch.cos(original_yaw)
        sin_yaw = torch.sin(original_yaw)

        delta_x_global = x_local * cos_yaw - y_local * sin_yaw
        delta_y_global = x_local * sin_yaw + y_local * cos_yaw
        
        return torch.stack([
            delta_x_global,
            delta_y_global,
            y_physical_local[:, 2], # Δz
            y_physical_local[:, 3]  # Δyaw
        ], dim=1)

    # ── Serialization ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "x_keep_cols":   self.X_KEEP_COLS,
            "x_mean":        self.x_mean.tolist(),
            "x_std":         self.x_std.tolist(),
            "target_mean":   self.target_mean.tolist(),
            "target_std":    self.target_std.tolist(),
            "edge_mean":     self.edge_mean.tolist(),
            "edge_std":      self.edge_std.tolist(),
            "input_names":   INPUT_NAMES,
            "label_names":   LABEL_NAMES,
            "edge_names":    EDGE_NAMES,
            "input_dim":     len(INPUT_NAMES),
            "output_dim":    len(LABEL_NAMES),
            "edge_dim":      len(EDGE_NAMES),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SetpointNormalizer":
        return cls(
            x_mean      = torch.tensor(d["x_mean"]),
            x_std       = torch.tensor(d["x_std"]),
            target_mean = torch.tensor(d["target_mean"]),
            target_std  = torch.tensor(d["target_std"]),
            edge_mean   = torch.tensor(d["edge_mean"]),
            edge_std    = torch.tensor(d["edge_std"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SetpointNormalizer":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── Normalized Dataset Wrapper ─────────────────────────────────────────────────
class NormalizedDroneDataset(TorchDataset):
    """Applies SetpointNormalizer on-the-fly."""

    def __init__(self, base_dataset: DroneSwarmDataset, normalizer: SetpointNormalizer):
        self.base       = base_dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Data:
        return self.normalizer.transform_graph(self.base[idx])


# ── Convenience function ───────────────────────────────────────────────────────
def get_dataloaders(
    dataset_dir: str | Path,
    dataset_name: str = "setpoint_mixed_v1_mixed_formations",
    batch_size: int = 32,
    num_workers: int = 0,
    normalizer_path: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, SetpointNormalizer]:
    """
    One-liner to get train/val/test DataLoaders + normalizer.

    Args:
        dataset_dir:     Path to the datasets/ folder
        dataset_name:    Base name of the dataset
        batch_size:      Batch size for training
        num_workers:     DataLoader workers (0 on Windows)
        normalizer_path: If provided, load pre-computed stats instead of re-fitting

    Returns:
        (train_loader, val_loader, test_loader, normalizer)
    """
    dataset_dir = Path(dataset_dir)

    train_ds = DroneSwarmDataset(dataset_dir / f"{dataset_name}_train.pt")
    val_ds   = DroneSwarmDataset(dataset_dir / f"{dataset_name}_val.pt")
    test_ds  = DroneSwarmDataset(dataset_dir / f"{dataset_name}_test.pt")

    if normalizer_path and Path(normalizer_path).exists():
        normalizer = SetpointNormalizer.load(normalizer_path)
    else:
        normalizer = SetpointNormalizer.fit(train_ds)

    norm_train = NormalizedDroneDataset(train_ds, normalizer)
    norm_val   = NormalizedDroneDataset(val_ds,   normalizer)
    norm_test  = NormalizedDroneDataset(test_ds,  normalizer)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(norm_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(norm_val,   batch_size=batch_size * 2, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(norm_test,  batch_size=batch_size * 2, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, normalizer


import math
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader


# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_NAMES = [
    "vel_x", "vel_y", "vel_z",
    "ang_x", "ang_y",
    "sin_yaw", "cos_yaw",  # Replaced ang_z
    "form_a", "form_rect", "form_tri",
]
LABEL_NAMES = ["Δx_local", "Δy_local", "Δz", "Δyaw"]  # Updated labels
EDGE_NAMES  = ["rel_x", "rel_y", "rel_z", "distance"]


# ── Helper Functions ───────────────────────────────────────────────────────────
def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ── Dataset ────────────────────────────────────────────────────────────────────
class DroneSwarmDataset(InMemoryDataset):
    """
    Loader for pre-built .pt split files from datacollection.py.

    Fields per graph:
      x            [N, 25]  – Raw node features
      target       [N, 4]   – Expert setpoint error [Δx, Δy, Δz, Δyaw] → LABEL
      y            [N, 4]   – Motor PWM (not used in setpoint task)
      edge_index   [2, E]   – Communication graph
      edge_attr    [E, 3]   – Relative 3D position (pⱼ - pᵢ)
      pos          [N, 3]   – World position (visualization only)
    """

    def __init__(self, split_path: str | Path):
        split_path = Path(split_path)
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {split_path}")

        payload = torch.load(split_path, weights_only=False)

        self.formation_names = payload.get("formation_names", ["a", "rectangle", "triangle"])
        self.split_name      = payload.get("split_name", "unknown")

        super().__init__(root="")

        self.data   = payload["data"]
        self.slices = payload["slices"]


# ── Normalizer ─────────────────────────────────────────────────────────────────
class SetpointNormalizer:
    """
    Feature engineering for setpoint prediction.

    - Selects useful x columns (drops constant obstacle placeholders)
    - Replaces yaw with sin(yaw) and cos(yaw)
    - Z-score normalizes velocity & angular features
    - Leaves formation one-hot at {0, 1}
    - Transforms Δx, Δy labels to local body frame
    - Z-score normalizes setpoint labels
    - Adds Euclidean distance to edges → 4D
    - Z-score normalizes edge features

    At inference: real_setpoint = denormalize_target(model_output)
    """

    X_KEEP_COLS = [0, 1, 2, 3, 4, 5, 22, 23, 24]  # ang_z is at index 5
    N_ONEHOT    = 3

    def __init__(self, x_mean, x_std, target_mean, target_std, edge_mean, edge_std):
        self.x_mean      = x_mean       # [10]
        self.x_std       = x_std        # [10]
        self.target_mean = target_mean  # [4]
        self.target_std  = target_std   # [4]
        self.edge_mean   = edge_mean    # [4]
        self.edge_std    = edge_std     # [4]

    @classmethod
    def fit(cls, train_dataset: DroneSwarmDataset) -> "SetpointNormalizer":
        """Compute stats from train set. Call ONCE."""
        # Process inputs (x)
        raw_x_selected = train_dataset._data.x[:, cls.X_KEEP_COLS]
        yaw = wrap_angle(raw_x_selected[:, 5])
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        # Replace yaw with sin/cos representation
        raw_x_transformed = torch.cat([raw_x_selected[:, :5], sin_yaw, cos_yaw, raw_x_selected[:, 6:]], dim=1)

        xm = raw_x_transformed.mean(0); xs = raw_x_transformed.std(0).clamp(min=1e-6)
        xm[-cls.N_ONEHOT:] = 0.0  # one-hot pass-through
        xs[-cls.N_ONEHOT:] = 1.0

        # Process targets (y)
        raw_t = train_dataset._data.target
        delta_x, delta_y = raw_t[:, 0], raw_t[:, 1]
        
        # Note: yaw is from the *input* features, representing the drone's current state
        x_local = delta_x * torch.cos(yaw) + delta_y * torch.sin(yaw)
        y_local = -delta_x * torch.sin(yaw) + delta_y * torch.cos(yaw)
        
        raw_t_local = torch.stack([x_local, y_local, raw_t[:, 2], wrap_angle(raw_t[:, 3])], dim=1)
        
        tm = raw_t_local.mean(0); ts = raw_t_local.std(0).clamp(min=1e-6)

        # Process edges (e)
        raw_e = train_dataset._data.edge_attr
        raw_dist = torch.norm(raw_e, dim=1, keepdim=True)
        raw_e_full = torch.cat([raw_e, raw_dist], dim=1)
        em = torch.zeros(4); es = raw_e_full.std(0).clamp(min=1e-6)

        return cls(xm, xs, tm, ts, em, es)

    def transform_graph(self, graph: Data) -> Data:
        """Raw graph → GNN-ready graph."""
        # --- Input Features (x) ---
        x_sel = graph.x[:, self.X_KEEP_COLS]
        
        # Angle Wrapping and Sin/Cos Representation for Yaw
        yaw = wrap_angle(x_sel[:, 5])
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        x_transformed = torch.cat([x_sel[:, :5], sin_yaw, cos_yaw, x_sel[:, 6:]], dim=1)
        
        x_norm = (x_transformed - self.x_mean) / self.x_std

        # --- Target Labels (y) ---
        delta_x, delta_y = graph.target[:, 0], graph.target[:, 1]
        
        # Local Body Frame Transformation
        x_local = delta_x * torch.cos(yaw) + delta_y * torch.sin(yaw)
        y_local = -delta_x * torch.sin(yaw) + delta_y * torch.cos(yaw)
        
        # Assemble local targets and wrap delta_yaw
        y_local_transformed = torch.stack([x_local, y_local, graph.target[:, 2], wrap_angle(graph.target[:, 3])], dim=1)
        
        y_norm = (y_local_transformed - self.target_mean) / self.target_std

        # --- Edge Features (edge_attr) ---
        dist = torch.norm(graph.edge_attr, dim=1, keepdim=True)
        edge_full = torch.cat([graph.edge_attr, dist], dim=1)
        e_norm = (edge_full - self.edge_mean) / self.edge_std

        return Data(
            x          = x_norm,
            y          = y_norm,
            edge_index = graph.edge_index,
            edge_attr  = e_norm,
            pos        = graph.pos,
            # Pass original yaw for inverse transform during evaluation
            original_yaw = yaw.unsqueeze(1),
            formation_id = graph.formation_id,
            episode_id   = graph.episode_id,
            step_idx     = graph.step_idx,
            num_drones   = graph.num_drones,
        )

    def denormalize_target(self, y_norm: torch.Tensor, original_yaw: torch.Tensor | None = None) -> torch.Tensor:
        """
        Convert normalized predictions → physical setpoints (metres/radians).
        If original_yaw is provided, it performs the inverse transform from local to global frame.
        """
        y_physical_local = y_norm * self.target_std + self.target_mean
        
        if original_yaw is None:
            return y_physical_local

        # Inverse transform from Local Body Frame to Global Frame
        x_local, y_local = y_physical_local[:, 0], y_physical_local[:, 1]
        
        # Ensure yaw is in the correct shape
        if original_yaw.dim() == 2 and original_yaw.shape[1] == 1:
            original_yaw = original_yaw.squeeze(1)

        cos_yaw = torch.cos(original_yaw)
        sin_yaw = torch.sin(original_yaw)

        delta_x_global = x_local * cos_yaw - y_local * sin_yaw
        delta_y_global = x_local * sin_yaw + y_local * cos_yaw
        
        return torch.stack([
            delta_x_global,
            delta_y_global,
            y_physical_local[:, 2], # Δz
            y_physical_local[:, 3]  # Δyaw
        ], dim=1)

    # ── Serialization ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "x_keep_cols":   self.X_KEEP_COLS,
            "x_mean":        self.x_mean.tolist(),
            "x_std":         self.x_std.tolist(),
            "target_mean":   self.target_mean.tolist(),
            "target_std":    self.target_std.tolist(),
            "edge_mean":     self.edge_mean.tolist(),
            "edge_std":      self.edge_std.tolist(),
            "input_names":   INPUT_NAMES,
            "label_names":   LABEL_NAMES,
            "edge_names":    EDGE_NAMES,
            "input_dim":     len(INPUT_NAMES),
            "output_dim":    len(LABEL_NAMES),
            "edge_dim":      len(EDGE_NAMES),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SetpointNormalizer":
        return cls(
            x_mean      = torch.tensor(d["x_mean"]),
            x_std       = torch.tensor(d["x_std"]),
            target_mean = torch.tensor(d["target_mean"]),
            target_std  = torch.tensor(d["target_std"]),
            edge_mean   = torch.tensor(d["edge_mean"]),
            edge_std    = torch.tensor(d["edge_std"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SetpointNormalizer":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ── Normalized Dataset Wrapper ─────────────────────────────────────────────────
class NormalizedDroneDataset(TorchDataset):
    """Applies SetpointNormalizer on-the-fly."""

    def __init__(self, base_dataset: DroneSwarmDataset, normalizer: SetpointNormalizer):
        self.base       = base_dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Data:
        return self.normalizer.transform_graph(self.base[idx])


# ── Convenience function ───────────────────────────────────────────────────────
def get_dataloaders(
    dataset_dir: str | Path,
    dataset_name: str = "setpoint_mixed_v1_mixed_formations",
    batch_size: int = 32,
    num_workers: int = 0,
    normalizer_path: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, SetpointNormalizer]:
    """
    One-liner to get train/val/test DataLoaders + normalizer.

    Args:
        dataset_dir:     Path to the datasets/ folder
        dataset_name:    Base name of the dataset
        batch_size:      Batch size for training
        num_workers:     DataLoader workers (0 on Windows)
        normalizer_path: If provided, load pre-computed stats instead of re-fitting

    Returns:
        (train_loader, val_loader, test_loader, normalizer)
    """
    dataset_dir = Path(dataset_dir)

    train_ds = DroneSwarmDataset(dataset_dir / f"{dataset_name}_train.pt")
    val_ds   = DroneSwarmDataset(dataset_dir / f"{dataset_name}_val.pt")
    test_ds  = DroneSwarmDataset(dataset_dir / f"{dataset_name}_test.pt")

    if normalizer_path and Path(normalizer_path).exists():
        normalizer = SetpointNormalizer.load(normalizer_path)
    else:
        normalizer = SetpointNormalizer.fit(train_ds)

    norm_train = NormalizedDroneDataset(train_ds, normalizer)
    norm_val   = NormalizedDroneDataset(val_ds,   normalizer)
    norm_test  = NormalizedDroneDataset(test_ds,  normalizer)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(norm_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(norm_val,   batch_size=batch_size * 2, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(norm_test,  batch_size=batch_size * 2, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, normalizer
