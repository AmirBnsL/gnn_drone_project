from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class ObstacleConfig:
    """Simple spherical obstacle configuration."""

    positions: np.ndarray  # (O, 3)
    radii: np.ndarray  # (O,)


def place_path_obstacles(
    start_pos: np.ndarray,
    slot_pos: np.ndarray,
    num_blocked_paths: int = 1,
    radius: float = 1.0,
    offset_factor: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    jitter_along: float = 0.15,
) -> ObstacleConfig:
    """
    Place obstacles on the path between a subset of drones and their slots.
    """
    num_drones = start_pos.shape[0]
    if num_drones == 0:
        return ObstacleConfig(
            positions=np.zeros((0, 3), dtype=np.float32),
            radii=np.zeros((0,), dtype=np.float32),
        )

    num_blocked_paths = max(0, min(num_blocked_paths, num_drones))
    if num_blocked_paths == 0:
        return ObstacleConfig(
            positions=np.zeros((0, 3), dtype=np.float32),
            radii=np.zeros((0,), dtype=np.float32),
        )

    if rng is None:
        rng = np.random.default_rng()
    drone_indices: List[int] = rng.choice(
        num_drones, size=num_blocked_paths, replace=False
    ).tolist()

    obstacles: List[Tuple[float, float, float]] = []
    for idx in drone_indices:
        p0 = start_pos[idx]
        p1 = slot_pos[idx]
        t = 0.5 + rng.uniform(-jitter_along, jitter_along)
        t = float(np.clip(t, 0.2, 0.8))
        mid = p0 + t * (p1 - p0)
        direction = p1 - p0
        ortho = np.array([-direction[1], direction[0], 0.0], dtype=np.float32)
        norm = float(np.linalg.norm(ortho[:2])) or 1.0
        ortho /= norm
        mid_xy = mid + offset_factor * radius * ortho
        obstacles.append((float(mid_xy[0]), float(mid_xy[1]), float(mid_xy[2])))

    positions = np.asarray(obstacles, dtype=np.float32)
    radii = np.full((positions.shape[0],), float(radius), dtype=np.float32)
    return ObstacleConfig(positions=positions, radii=radii)


def place_slot_obstacles(
    slot_pos: np.ndarray,
    num_blocked_slots: int = 1,
    radius: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> ObstacleConfig:
    """Place obstacles directly on top of a subset of slots."""
    num_slots = slot_pos.shape[0]
    if num_slots == 0:
        return ObstacleConfig(
            positions=np.zeros((0, 3), dtype=np.float32),
            radii=np.zeros((0,), dtype=np.float32),
        )

    num_blocked_slots = max(0, min(num_blocked_slots, num_slots))
    if num_blocked_slots == 0:
        return ObstacleConfig(
            positions=np.zeros((0, 3), dtype=np.float32),
            radii=np.zeros((0,), dtype=np.float32),
        )

    if rng is None:
        rng = np.random.default_rng()
    indices: List[int] = rng.choice(num_slots, size=num_blocked_slots, replace=False).tolist()

    obstacles = slot_pos[indices].astype(np.float32)
    radii = np.full((obstacles.shape[0],), float(radius), dtype=np.float32)
    return ObstacleConfig(positions=obstacles, radii=radii)


def merge_obstacle_configs(configs: Iterable[ObstacleConfig]) -> ObstacleConfig:
    """Merge multiple obstacle configs into one."""
    positions_list: List[np.ndarray] = []
    radii_list: List[np.ndarray] = []
    for cfg in configs:
        if cfg.positions.size == 0:
            continue
        positions_list.append(cfg.positions)
        radii_list.append(cfg.radii)

    if not positions_list:
        return ObstacleConfig(
            positions=np.zeros((0, 3), dtype=np.float32),
            radii=np.zeros((0,), dtype=np.float32),
        )

    positions = np.concatenate(positions_list, axis=0)
    radii = np.concatenate(radii_list, axis=0)
    return ObstacleConfig(positions=positions, radii=radii)


def synchronized_slot_shift(
    slots: np.ndarray,
    obstacles: ObstacleConfig | None = None,
    delta_z: float = 2.0,
) -> np.ndarray:
    """Deterministic global Z shift for full formation (obstacles arg kept for API compat)."""
    _ = obstacles
    shifted = np.asarray(slots, dtype=np.float32).copy()
    shifted[:, 2] += float(delta_z)
    return shifted
