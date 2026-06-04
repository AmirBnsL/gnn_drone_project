"""3D-aware obstacle utilities for Website runtime safety layers."""

from __future__ import annotations

import numpy as np

import viz_sim_config as vcfg


def obstacles_to_3d_array(obs_cfg) -> np.ndarray:
    """Return (O,4) as [x, y, radius, z] from obstacle config."""
    if obs_cfg.positions.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    pos = np.asarray(obs_cfg.positions, dtype=np.float32)
    radii = np.asarray(obs_cfg.radii, dtype=np.float32)
    return np.column_stack([pos[:, 0], pos[:, 1], radii, pos[:, 2]]).astype(np.float32)


def filter_obstacles_by_altitude(
    drone_z: float,
    obstacles_3d: np.ndarray,
    margin: float | None = None,
) -> np.ndarray:
    """
    Return (O',3) [x, y, radius] for obstacles intersecting drone altitude band.

    Each sphere spans [z-r, z+r]. We keep obstacles where drone_z is within
    [z-r-margin, z+r+margin].
    """
    if obstacles_3d.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    margin_v = float(vcfg.OBSTACLE_ALTITUDE_MARGIN if margin is None else margin)
    obs = np.asarray(obstacles_3d, dtype=np.float32)
    obs_z = obs[:, 3]
    obs_r = obs[:, 2]
    in_range = (drone_z >= (obs_z - obs_r - margin_v)) & (drone_z <= (obs_z + obs_r + margin_v))
    if not np.any(in_range):
        return np.zeros((0, 3), dtype=np.float32)
    return obs[in_range, :3].copy()
