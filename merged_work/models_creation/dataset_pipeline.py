from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data

from merged_work.models_creation.digit_formations import sample_digit_offsets
from merged_work.models_creation.obstacle_scenarios import (
    ObstacleConfig,
    merge_obstacle_configs,
    place_path_obstacles,
    place_slot_obstacles,
    synchronized_slot_shift,
)

ScenarioKind = Literal["clean", "path", "slot", "both"]
SCENARIO_CHOICES: Tuple[ScenarioKind, ...] = ("clean", "path", "slot", "both")
SCENARIO_WEIGHTS: Tuple[Tuple[ScenarioKind, float], ...] = (
    ("clean", 0.1),
    ("path", 0.3),
    ("slot", 0.3),
    ("both", 0.3),
)

NUM_DIGIT_FORMATIONS = 10
COMM_RADIUS_DEFAULT = 10.0
ASSIGNMENT_COMM_RADIUS = 10.0
SLOT_VISIBILITY_RADIUS = 10.0


def sync_assignment_radii(
    comm_radius: float = ASSIGNMENT_COMM_RADIUS,
    slot_radius: float = SLOT_VISIBILITY_RADIUS,
) -> None:
    """Sync local_negotiator module globals (assignment GAT + slot bidding)."""
    import local_negotiator_v2 as ln

    ln.COMM_RADIUS = float(comm_radius)
    ln.SLOT_VISIBILITY_RADIUS = float(slot_radius)


@dataclass
class EpisodeConfig:
    seed: int
    num_drones: int
    digit: int
    scenario: ScenarioKind = "both"
    xy_limit: float = 10.0
    altitude_range: Tuple[float, float] = (0.5, 5.0)
    communication_radius: float = COMM_RADIUS_DEFAULT
    path_obstacles: int = 1
    slot_obstacles: int = 1
    obstacle_radius: float = 1.0
    spacing: float = 2.0


def sample_scenario(rng: np.random.Generator) -> ScenarioKind:
    choices, weights = zip(*SCENARIO_WEIGHTS)
    picked = rng.choice(choices, p=np.asarray(weights, dtype=np.float64))
    return str(picked)  # type: ignore[return-value]


def make_episode_config(
    seed: int,
    num_drones: int,
    digit: Optional[int] = None,
    scenario: Optional[ScenarioKind] = None,
) -> EpisodeConfig:
    rng = np.random.default_rng(seed)
    if digit is None:
        digit = int(rng.integers(0, NUM_DIGIT_FORMATIONS))
    if scenario is None:
        scenario = sample_scenario(rng)
    return EpisodeConfig(
        seed=seed,
        num_drones=num_drones,
        digit=digit,
        scenario=scenario,
    )


def sample_initial_state(cfg: EpisodeConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    start_pos = rng.uniform(-cfg.xy_limit, cfg.xy_limit, size=(cfg.num_drones, 3)).astype(
        np.float32
    )
    start_pos[:, 2] = rng.uniform(
        cfg.altitude_range[0], cfg.altitude_range[1], size=(cfg.num_drones,)
    )
    start_orn = np.zeros((cfg.num_drones, 3), dtype=np.float32)
    start_orn[:, 2] = rng.uniform(-np.pi, np.pi, size=(cfg.num_drones,))
    return start_pos, start_orn


def build_naive_slots(cfg: EpisodeConfig, start_pos: np.ndarray) -> np.ndarray:
    """Return naive digit slots centered on the swarm XY mean (uniform Z)."""
    formation_center = np.mean(start_pos[:, :2], axis=0)
    altitude = float(np.mean(start_pos[:, 2]))
    offsets = sample_digit_offsets(
        digit=cfg.digit, num_drones=cfg.num_drones, spacing=cfg.spacing
    )
    slots = np.zeros((cfg.num_drones, 3), dtype=np.float32)
    slots[:, :2] = formation_center[None, :] + offsets[:, :2]
    slots[:, 2] = altitude
    return slots


def assign_drones_to_slots(start_pos: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Hungarian matching as ground-truth assignment for training."""
    dist = np.linalg.norm(start_pos[:, None, :2] - slots[None, :, :2], axis=2)
    _, col_ind = linear_sum_assignment(dist)
    return col_ind.astype(np.int64)


def _scenario_obstacle_counts(cfg: EpisodeConfig) -> Tuple[int, int]:
    if cfg.scenario == "clean":
        return 0, 0
    if cfg.scenario == "path":
        return cfg.path_obstacles, 0
    if cfg.scenario == "slot":
        return 0, cfg.slot_obstacles
    return cfg.path_obstacles, cfg.slot_obstacles


def build_obstacles_for_episode(
    cfg: EpisodeConfig,
    start_pos: np.ndarray,
    slots: np.ndarray,
    assignment: np.ndarray,
) -> ObstacleConfig:
    rng = np.random.default_rng(cfg.seed + 17_000)
    n_path, n_slot = _scenario_obstacle_counts(cfg)
    assigned_slots = slots[assignment]
    configs = []
    if n_path > 0:
        configs.append(
            place_path_obstacles(
                start_pos=start_pos,
                slot_pos=assigned_slots,
                num_blocked_paths=n_path,
                radius=cfg.obstacle_radius,
                rng=rng,
            )
        )
    if n_slot > 0:
        configs.append(
            place_slot_obstacles(
                slot_pos=assigned_slots,
                num_blocked_slots=n_slot,
                radius=cfg.obstacle_radius,
                rng=rng,
            )
        )
    if not configs:
        return ObstacleConfig(
            positions=np.zeros((0, 3), dtype=np.float32),
            radii=np.zeros((0,), dtype=np.float32),
        )
    return merge_obstacle_configs(configs)


def build_assignment_graph(
    cfg: EpisodeConfig,
    start_pos: np.ndarray,
    slots: np.ndarray,
    assignment: np.ndarray,
) -> Data:
    """
    Negotiator/Bertsekas dataset sample (features built at train time).
    """
    drone_xy = torch.tensor(start_pos[:, :2], dtype=torch.float32)
    slots_xy = torch.tensor(slots[:, :2], dtype=torch.float32)
    y = torch.tensor(assignment, dtype=torch.long)

    return Data(
        drone_pos=drone_xy,
        slots=slots_xy,
        y=y,
        formation_id=torch.tensor(cfg.digit, dtype=torch.long),
        num_drones=torch.tensor(cfg.num_drones, dtype=torch.long),
        scenario=torch.tensor(SCENARIO_CHOICES.index(cfg.scenario), dtype=torch.long),
    )


def detect_slot_blocked(
    local_vel_window: np.ndarray,
    desired_direction: np.ndarray,
    obstacle_direction: np.ndarray,
    obstacle_distance_along_slot: float,
    vel_eps: float = 0.05,
    cos_eps: float = 0.3,
    max_along_dist: float = 0.5,
) -> bool:
    """
    Heuristic: stalled toward slot while an obstacle blocks the goal direction.
    """
    if local_vel_window.size == 0:
        return False
    speeds = np.linalg.norm(local_vel_window, axis=1)
    if float(np.max(speeds)) > vel_eps:
        return False
    dir_norm = np.linalg.norm(desired_direction[:2])
    obs_norm = np.linalg.norm(obstacle_direction[:2])
    if dir_norm < 1e-6 or obs_norm < 1e-6:
        return False
    goal_dir = desired_direction[:2] / dir_norm
    obs_dir = obstacle_direction[:2] / obs_norm
    if obstacle_distance_along_slot < 0.0 or obstacle_distance_along_slot > max_along_dist:
        return False
    cos_sim = float(np.dot(goal_dir, obs_dir))
    return cos_sim > cos_eps


def obstacles_to_lidar_array(obstacles: ObstacleConfig) -> np.ndarray:
    """(O, 3) with columns x, y, radius for rollout lidar/APF."""
    if obstacles.positions.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pos = obstacles.positions
    r = obstacles.radii
    return np.column_stack([pos[:, 0], pos[:, 1], r]).astype(np.float32)
