"""In-memory simulation session for the theoretical visualization API."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from merged_work.models_creation.dataset_pipeline import (
    EpisodeConfig,
    assign_drones_to_slots,
    build_naive_slots,
    make_episode_config,
    sample_initial_state,
)

ScenarioKind = Literal["clean", "path", "slot", "both"]
SimMode = Literal["central", "decentral"]


def _vec3(arr: np.ndarray, i: int) -> List[float]:
    return [float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])]


def _drones_payload(start_pos: np.ndarray, start_orn: np.ndarray) -> List[Dict[str, Any]]:
    out = []
    for i in range(start_pos.shape[0]):
        out.append(
            {
                "id": i,
                "pos": _vec3(start_pos, i),
                "yaw": float(start_orn[i, 2]),
                "alert": False,
            }
        )
    return out


@dataclass
class SimulationSession:
    num_drones: int = 12
    scenario: ScenarioKind = "clean"
    digit: Optional[int] = None
    seed: int = 42
    start_pos: Optional[np.ndarray] = None
    start_orn: Optional[np.ndarray] = None
    slots: Optional[np.ndarray] = None
    assignment: Optional[np.ndarray] = None
    trajectory_central: Optional[Dict[str, Any]] = None
    trajectory_decentral: Optional[Dict[str, Any]] = None
    dual_metadata: Optional[Dict[str, Any]] = None
    active_mode: SimMode = "central"
    obstacles_visible: bool = False
    is_running: bool = False
    last_error: Optional[str] = None

    def _cfg(self, seed: Optional[int] = None) -> EpisodeConfig:
        d = self.digit if self.digit is not None else 0
        return make_episode_config(
            seed=seed if seed is not None else self.seed,
            num_drones=self.num_drones,
            digit=d,
            scenario=self.scenario,
        )

    def apply_config(self, num_drones: int, scenario: ScenarioKind) -> None:
        self.num_drones = int(num_drones)
        self.scenario = scenario
        self.digit = None
        self.trajectory_central = None
        self.trajectory_decentral = None
        self.dual_metadata = None
        self.obstacles_visible = False
        self.slots = None
        self.assignment = None
        self.last_error = None
        self.seed = int(time.time()) % 1_000_000
        cfg = self._cfg(self.seed)
        self.start_pos, self.start_orn = sample_initial_state(cfg)

    def apply_formation(self, digit: int) -> None:
        if self.start_pos is None:
            raise ValueError("Configure drones first (Save in setup panel).")
        self.digit = int(digit)
        self.trajectory_central = None
        self.trajectory_decentral = None
        self.dual_metadata = None
        self.obstacles_visible = False
        cfg = self._cfg()
        self.slots = build_naive_slots(cfg, self.start_pos)
        self.assignment = assign_drones_to_slots(self.start_pos, self.slots)

    def reset(self) -> None:
        self.apply_config(self.num_drones, self.scenario)

    def set_dual_trajectories(
        self,
        central: Dict[str, Any],
        decentral: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> None:
        self.trajectory_central = central
        self.trajectory_decentral = decentral
        self.dual_metadata = meta
        self.obstacles_visible = self.scenario != "clean"

    def get_trajectory(self, mode: SimMode | None = None) -> Optional[Dict[str, Any]]:
        m = mode or self.active_mode
        if m == "decentral":
            return self.trajectory_decentral
        return self.trajectory_central

    def to_dict(self) -> Dict[str, Any]:
        center = [0.0, 0.0]
        slots_list: List[List[float]] = []
        assignment_list: List[int] = []
        if self.start_pos is not None and self.start_pos.size > 0:
            center = [
                float(np.mean(self.start_pos[:, 0])),
                float(np.mean(self.start_pos[:, 1])),
            ]
        if self.slots is not None:
            slots_list = [_vec3(self.slots, i) for i in range(self.slots.shape[0])]
        if self.assignment is not None:
            assignment_list = [int(x) for x in self.assignment.tolist()]

        traj = self.get_trajectory()
        obstacles: List[List[float]] = []
        if traj and traj.get("frames"):
            obstacles = traj["frames"][0].get("obstacles", [])
        elif self.obstacles_visible:
            obstacles = []

        drones = (
            _drones_payload(self.start_pos, self.start_orn)
            if self.start_pos is not None
            else []
        )

        has_c = bool(
            self.trajectory_central and len(self.trajectory_central.get("frames", [])) > 0
        )
        has_d = bool(
            self.trajectory_decentral
            and len(self.trajectory_decentral.get("frames", [])) > 0
        )

        return {
            "num_drones": self.num_drones,
            "scenario": self.scenario,
            "digit": self.digit,
            "center": center,
            "drones": drones,
            "slots": slots_list,
            "assignment": assignment_list,
            "obstacles": obstacles if self.obstacles_visible else [],
            "has_trajectory": has_c or has_d,
            "has_trajectory_central": has_c,
            "has_trajectory_decentral": has_d,
            "active_mode": self.active_mode,
            "frame_count": len(traj.get("frames", [])) if traj else 0,
            "dt": float(traj.get("dt", 1 / 240)) if traj else 1 / 240,
            "is_running": self.is_running,
            "last_error": self.last_error,
            "dual_metadata": self.dual_metadata,
        }


SESSION = SimulationSession()
SESSION.apply_config(12, "clean")
