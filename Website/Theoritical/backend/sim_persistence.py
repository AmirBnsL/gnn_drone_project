"""Save/load named simulation runs to JSON files."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from json_sanitize import sanitize_for_json

SAVES_DIR = Path(__file__).resolve().parent / "saved_simulations"
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 _\-]{0,63}$")
_RESERVED = frozenset({"con", "prn", "aux", "nul"})


def _ensure_dir() -> Path:
    SAVES_DIR.mkdir(parents=True, exist_ok=True)
    return SAVES_DIR


def validate_save_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        raise ValueError("Save name cannot be empty.")
    if ".." in n or "/" in n or "\\" in n:
        raise ValueError("Invalid save name.")
    if not _NAME_RE.match(n):
        raise ValueError(
            "Save name must be 1–64 chars: letters, digits, spaces, hyphen, underscore."
        )
    if n.lower() in _RESERVED:
        raise ValueError("Reserved save name.")
    return n


def _path_for(name: str) -> Path:
    safe = validate_save_name(name)
    return _ensure_dir() / f"{safe}.json"


def list_saves() -> List[str]:
    _ensure_dir()
    items = []
    for p in SAVES_DIR.glob("*.json"):
        items.append((p.stem, p.stat().st_mtime))
    items.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in items]


def _ndarray_to_list(arr: np.ndarray | None) -> List | None:
    if arr is None:
        return None
    return np.asarray(arr).tolist()


def save_run(name: str, session: Any) -> Dict[str, Any]:
    if not (
        (session.trajectory_central and session.trajectory_central.get("frames"))
        or (session.trajectory_decentral and session.trajectory_decentral.get("frames"))
    ):
        raise ValueError("No simulation trajectories to save. Run a simulation first.")

    payload: Dict[str, Any] = {
        "version": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "name": validate_save_name(name),
        "num_drones": session.num_drones,
        "scenario": session.scenario,
        "digit": session.digit,
        "seed": session.seed,
        "active_mode": session.active_mode,
        "start_pos": _ndarray_to_list(session.start_pos),
        "start_orn": _ndarray_to_list(session.start_orn),
        "slots": _ndarray_to_list(session.slots),
        "assignment": _ndarray_to_list(session.assignment),
        "trajectory_central": session.trajectory_central,
        "trajectory_decentral": session.trajectory_decentral,
        "dual_metadata": session.dual_metadata,
    }
    path = _path_for(name)
    clean = sanitize_for_json(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, separators=(",", ":"))
    return {"name": validate_save_name(name), "path": str(path)}


def load_run(name: str, session: Any) -> Dict[str, Any]:
    path = _path_for(name)
    if not path.is_file():
        raise FileNotFoundError(f"No saved simulation named '{validate_save_name(name)}'.")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    session.num_drones = int(data["num_drones"])
    session.scenario = data["scenario"]
    session.digit = data.get("digit")
    session.seed = int(data.get("seed", session.seed))
    session.active_mode = data.get("active_mode", "central")
    session.start_pos = (
        np.asarray(data["start_pos"], dtype=np.float32)
        if data.get("start_pos") is not None
        else None
    )
    session.start_orn = (
        np.asarray(data["start_orn"], dtype=np.float32)
        if data.get("start_orn") is not None
        else None
    )
    session.slots = (
        np.asarray(data["slots"], dtype=np.float32)
        if data.get("slots") is not None
        else None
    )
    session.assignment = (
        np.asarray(data["assignment"], dtype=np.int64)
        if data.get("assignment") is not None
        else None
    )
    session.trajectory_central = data.get("trajectory_central")
    session.trajectory_decentral = data.get("trajectory_decentral")
    session.dual_metadata = data.get("dual_metadata")
    session.obstacles_visible = session.scenario != "clean"
    session.last_error = None
    session.is_running = False
    return data


def delete_run(name: str) -> None:
    path = _path_for(name)
    if not path.is_file():
        raise FileNotFoundError(f"No saved simulation named '{validate_save_name(name)}'.")
    path.unlink()
