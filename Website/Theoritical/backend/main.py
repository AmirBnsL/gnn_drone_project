"""FastAPI server for theoretical PyFlyt swarm visualization."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

_backend = Path(__file__).resolve().parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from path_setup import ensure_repo_path

ensure_repo_path()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from json_sanitize import sanitize_for_json
from resources import missing_checkpoints
from session import SESSION, SimMode
from sim_persistence import delete_run, list_saves, load_run, save_run

app = FastAPI(title="Theoretical Swarm Viz API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConfigBody(BaseModel):
    num_drones: int = Field(ge=5, le=20, default=12)
    scenario: Literal["clean", "path", "slot", "both"] = "clean"


class FormationBody(BaseModel):
    digit: int = Field(ge=0, le=9)


class ModeBody(BaseModel):
    mode: Literal["central", "decentral"] = "central"


class SaveNameBody(BaseModel):
    name: str = Field(min_length=1, max_length=64)


@app.get("/api/state")
def get_state():
    return SESSION.to_dict()


@app.get("/api/trajectory")
def get_trajectory(mode: Optional[SimMode] = Query(None)):
    traj = SESSION.get_trajectory(mode)
    if not traj:
        return {"frames": [], "dt": 1 / 240, "mode": mode or SESSION.active_mode}
    out = dict(traj)
    out["mode"] = mode or SESSION.active_mode
    return sanitize_for_json(out)


@app.get("/api/saves")
def get_saves():
    return {"names": list_saves()}


@app.post("/api/saves")
def post_save(body: SaveNameBody):
    try:
        info = save_run(body.name, SESSION)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return sanitize_for_json(info)


@app.post("/api/saves/load")
def post_save_load(body: SaveNameBody):
    try:
        load_run(body.name, SESSION)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    out = SESSION.to_dict()
    out["trajectory_central"] = SESSION.trajectory_central
    out["trajectory_decentral"] = SESSION.trajectory_decentral
    return sanitize_for_json(out)


@app.delete("/api/saves/{name}")
def delete_save(name: str):
    try:
        delete_run(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/api/mode")
def post_mode(body: ModeBody):
    SESSION.active_mode = body.mode  # type: ignore[assignment]
    return SESSION.to_dict()


@app.post("/api/config")
def post_config(body: ConfigBody):
    try:
        SESSION.apply_config(body.num_drones, body.scenario)  # type: ignore[arg-type]
    except Exception as exc:
        SESSION.last_error = str(exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SESSION.to_dict()


@app.post("/api/formation")
def post_formation(body: FormationBody):
    try:
        SESSION.apply_formation(body.digit)
    except Exception as exc:
        SESSION.last_error = str(exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SESSION.to_dict()


@app.post("/api/reset")
def post_reset():
    SESSION.reset()
    return SESSION.to_dict()


@app.post("/api/simulate")
def post_simulate():
    if SESSION.digit is None:
        raise HTTPException(status_code=400, detail="Select a formation (0-9) first.")
    if SESSION.start_pos is None:
        raise HTTPException(status_code=400, detail="Save drone configuration first.")
    if SESSION.slots is None or SESSION.assignment is None:
        raise HTTPException(
            status_code=400,
            detail="Formation slots missing; select a digit (0-9) again.",
        )

    missing = missing_checkpoints()
    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Missing GNN checkpoints: {', '.join(missing)}. "
                "Copy files to Website/Theoritical/resources/checkpoints/ "
                "(see resources/README.md)."
            ),
        )

    SESSION.is_running = True
    SESSION.last_error = None
    central: dict = {}
    decentral: dict = {}
    try:
        from dual_simulator import run_dual

        cfg = SESSION._cfg()
        central, decentral, meta = run_dual(
            cfg,
            start_pos=SESSION.start_pos.copy(),
            start_orn=SESSION.start_orn.copy(),
            slots=SESSION.slots.copy(),
            assignment_central=SESSION.assignment.copy(),
        )
        SESSION.set_dual_trajectories(central, decentral, meta)
    except FileNotFoundError as exc:
        SESSION.last_error = str(exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        SESSION.last_error = str(exc)
        SESSION.is_running = False
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        SESSION.is_running = False

    out = SESSION.to_dict()
    out["trajectory"] = SESSION.get_trajectory()
    out["trajectory_central"] = central
    out["trajectory_decentral"] = decentral
    return sanitize_for_json(out)
