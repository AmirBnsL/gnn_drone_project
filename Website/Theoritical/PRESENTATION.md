# Theoretical Swarm Visualization — Architecture Presentation

Browser-based **theoretical swarm lab** that runs two physics episodes on the **same spawn, digit formation, and obstacles**, then compares **Central** (classical control) vs **Decentral (GNN)** in 3D playback.

- Same initial poses and slot layout; different **assignment** and **control** policies.
- Headless simulation on the backend; the browser only **replays** recorded frames.
- For setup and config knobs, see [README.md](README.md).

---

## 1. System architecture

The app splits into four layers: React UI, FastAPI session API, dual recorders, and a single PyFlyt/PyBullet world shared with training code in `merged_work`.

```mermaid
flowchart TB
  subgraph frontend [Frontend]
    UI[React_Vite_UI]
    R3F[React_Three_Fiber]
    Scene[Scene3D_layers]
    UI --> R3F --> Scene
  end

  subgraph api [API]
    FastAPI[main.py_FastAPI]
    Session[SimulationSession]
    FastAPI --> Session
  end

  subgraph sim [Simulation]
    Dual[dual_simulator.run_dual]
    Central[sim_recorder_central]
    Decentral[sim_recorder_decentral]
    Dual --> Central
    Dual --> Decentral
  end

  subgraph physics [Physics_and_training_parity]
    PyFlyt[PyFlyt_Aviary]
    PyBullet[PyBullet_collision]
    Merged[merged_work_models_creation]
    PyFlyt --> PyBullet
    Central --> PyFlyt
    Decentral --> PyFlyt
    Central --> Merged
    Decentral --> Merged
  end

  subgraph models [GNN_checkpoints]
    Assign[assignment_runner]
    Setpoint[setpoint_runner]
    Decentral --> Assign
    Decentral --> Setpoint
  end

  Scene -->|HTTP_api| FastAPI
  FastAPI -->|POST_simulate| Dual
  Dual -->|JSON_trajectories| FastAPI
  FastAPI -->|frames| Scene
```

| Layer | Role | Key paths |
|-------|------|-----------|
| Frontend | Setup, dual run, timeline, Central/Decentral tabs | `frontend/src/` |
| API | Session state, simulate, save/load runs | `backend/main.py`, `session.py` |
| Simulation | Record frame streams | `dual_simulator.py`, `sim_recorder_*.py` |
| Physics | 240 Hz integration, 1 m obstacle spheres | PyFlyt + PyBullet |
| Training parity | Obstacles, APF teacher, drone frames | `merged_work/models_creation/` |

---

## 2. User workflow

```mermaid
sequenceDiagram
  participant User
  participant Browser
  participant API as FastAPI
  participant Dual as run_dual
  participant Phys as PyFlyt_PyBullet

  User->>Browser: Setup drones and scenario
  Browser->>API: POST /api/config
  User->>Browser: Pick formation digit 0-9
  Browser->>API: POST /api/formation
  User->>Browser: Run Dual Sim
  Browser->>API: POST /api/simulate
  API->>Dual: identical ICs
  Dual->>Phys: central episode
  Dual->>Phys: decentral episode
  Phys-->>Dual: frame logs
  Dual-->>API: trajectory_central + trajectory_decentral
  API-->>Browser: JSON + metadata
  User->>Browser: Play scrub Central or Decentral tab
```

Optional: **Save run** / **Load** persists trajectories under `backend/saved_simulations/` without re-simulating.

---

## 3. Dual simulation entry

`run_dual` in [`backend/dual_simulator.py`](backend/dual_simulator.py) fixes one random seed episode, then runs two recorders.

```mermaid
flowchart LR
  IC[Shared_ICs_start_pos_slots_orn]
  Obs[Shared_obs_cfg]
  IC --> CentralRun[Central_recorder]
  IC --> DecentralRun[Decentral_recorder]
  Obs --> CentralRun
  Obs --> DecentralRun
  Hung[Hungarian_from_UI] --> CentralRun
  IC --> AssignGNN[assign_swarm_Bertsekas]
  AssignGNN --> DecentralRun
  CentralRun --> TC[trajectory_central]
  DecentralRun --> TD[trajectory_decentral]
```

| Input | Central | Decentral |
|-------|---------|-----------|
| Assignment | Hungarian from UI (`session.apply_formation`) | `assign_swarm` — Bertsekas + local negotiator GNN |
| Obstacles | Same `obs_cfg` built from central assignment | Same `obs_cfg` (fair comparison) |
| Physics cap | `MAX_STEPS_CONVERGED` when `RUN_UNTIL_CONVERGED` | Same |

---

## 4. Central simulation pipeline

**No neural networks.** Assignment is global at t=0; each step uses slot setpoints, sensor-gated APF, and a safety clamp before PyFlyt integrates.

```mermaid
flowchart LR
  Read[Read_drone_states]
  Stuck[Strict_stuck_check]
  Shift{Shift_eligible?}
  ZShift[+2m_Z_slot_shift]
  APF[APF_viz_plus_altitude_filter]
  Clamp[safety_clamp_setpoint]
  Set[env.set_setpoint]
  Step[env.step_PyBullet]
  Conv{max_slot_error_under_threshold?}
  Rec[Record_frame_every_N_steps]

  Read --> Stuck --> Shift
  Shift -->|yes| ZShift --> APF
  Shift -->|no| APF
  APF --> Clamp --> Set --> Step --> Conv
  Conv -->|not_yet| Read
  Conv -->|25_consecutive| Rec
  Step --> Rec
```

**Convergence:** all drones within **0.35 m** (3D) of their **assigned slot** for **25** consecutive steps (`max_assigned_slot_error` in `sim_frame_utils.py`).

---

## 5. Decentral simulation pipeline

Decentral adds **comm-radius messaging**, **Bertsekas assignment** (once), and a **setpoint GATv2** forward each control step, then optional APF and clamp.

```mermaid
flowchart TB
  subgraph once [Once_at_start]
    A1[assign_swarm]
    A2[Repair_bijection]
    A1 --> A2
  end

  subgraph eachStep [Each_physics_step]
    R1[Read_states]
    M1[exchange_messages_COMM_RADIUS]
    R2[Stuck_and_shift_gates]
    G1[build_setpoint_graph]
    G2[forward_setpoint_GNN]
    G3[pred_to_global_setpoints]
    APF[Optional_decentral_APF]
    CL[safety_clamp]
    ST[set_setpoint_and_step]
    R1 --> M1 --> R2 --> G1 --> G2 --> G3 --> APF --> CL --> ST
  end

  once --> eachStep
```

```mermaid
stateDiagram-v2
  [*] --> Setup: spawn_frame
  Setup --> Running: physics_loop
  Running --> Running: step_until_converged
  Running --> Done: converged_or_max_steps
  Done --> Playback: browser_timeline
  Playback --> [*]
```

Hybrid control blends GNN prediction with pull toward assigned slots (`SETPOINT_PRED_GAIN` / `SETPOINT_GOAL_GAIN` in `viz_sim_config.py`). Learned shift head is optional (`ENABLE_HYBRID_SLOT_SHIFT`); goal-repair Z-shift uses the same strict stuck gates as Central.

---

## 6. How drones talk — and what is faithful

Messages are **StateMsg** (pose, velocity, alert, stuck) and **ShiftProposal** when locally stuck. Delivery is symmetric: drone *i* receives from *j* only if ‖pos_i − pos_j‖ ≤ **10 m** (`COMM_RADIUS`).

```mermaid
sequenceDiagram
  participant Di as Drone_i
  participant Dj as Drone_j
  participant Sim as Simulator_loop

  Sim->>Di: prepare_outbox
  Sim->>Dj: prepare_outbox
  Note over Sim: distance_ij le 10m
  Sim->>Di: receive_from_Dj
  Sim->>Dj: receive_from_Di
  Di->>Di: inbox_states_and_shift_proposals
  Dj->>Dj: inbox_states_and_shift_proposals
```

### Faithful vs orchestrated

| Aspect | Faithful to decentral design | Orchestrated in this demo |
|--------|---------------------------|---------------------------|
| Who can message whom | Only pairs within `COMM_RADIUS` | Simulator knows all positions to test range |
| GNN graph edges | Built from comm-radius pairs | One batched forward over full swarm |
| Assignment slots visible | `SLOT_VISIBILITY_RADIUS` (10 m) | Single `assign_swarm` call returns global map |
| Physics | PyFlyt setpoints per drone | **One** shared PyBullet world, not N processes |
| UI comm mesh | Purple links when distance ≤ 10 m | `CommLinksLayer.tsx` |

**Takeaway:** algorithms respect **local visibility and comm range**; the **implementation** is a centralized Python loop that simulates radio delivery and runs leader-style GNN inference — not independent drones on separate machines.

---

## 7. PyBullet and PyFlyt — supervisor and monitor

| Component | Role |
|-----------|------|
| **PyFlyt** | High-level quad API: spawn `quadx` drones, position-mode setpoints (`env.set_mode(7)`), `env.step()` |
| **PyBullet** | Underlying rigid-body engine: integration, **1 m** sphere obstacles (`spawn_spheres`), collisions |
| **Backend loop** | **Supervisor:** writes setpoints each step from APF/GNN logic |
| **Backend loop** | **Monitor:** reads position, euler, body/world velocity; logs alerts from planar lidar |
| **Frontend** | **Playback only** — no physics during scrub; `trajectory.dt` = physics_dt × `RECORD_EVERY` |

Simulation runs **headless** (`render=False`). The 3D view shows **recorded** poses; obstacle spheres are drawn at **half** visual scale (`OBSTACLE_MESH_VIS_SCALE = 0.5`) while collision radius stays **1.0 m**.

---

## 8. Model inference

Checkpoints live in [`resources/checkpoints/`](resources/checkpoints/) (four files required for simulate).

| Model | Files | When | Output |
|-------|-------|------|--------|
| Local negotiator + Bertsekas | `strict_local_negotiator_best_v1.pt`, `bertsekas_best_digits.pt` | Once per decentral run | `assignment[i]` → slot index |
| Setpoint GATv2 | `best_gatv2_digits.pth`, `normalization_stats_digits.pt` | Each control step (`SETPOINT_CTRL_EVERY = 1`) | Normalized 4D pred + shift logit → `pred_to_global_setpoints` |

The setpoint GNN is trained to imitate **APF teacher** displacements using **16-ray planar lidar** and graph features from `merged_work`. At runtime, the Website may still apply an optional APF prior and `safety_clamp_setpoint` (`ENABLE_DECENTRAL_APF`).

```mermaid
flowchart LR
  Frames[Per_drone_41D_frames]
  Graph[Comm_radius_Graph]
  Norm[DatasetNormalizerV10]
  GAT[SetpointGATv2]
  Hybrid[pred_to_global_setpoints]
  PyF[PyFlyt_setpoint]

  Frames --> Graph --> Norm --> GAT --> Hybrid --> PyF
```

---

## 9. Central vs Decentral — at a glance

| | **Central** | **Decentral (GNN)** |
|---|-------------|---------------------|
| Assignment | Hungarian (`scipy.linear_sum_assignment`) | Bertsekas + learned negotiator |
| Control | Fixed slot targets + APF | GNN + goal blend + optional APF |
| Communication | None (implicit global plan) | `StateMsg` / `ShiftProposal` in 10 m |
| Slot shift | Strict stuck → +2 m Z (once) | Same + optional learned shift gate |
| Convergence metric | Assigned slot error &lt; 0.35 m | Same |

```mermaid
flowchart TB
  subgraph central [Central_tab]
    C1[Hungarian_t0]
    C2[APF_each_step]
    C1 --> C2
  end

  subgraph decentral [Decentral_tab]
    D1[Bertsekas_t0]
    D2[Messages_each_step]
    D3[GNN_each_step]
    D1 --> D2 --> D3
  end

  Shared[Same_spawn_and_obstacles]
  Shared --> central
  Shared --> decentral
```

---

## 10. Technology stack

**Backend**

- FastAPI, Uvicorn — HTTP API
- Pydantic — request bodies
- NumPy, SciPy — geometry, Hungarian assignment
- PyTorch, PyTorch Geometric — GNN inference
- PyBullet, PyFlyt — physics
- Shared training code — `merged_work/models_creation`

**Frontend**

- React 18, TypeScript, Vite
- Three.js, `@react-three/fiber`, `@react-three/drei`
- Crazyflie GLB — `frontend/public/models/crazyflie.glb`

---

## 11. Key files map

| Topic | Path |
|-------|------|
| API entry | `backend/main.py` |
| Dual run | `backend/dual_simulator.py` |
| Central recorder | `backend/sim_recorder_central.py` |
| Decentral recorder | `backend/sim_recorder_decentral.py` |
| Messaging | `backend/agents/comm_orchestrator.py`, `drone_agent.py` |
| Setpoint GNN | `backend/website_gnn/setpoint_runner.py` |
| Assignment GNN | `backend/website_gnn/assignment_runner.py` |
| Viz knobs | `backend/viz_sim_config.py` |
| 3D scene | `frontend/src/components/Scene3D.tsx` |
| Session | `backend/session.py` |

---

## 12. Notes for presenters

- **Converged** in the UI means every drone stayed within **0.35 m** of its **assigned slot ring** for 25 physics steps — not merely that playback ended.
- Top-bar metadata may show `max_steps` with **slot error** (e.g. `err 4.20m`) when the cap is hit before convergence.
- **Decentral** is a *centralized simulator executing decentralized algorithms* — ideal for theory and comparison, not a deployed multi-agent network.

---

*Generated for the Website/Theoritical app. Operational details: [README.md](README.md).*
