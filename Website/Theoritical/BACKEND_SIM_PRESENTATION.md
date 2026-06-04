# Backend Dual Simulation — Presentation (Central vs Decentral)

**Scope:** headless physics + control only (`Website/Theoritical/backend`).  
**Goal:** same spawn, digit formation, obstacles — two policies compared fairly.

---

## Slide 1 — One episode, two controllers

```mermaid
flowchart TB
  subgraph shared [Shared episode — run_dual]
    S[seed + digit + num_drones]
    P[start_pos / start_orn]
    L[slot targets on digit]
    O[obstacle layout from central assignment]
    S --> P --> L
    L --> O
  end

  shared --> C[Central recorder]
  shared --> D[Decentral recorder]

  C --> TC[trajectory_central + metadata]
  D --> TD[trajectory_decentral + metadata]

  H[Hungarian assignment — UI/session] --> C
  G[assign_swarm — Bertsekas + GNN] --> D
```

| Output field | Meaning |
|--------------|---------|
| `steps` | Physics steps until stop |
| `converged` | All drones within **0.35 m** of **assigned** slot for **25** steps |
| `stopped_reason` | `converged` or `max_steps` |
| `assignment_*` | Who flies to which slot ring |

**Entry:** [`dual_simulator.py`](backend/dual_simulator.py) → `record_central` + `record_episode_frames_decentral`.

---

## Slide 2 — Swarm geometry (what is identical)

```mermaid
flowchart LR
  subgraph world [PyFlyt Aviary — PyBullet]
    D0((drone 0))
    D1((drone 1))
    Dn((drone N))
    S0[[slot 0]]
    S1[[slot 1]]
    Sn[[slot N]]
    Obs((obstacle sphere r=1m))
  end

  D0 -.->|assigned| S0
  D1 -.->|assigned| S1
  Dn -.->|assigned| Sn
```

- **240 Hz** integration (`PHYSICS_HZ = 240`), position-mode setpoints (`env.set_mode(7)`).
- Slots = digit formation targets; **assignment** maps drone → slot index.
- Obstacles: **1.0 m** radius spheres from `build_obstacles_for_episode` (same `obs_cfg` for both runs).

---

## Slide 3 — Central pipeline (classical, no GNN)

```mermaid
flowchart LR
  A0[t=0: Hungarian assignment] --> LOOP

  subgraph LOOP [Each physics step]
    R[read poses + yaw]
    ST[strict stuck gates]
    SH{shift once?}
    Z[+2 m Z on all slots]
    APF[APF setpoint + altitude filter]
    CL[safety_clamp]
    SP[env.set_setpoint]
    PB[env.step]
    CV{max slot error < 0.35 m?}
    R --> ST --> SH
    SH -->|yes| Z --> APF
    SH -->|no| APF
    APF --> CL --> SP --> PB --> CV
  end
```

| Property | Central |
|----------|---------|
| Assignment | Global optimum at start (`scipy.linear_sum_assignment`) |
| Control | Fixed slot targets + viz APF repulsion |
| Messages | None — implicit global plan |
| Shift | Strict stuck → one-time **+2 m Z** on formation |

**Code:** [`sim_recorder_central.py`](backend/sim_recorder_central.py)

---

## Slide 4 — Decentral pipeline (learned swarm)

```mermaid
flowchart TB
  subgraph t0 [Once at t=0]
    AS[assign_swarm]
    RP[repair to bijection if needed]
    AS --> RP
  end

  subgraph step [Each physics step]
    R[read states]
    MSG[exchange_messages — COMM_RADIUS 10 m]
    ST[stuck + shift gates]
    G[build setpoint graph in comm range]
    NN[Setpoint GATv2 forward]
    H[pred_to_global_setpoints + goal blend]
    APF{ENABLE_DECENTRAL_APF?}
    CL[safety_clamp]
    SP[set_setpoint + step]
    R --> MSG --> ST --> G --> NN --> H --> APF --> CL --> SP
  end

  t0 --> step
```

| Property | Decentral |
|----------|-----------|
| Assignment | Bertsekas + **local negotiator** GNN (not Hungarian) |
| Control | **GNN** each step + pull toward assigned slot (`SETPOINT_GOAL_GAIN`) |
| Sensing | **16-ray planar lidar** + graph features (training parity) |
| Messages | `StateMsg` + `ShiftProposal` within **10 m** |
| Optional | Runtime APF layer (`ENABLE_DECENTRAL_APF = True`) |

**Code:** [`sim_recorder_decentral.py`](backend/sim_recorder_decentral.py), [`comm_orchestrator.py`](backend/agents/comm_orchestrator.py), [`website_gnn/`](backend/website_gnn/)

---

## Slide 5 — How drones “talk” (comm radius)

```mermaid
flowchart TB
  subgraph swarm [Top-down view — COMM_RADIUS = 10 m]
    L((leader 0))
    A((drone A))
    B((drone B))
    C((drone C))
    L ---|in range| A
    L ---|in range| B
    A ---|in range| B
    C x---|out of range| L
  end
```

```mermaid
sequenceDiagram
  participant Sim as Python_simulator_loop
  participant i as Drone_i
  participant j as Drone_j

  Sim->>i: prepare_outbox pose vel alert stuck
  Sim->>j: prepare_outbox
  Note over i,j: deliver iff distance ≤ 10 m
  Sim->>i: receive from j inbox
  Sim->>j: receive from i inbox
  Sim->>i: GNN graph edges from comm pairs only
```

**Inbox contents:** neighbor state + optional shift proposal when sender is locally stuck.

---

## Slide 6 — Faithfulness (what is honest vs orchestrated)

| Aspect | Faithful to decentral design | Orchestrated in this backend |
|--------|------------------------------|------------------------------|
| Who hears whom | Only pairs ≤ **COMM_RADIUS** | Simulator knows all positions to test range |
| GNN graph | Edges only between comm neighbors | One batched forward over full swarm |
| Slot visibility | `SLOT_VISIBILITY_RADIUS` (10 m) in training design | `assign_swarm` returns a global map at t=0 |
| Physics | Per-drone PyFlyt setpoints | **Single** shared PyBullet world |
| Alerts | `planar_lidar` on active drones | Same formula as training frame builder |

**Presenter line:**  
*“We run decentralized **algorithms** inside a centralized **simulator** — correct for theory and A/B tests, not a deployed radio network.”*

```mermaid
flowchart LR
  subgraph faithful [Algorithm-faithful]
    F1[comm-range messaging]
    F2[local lidar features]
    F3[learned assignment + setpoint]
  end

  subgraph orch [Implementation-orchestrated]
    O1[one Python process]
    O2[global position oracle for range test]
    O3[batched GNN inference]
  end
```

---

## Slide 7 — Central vs Decentral (comparison slide)

| | **Central** | **Decentral** |
|---|-------------|----------------|
| **Assignment @ t=0** | Hungarian (optimal sum of distances) | Bertsekas + negotiator GNN |
| **Every step** | Slot + APF + clamp | Messages → GNN → hybrid goal + optional APF |
| **Uses checkpoints** | No | Yes (4 files under `resources/checkpoints/`) |
| **Comm** | — | 10 m radius |
| **Convergence** | `max_assigned_slot_error` < **0.35 m** × **25** steps | Same metric |
| **Step cap** | `MAX_STEPS_CONVERGED` = **6000** when `RUN_UNTIL_CONVERGED` | Same |

```mermaid
flowchart TB
  IC[Identical ICs + obstacles]
  IC --> CEN[Central: plan once → APF loop]
  IC --> DEC[Decentral: assign once → comm + GNN loop]
  CEN --> M[Compare trajectories + converged flags]
  DEC --> M
```

**Fairness note:** Decentral reuses **central** `obs_cfg` so obstacle layout does not favor either controller.

---

## Slide 8 — Control stack under the hood

```mermaid
flowchart TB
  subgraph pyflyt [PyFlyt supervisor]
    SET[setpoint xyz + yaw]
    STEP[step 1/240 s]
  end

  subgraph pybullet [PyBullet monitor]
    POS[read position / euler / velocity]
    COL[collision with 1 m spheres]
  end

  SET --> STEP --> POS --> COL
  POS -->|next setpoint| SET
```

| Knob (`viz_sim_config.py`) | Typical role |
|----------------------------|--------------|
| `SETPOINT_PRED_GAIN` / `SETPOINT_GOAL_GAIN` | GNN vs slot pull (0.65 / 0.35) |
| `CENTRAL_APF_*` / `DECENTRAL_APF_*` | Obstacle repulsion strength |
| `SLOT_SHIFT_DELTA_Z` | +2 m escape when stuck |
| `RECORD_EVERY` | Log every 2 physics steps |

---

## Slide 9 — Models (decentral only)

```mermaid
flowchart LR
  subgraph assign [t = 0]
    N1[strict_local_negotiator]
    N2[bertsekas_digits]
    N1 --> N2
    N2 --> MAP[drone → slot index]
  end

  subgraph ctrl [each step]
    LID[16-ray lidar + 41D frame]
    GR[comm-radius graph]
    GAT[SetpointGATv2]
    OUT[global setpoint command]
    LID --> GR --> GAT --> OUT
  end

  MAP --> ctrl
```

Checkpoints: `resources/checkpoints/` (required before `run_dual`).

Training alignment: feature builders and APF teacher live in `merged_work/models_creation/`.

---

## Slide 10 — Key files (backend only)

| Topic | Path |
|-------|------|
| Dual run | `backend/dual_simulator.py` |
| Central episode | `backend/sim_recorder_central.py` |
| Decentral episode | `backend/sim_recorder_decentral.py` |
| Comm + agents | `backend/agents/comm_orchestrator.py`, `drone_agent.py` |
| GNN inference | `backend/website_gnn/assignment_runner.py`, `setpoint_runner.py` |
| APF / shift / clamp | `backend/slot_shift_safety.py` |
| Convergence helper | `backend/sim_frame_utils.py` |
| Viz knobs | `backend/viz_sim_config.py` |

---

## Presenter cheat sheet (30 s each)

1. **Dual sim** — one seed, two trajectories; only assignment + control differ.  
2. **Central** — Hungarian then classical APF to slots; no learning.  
3. **Decentral** — learned assign + per-step GNN; comm range limits who affects the graph.  
4. **Faithful** — range, lidar, and policies match the research story; **orchestrated** — one process, batched inference.  
5. **Done** — converged means every drone held **0.35 m** from its **assigned** slot for **25** steps, not merely “animation ended.”

---

*Backend-only deck for `Website/Theoritical`. Full-stack version: [PRESENTATION.md](PRESENTATION.md). Run commands: [README.md](README.md).*
