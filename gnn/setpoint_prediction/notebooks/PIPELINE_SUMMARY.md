# Setpoint Prediction — Technical Pipeline Summary

> **Author**: Setpoint Prediction Team  
> **Date**: April 2026  
> **Notebook**: `gnn/setpoint_prediction/notebooks/01_dataloader_feature_engineering.ipynb`  
> **Reference**: *Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks* (Tolstaya et al., arXiv:1903.10527)



## The Three-Level Stack

```
┌──────────────────────────────┐
│  Friend's GNN (Strategy)     │  "Drone i → Slot j"  (formation assignment)
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  YOUR GNN (Navigator)        │  "Fly 2m left, 1m up"  (setpoint prediction)
│  ← THIS NOTEBOOK             │  Input: local state + neighbor edges
└──────────┬───────────────────┘  Output: [Δx, Δy, Δz, Δyaw]
           ▼
┌──────────────────────────────┐
│  PID Controller (Pilot)      │  Converts setpoints → motor PWM → drone flies
│  Already built in PyFlyt     │  Handles physics at 250+ Hz
└──────────────────────────────┘
```

### Why setpoints instead of PWM?
- **Paper-aligned**: arXiv:1903.10527 predicts velocity/position setpoints
- **Robust**: PID handles stability at high frequency; GNN runs at 5-10Hz
- **Generalizable**: setpoints work on any drone; PWM is hardware-specific
- **Learnable**: GNN only needs to learn geometry & coordination, not physics

---

### The Data Contract

```
INPUT (what each drone senses locally):
  ├── velocity         [3]   local linear velocity (vx, vy, vz)
  ├── angular velocity [3]   local angular velocity (ωx, ωy, ωz)
  ├── formation type   [3]   one-hot [A, rectangle, triangle]
  └── Total: 9 features per drone

  + edge_index  [2, E]   communication graph (who talks to whom)
  + edge_attr   [E, 4]   relative 3D position + Euclidean distance

OUTPUT / LABEL (what the GNN predicts):
  └── Setpoint error  [4]   body-frame [Δx, Δy, Δz, Δyaw]
                             ↑ imitating the expert PID's target
                             Z-score normalized during training
                             Denormalized at inference → fed to PID

LOSS: MSE(predicted_setpoint_norm, expert_setpoint_norm)
```

---


---

## 1. System Architecture — The Three-Level Stack

Our decentralized drone swarm controller is decomposed into three hierarchical layers, each operating at a different abstraction level and frequency:

```
┌─────────────────────────────────────────────────────────────┐
│  Level 3 — Strategy (Formation Assignment GNN)              │
│  Frequency: Once per formation change                       │
│  Owner: Collaborator's model                                │
│                                                             │
│  Task: Solve a discrete assignment problem.                 │
│  "Drone i is assigned to formation slot j."                 │
│  This is a combinatorial decision — which drone goes where. │
└────────────────────────┬────────────────────────────────────┘
                         │  slot assignment
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Level 2 — Navigator (Setpoint Prediction GNN) ◄ THIS WORK │
│  Frequency: 5–10 Hz                                         │
│  Owner: This notebook and future model                      │
│                                                             │
│  Task: Predict continuous body-frame setpoints.              │
│  "Fly 2.1m left, 0.5m forward, 0.3m up, turn 7°."          │
│  Uses only LOCAL sensing + neighbor communication.           │
│  Learns geometry, coordination, and collision avoidance      │
│  through GNN message passing — no global state required.     │
└────────────────────────┬────────────────────────────────────┘
                         │  [Δx, Δy, Δz, Δyaw] setpoint
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Level 1 — Pilot (PID Controller)                           │
│  Frequency: 250–1000 Hz                                     │
│  Owner: PyFlyt / PyBullet                                   │
│                                                             │
│  Task: Convert position/velocity setpoints into motor PWM.  │
│  Handles gravity compensation, torque, aerodynamics.         │
│  Battle-tested, runs at hardware frequency.                  │
└─────────────────────────────────────────────────────────────┘
```

**Why this decomposition matters:** The GNN at Level 2 only needs to learn *spatial coordination* — deciding where each drone should fly based on its neighbors. The hard physics of flight (gravity, motor dynamics, aerodynamic drag) is delegated entirely to the PID at Level 1. This separation is what makes the system trainable with only ~200 episodes and robust enough to fly in simulation.

---

## 2. Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Graph Framework** | PyTorch Geometric (PyG) | Graph data structures, batching, DataLoader |
| **Data Storage** | `InMemoryDataset` + `.pt` files | Pre-collated train/val/test splits |
| **Normalization** | Custom `SetpointNormalizer` class | Feature engineering, serialization to JSON |
| **Simulation** | PyFlyt (PyBullet-based) | Data generation via expert PID |
| **Visualization** | Matplotlib | Feature distributions, graph topology |
| **Model (planned)** | GATv2Conv | Attention-based message passing with edge features |
| **Training (planned)** | PyTorch + MSE Loss | Node-level regression |

---

## 3. Pipeline Breakdown

### 3.1 Switching the Label to Setpoint Error

**What we did:** The label `y` is the expert PID controller's body-frame setpoint error `[Δx, Δy, Δz, Δyaw]`, stored in the dataset's `target` field. We discarded the motor PWM field (`y` in raw data) entirely.

**Why:** Predicting motor PWM would require the GNN to implicitly learn the full flight dynamics of a quadrotor — gravity compensation, rotational torque, and aerodynamic effects. This is:
- **Data-hungry**: 200 episodes is insufficient to cover the motor dynamics state space.
- **Fragile**: PWM control operates at 250+ Hz. Any latency or prediction jitter causes oscillation and crashes.
- **Non-transferable**: PWM values are specific to motor constants, battery voltage, and airframe weight. A model trained on one drone configuration would fail on another.

Setpoint prediction sidesteps all of this. The GNN learns only spatial reasoning ("where should I be?"), while the PID — a mature, well-tuned controller — handles the physics of getting there.

**How it helps the GNN learn:** The setpoint error has clear geometric structure. Drones that are far from their target have large errors; drones near their target have small errors. This smooth, spatially coherent signal is exactly what GNN message passing is designed to propagate and reason about.

---

### 3.2 Removing Target Error from Input (Data Leakage Prevention)

**What we did:** The input feature vector `x` contains only the drone's local sensor state — **9 dimensions**: linear velocity (3), angular velocity (3), and formation type one-hot (3). The target error is *not* included in the input.

**Why:** If the target error were present in both the input and the label, the GNN could learn a trivial identity mapping (`output ≈ input[9:13]`), achieving near-zero training loss while learning nothing about coordination. This is a textbook case of **data leakage**.

The entire value of the GNN lies in its ability to reconstruct the missing spatial information through **multi-hop message passing**. By withholding the target from the input, we force the network to:
1. Aggregate neighbor positions from `edge_attr` (relative 3D coordinates).
2. Propagate spatial context across 2–3 hops of the communication graph.
3. Infer its own role within the formation from the collective arrangement of its neighbors and the formation type embedding.

**How it helps the GNN learn:** This design constraint is what makes the controller *decentralized*. Each drone computes its setpoint using only local sensing and neighbor communication — no GPS, no centralized coordinator, no explicit target broadcast. This aligns precisely with the methodology described in arXiv:1903.10527.

---

### 3.3 Adding Euclidean Distance to Edge Features

**What we did:** We augmented the raw 3D relative position vector `[Δx, Δy, Δz]` with a fourth scalar: the Euclidean distance `d = √(Δx² + Δy² + Δz²)`. The edge feature dimension went from 3 to 4.

**Why:** While the distance is mathematically derivable from the three spatial components, providing it explicitly as a pre-computed feature is a well-known practice in geometric deep learning. The key insight relates to how **GATv2 attention** works:

The attention mechanism computes scalar attention weights for each edge. Learning to threshold or gate attention based on neighbor distance (e.g., "ignore drones beyond 5 metres") requires the network to internally compose a non-linear function `√(x² + y² + z²)` from three separate inputs. By providing the distance directly, we reduce the burden on the attention heads and enable faster, more reliable learning of distance-dependent communication patterns.

**How it helps the GNN learn:** In swarm scenarios, a drone's nearest neighbors carry far more relevant information than distant ones. The explicit distance feature allows the attention mechanism to learn sharp distance-gating behaviour in fewer training epochs, improving both sample efficiency and final performance.

---

### 3.4 One-Hot Formation ID Preservation

**What we did:** The formation type is represented as a one-hot vector `[form_a, form_rect, form_tri]` at input indices 6–8. During normalization, we explicitly override these columns with identity statistics (`mean=0, std=1`), ensuring the values remain exactly `{0, 1}` after the Z-score transform.

```python
# In SetpointNormalizer.fit():
xm[-cls.N_ONEHOT:] = 0.0   # (val - 0) / 1 = val
xs[-cls.N_ONEHOT:] = 1.0   # unchanged
```

**Why:** One-hot encodings function as **categorical switches** within the network. A GATv2 layer can learn distinct weight modulations conditioned on these binary flags — effectively learning separate behaviours for each formation type. If these values were Z-score transformed (e.g., `form_tri` becoming 1.73 instead of 1.0 and `form_a` becoming -0.76 instead of 0.0), the categorical semantics would be destroyed and the network would need to waste capacity re-learning the category boundaries.

**How it helps the GNN learn:** Clean `{0, 1}` signals act as direct conditional switches. The network can learn: "If `form_tri = 1`, arrange in a triangular pattern" without first having to decode a continuous value back into a categorical decision.

---

### 3.5 Z-Score Normalization of Setpoint Labels

**What we did:** The setpoint labels are Z-score normalized using training set statistics: `y_norm = (target - μ) / σ`. The statistics are:

| Label | Mean (μ) | Std (σ) | Raw Range |
|-------|---------|---------|-----------|
| Δx | 0.0069 | 2.3725 | [-12.8, +14.4] metres |
| Δy | -0.0157 | 2.3735 | [-14.9, +12.1] metres |
| Δz | 0.3846 | 0.8292 | [-2.9, +4.7] metres |
| Δyaw | 0.0056 | 0.6601 | [-3.1, +3.1] radians |

**Why:** Without normalization, the MSE loss would be dominated by Δx and Δy errors (which can reach ±15 metres), while effectively ignoring Δz and Δyaw contributions (which are an order of magnitude smaller). This imbalanced gradient signal would cause the network to:
- Over-fit to horizontal positioning while under-fitting altitude and heading.
- Produce unstable training curves with large gradient variance.

Z-score normalization ensures all four label dimensions contribute equally to the loss and gradient computation.

**How it helps the GNN learn:** Balanced gradients across all output dimensions lead to faster convergence and better final performance on all four setpoint components. The model learns altitude and heading control with the same priority as horizontal positioning.

> **Critical inference note:** The model outputs normalized setpoints. Before feeding predictions to the PID controller, they **must** be denormalized: `setpoint_real = pred * σ + μ`. The `denormalize_target()` method in `SetpointNormalizer` handles this.

---

## 4. Future Integration — Inference Pipeline

At inference time, the trained GNN integrates into the swarm control loop as follows:

```python
# ── Load normalizer (computed during training, never re-fitted) ──
normalizer = SetpointNormalizer.load("results/normalizer.json")

# ── Per-timestep control loop (5-10 Hz) ──
for timestep in simulation:
    # 1. Build graph from current drone states
    graph = build_graph_from_sensors(drones, comm_radius=10.0)
    
    # 2. Normalize input features
    graph_norm = normalizer.transform_graph(graph)
    
    # 3. GNN forward pass → normalized setpoint
    pred_norm = model(graph_norm.x, graph_norm.edge_index, graph_norm.edge_attr)
    
    # 4. Denormalize → physical setpoints in metres and radians
    setpoint_real = normalizer.denormalize_target(pred_norm)
    #   setpoint_real[i] = [Δx_metres, Δy_metres, Δz_metres, Δyaw_radians]
    
    # 5. Feed to PID controller (handles motor physics at 250+ Hz)
    for i, drone in enumerate(drones):
        drone.pid_controller.set_target(setpoint_real[i])
```

The `normalizer.json` file is the bridge between the training notebook and the inference deployment. It encodes the exact statistics computed from the training set, ensuring that the feature scaling at inference matches what the model was trained on. Changing or re-computing these statistics would invalidate the trained model weights.

---

## 5. Final Data Contract

```
INPUT          [N, 9]    vel_xyz(3) + ang_xyz(3) + formation_onehot(3)
EDGE_ATTR      [E, 4]    rel_xyz(3) + euclidean_distance(1)
EDGE_INDEX     [2, E]    communication graph (radius = 10m)

LABEL          [N, 4]    setpoint_error [Δx, Δy, Δz, Δyaw] (z-scored)

TRAIN GRAPHS   48,013    (~700K total node predictions per epoch)
VAL GRAPHS      6,099
TEST GRAPHS     6,362
DRONES/GRAPH   10–20     (variable — GNN is size-agnostic)
```

---

*This pipeline positions the GNN to learn what it does best — relational reasoning and spatial coordination — while delegating low-level flight physics to the PID controller where it belongs.*
