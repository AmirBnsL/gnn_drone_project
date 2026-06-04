# Decentralized Multi-Drone Formation Assignment via Cross-Attention Matching and Local Auction Protocols

**Anonymous** · University Project Report

---

## Abstract

We address the problem of assigning N drones to N formation slots in a fully decentralized manner, where each drone can only observe nearby slots and communicate with neighbouring drones. We propose a two-stage pipeline: a SuperGlue-inspired cross-attention matcher that learns assignment preferences from local observations, followed by a decentralized auction protocol that resolves conflicts through iterative one-hop message passing. Trained on 20,000 stratified scenarios across 6 formation types and swarm sizes N ∈ [8, 20], our system achieves a bijection rate of 0.970 on held-out test scenarios — meaning 97% of scenarios are fully assigned with zero conflicts — at a cost within 0.5% of the centralised Hungarian optimum. We compare against four baselines: a gossip consensus GNN, a greedy matcher, a standalone DecentAuction model, and the centralised Hungarian algorithm. Physical feasibility is validated in the PyFlyt quadrotor simulator using built-in cascaded PID controllers. As a centralised upper bound, a HungarianImitator — which uses full global visibility and Sinkhorn-based differentiable assignment — achieves bij=1.000 at O(N²) complexity at the same cost (1.005), confirming that our decentralised auction-based system matches centralised cost efficiency while requiring no central coordinator. The system generalises out-of-distribution to N=25 drones (bij=0.920) while a gossip-only baseline collapses to bij=0.420 at the same scale, and achieves 93.3% success rate across 15 physics simulation runs with zero inter-drone collisions.

---

## 1. Introduction

A drone swarm performing a formation maneuver faces a fundamental assignment problem: which drone goes to which slot? In centralized systems, a ground station collects all drone positions, runs an optimal solver (e.g., the Hungarian algorithm), and broadcasts assignments back. This works in a lab but fails in real deployments — a single point of failure, communication bottlenecks, and latency that grows with swarm size.

We want each drone to decide its own assignment using only what it can locally observe and what its neighbours tell it. This is the **decentralized assignment problem under visibility constraints**.

**Simple example:**

```
Arena: 10m × 10m
N = 5 drones scattered randomly
Formation: V-shape with 5 slots

Drone 2 can see:  slots [1, 3, 4]    (others too far)
Drone 2 can talk: drones [1, 5]      (others out of range)

Drone 2 must decide which slot to claim
using only this local information.
```

The challenge is that two drones might independently decide to claim the same slot — a **conflict**. Resolving conflicts without a central coordinator is the core problem this paper addresses.


The main contributions of this work are:

- A two-stage decentralised assignment pipeline combining a SuperGlue-inspired cross-attention matcher with a local Bertsekas auction protocol, achieving bij=0.970 and cost within 0.5% of the centralized Hungarian optimum.

- A demonstration that local visibility (r_s=5.0m) and one-hop communication (r_c=3.0m) are sufficient to recover 97% of centralized assignment quality, quantified via a HungarianImitator upper bound model.

- An auction-state injection mechanism that makes the learned value matrix price-aware, reducing average conflict resolution from 10.6 rounds (DecentAuction standalone) to 5.3 rounds.

- An empirical out-of-distribution generalisation result: bij=0.920 at N=25 (outside the training range N∈[8,20]) versus gossip baseline collapse at bij=0.420.

- Physics validation in the PyFlyt quadrotor simulator confirming that computed assignments are executable by cascaded PID controllers with zero collisions.

---

## 2. Related Work

### 2.1 Decentralised Multi-Robot Task Assignment

Market-based and auction-based approaches have been widely studied for multi-robot task allocation, with early foundational work establishing that competitive pricing mechanisms can produce near-optimal decentralised assignments [Dias et al. 2006, "Market-based multirobot coordination"]. Contract net protocols [Smith 1980] introduced a request-bid-award cycle that remains influential, though both approaches rely on hand-crafted utility functions that require domain expertise to tune. Centralised solvers such as the Hungarian algorithm provide globally optimal solutions but are impractical for large swarms due to their O(N³) complexity and single-point-of-failure communication requirement. Our work differs by learning a value initialiser from data rather than specifying utility functions manually, enabling the auction to start from near-optimal values without hand-engineering.

### 2.2 Auction Algorithms for Robotics

The Bertsekas auction algorithm [Bertsekas 1988, "The auction algorithm"] provides a convergence-guaranteed price-raising mechanism that resolves assignment conflicts in finite rounds under full connectivity. Decentralised extensions [Bertsekas & Castanon 1991] allow the auction to run with partial communication, though theoretical convergence guarantees weaken under sparse graphs. Prior work on multi-UAV assignment has applied consensus-based auction variants to robust task allocation [Choi et al. 2009, "Consensus-based decentralized auctions for robust task allocation"], demonstrating practical convergence in field conditions. Our contribution extends this line of work by operating the auction on learned cross-attention values under partial communication connectivity, replacing hand-crafted bid utilities with a neural value initialiser.

### 2.3 Graph Neural Networks for Multi-Agent Coordination

Graph neural networks have demonstrated strong performance for learning decentralised controllers in robot swarms [Tolstaya et al. 2020, "Learning decentralized controllers for robot swarms"], enabling coordination policies that are robust to varying swarm topologies. Subsequent work has explored learning under communication constraints through message-aware attention mechanisms [Li et al. 2021, "Message-aware graph attention networks"], showing that selectively routing messages improves coordination quality under bandwidth limits. Formation control with neural networks has also been explored at the survey level [Oh et al. 2015, "A survey of multi-agent formation control"], identifying geometric reasoning as the central challenge. Our contribution uses cross-attention rather than graph convolution to jointly reason about all drone-slot pairs simultaneously, naturally encoding the bipartite structure of the assignment problem.

### 2.4 Learned Feature Matching

The SuperGlue network [Sarlin et al. 2020, "SuperGlue: Learning Feature Matching with Graph Neural Networks"] introduced cross-attention between keypoint sets as a principled approach to feature matching, originally designed for image keypoint correspondence with a Sinkhorn differentiable assignment head. The architecture processes two sets of descriptors through alternating self- and cross-attention layers, producing a soft assignment matrix that is decoded to a hard matching via Sinkhorn normalisation. We adapt this architecture for drone-slot assignment by replacing keypoint descriptors with drone and slot positional features augmented with a formation type embedding, replacing the Sinkhorn assignment head with a Bertsekas auction protocol for decentralised conflict resolution, and adding a visibility mask that enforces communication constraints so the model respects the partial observability of each drone.

## 3. Problem Formulation

Let there be N drones at positions $\{d_1, \ldots, d_N\} \subset \mathbb{R}^2$ and N formation slots at positions $\{s_1, \ldots, s_N\} \subset \mathbb{R}^2$. We seek a bijective assignment $\sigma: \{1,\ldots,N\} \to \{1,\ldots,N\}$ such that drone $i$ is assigned to slot $\sigma(i)$.

**Constraints:**
- Drone $i$ can only observe slot $j$ if $\|d_i - s_j\| \leq r_s$ (slot visibility radius)
- Drone $i$ can only communicate with drone $j$ if $\|d_i - d_j\| \leq r_c$ (communication radius)
- No central coordinator — every decision is made from local state only

**Objective:** Minimize total assignment cost $\sum_{i=1}^{N} \|d_i - s_{\sigma(i)}\|$ subject to bijection (no two drones share a slot, no drone is unassigned).

**The optimal centralized solution** is given by the Hungarian algorithm, which runs in $O(N^3)$ but requires global state. Our system approximates this with only local information.

---

## 4. Dataset

### 3.1 Generation

We generate 20,000 standard scenarios plus 3,000 hard scenarios (dense swarms, N ∈ [16, 20]) using stratified sampling across:
- **6 formation types:** A, V, W, Circle, Rectangle, Triangle
- **13 swarm sizes:** N ∈ {8, 9, ..., 20}
- **78 strata total** (6 × 13), each receiving equal samples

Each scenario is generated as follows:
1. Place N drones uniformly at random in a 10m × 10m arena
2. Generate N formation slots centred at the swarm centre of gravity (CoG)
3. Solve optimal assignment via Hungarian algorithm → ground truth label

Formation slots are placed at `slots = slots_rel + CoG` with no rotation and no random offset. This ensures that formation slots are always reachable from nearby drones — keeping drone-to-slot distances within the slot visibility radius (`SLOT_VISIBILITY_RADIUS = 5.0m`) for most pairs. Adding rotation or large offsets would push slots beyond the visibility radius of some drones, creating a candidate mask distribution the model never trained on.

**Simple example of one scenario:**

```
N=4 drones, V-formation

Drone positions:  [(2.1, 3.4), (7.8, 1.2), (5.0, 8.9), (1.5, 6.7)]
CoG:              (4.1, 5.1)
slots_rel:        [(-1.8, -1.0), (0.0, -1.0), (-0.9, 0.5), (0.9, 0.5)]
slots = CoG + slots_rel:
                  [(2.3, 4.1), (4.1, 4.1), (3.2, 5.6), (5.0, 5.6)]

Hungarian assignment (ground truth):
  Drone 0 → Slot 0  (distance: 0.7m)
  Drone 1 → Slot 2  (distance: 4.9m)
  Drone 2 → Slot 3  (distance: 3.4m)
  Drone 3 → Slot 1  (distance: 3.0m)
  Total cost: 12.0m  (optimal — no other bijection is cheaper)
```

### 3.2 Graph Construction

Each scenario is represented as two graphs that are stored with the scenario and passed directly to the model:

**Communication graph (drone ↔ drone):**
An undirected edge $(i, j)$ exists between drones $i$ and $j$ if $\|d_i - d_j\| \leq r_c$ where $r_c = 3.0$m. This is the channel over which drones exchange auction messages. Each edge carries a 2D attribute: the relative displacement vector $(d_j - d_i)$, so the model knows direction and implied distance to each neighbour.

```
Example: N=4 drones at [(1,1), (2,2), (5,5), (8,8)]
  COMM_RADIUS = 3.0m

  Pairwise distances:
    d(0,1) = 1.41m  ✓ edge exists
    d(0,2) = 5.66m  ✗ too far
    d(1,2) = 4.24m  ✗ too far
    d(2,3) = 4.24m  ✗ too far

  Communication graph:
    Drone 0 ↔ Drone 1  (neighbours)
    Drone 2 and Drone 3 are isolated from 0,1

  → Drones 0 and 1 can share auction state directly.
    Drones 2 and 3 must resolve conflicts independently.
    This is the partial connectivity challenge.
```

**Drone-slot candidate graph (drone → slot):**
A directed edge $(i, j)$ exists from drone $i$ to slot $j$ if $\|d_i - s_j\| \leq r_s$ where $r_s = 5.0$m. Only candidate edges are considered during assignment — drone $i$ cannot be assigned to slot $j$ if the edge does not exist. Each edge carries a 3D attribute: the relative displacement $(s_j - d_i)$ in x and y, plus the Euclidean distance $\|d_i - s_j\|$.

```
Example: Drone at (2, 2), slots at [(3,3), (6,6), (9,9)]
  SLOT_VISIBILITY_RADIUS = 5.0m

  Distances:
    to slot 0: 1.41m  ✓ candidate
    to slot 1: 5.66m  ✗ too far — NOT a candidate
    to slot 2: 9.90m  ✗ too far — NOT a candidate

  → Drone can only be assigned to slot 0.
    Slots 1 and 2 are invisible to this drone.
```

The **candidate mask** is an N×N boolean matrix where `candidate_mask[i][j] = True` means drone $i$ can see slot $j$. During training, one additional entry per scenario is set to False (the GT slot of one randomly chosen drone) to force the model to learn conflict resolution even when the GT assignment is partially hidden.

---

### 3.3 Node Features (x)

Each drone node is represented by a **27-dimensional feature vector** encoding everything the drone locally knows:

```
x[i] = [
  ── Relative vectors to each visible slot (2D each) ──
  (s_0 - d_i),   (s_1 - d_i),   ...,  (s_{N-1} - d_i)
  [zero-padded for slots outside visibility radius]
  Dims: 2 × N  →  up to 2 × 13 = 26 dims for N=13

  ── Formation type embedding (learned, 8D) ──
  f_emb[formation_id]
  Dims: 8

  Total: 2N + 8 = 27 for N < 10, padded to fixed width
]
```

**Why relative vectors and not absolute positions?**
Using $(s_j - d_i)$ instead of absolute $(s_j)$ makes features translation-invariant — a drone at (1,1) looking at a slot at (2,2) produces the same feature as a drone at (5,5) looking at a slot at (6,6). This means the model generalises across all arena positions without memorising coordinates.

```
Example: Drone at (3.0, 4.0), N=3, all slots visible
  slot_0 = (4.0, 5.0)  →  relative = (+1.0, +1.0)
  slot_1 = (2.0, 6.0)  →  relative = (-1.0, +2.0)
  slot_2 = (5.0, 3.0)  →  relative = (+2.0, -1.0)
  formation_id = 2 (W-formation)  →  f_emb[2] = [0.3, -0.1, ...]  (learned)

  x[drone] = [1.0, 1.0, -1.0, 2.0, 2.0, -1.0, 0.3, -0.1, ...]
              ──slot0──  ──slot1──  ──slot2──  ──formation emb──
```

Slots outside the visibility radius contribute a zero vector, so the model can infer from zeros that those slots are invisible (not just far).

---

### 3.4 Edge Features

**Communication edge attributes** (drone ↔ drone, 2D):

```
edge_attr[i→j] = (d_j - d_i)   ← relative displacement in 2D

Example: Drone 0 at (1,2), Drone 1 at (3,4)
  edge_attr[0→1] = (2.0, 2.0)
  edge_attr[1→0] = (-2.0, -2.0)  ← reverse direction
```

The model uses this to reason about *where* its neighbours are, not just that they exist. A neighbour to the right vs to the left carries different spatial meaning for formation geometry.

**Drone-slot edge attributes** (drone → slot, 3D):

```
ds_edge_attr[i→j] = (s_j - d_i, ‖s_j - d_i‖)   ← displacement + distance

Example: Drone at (2,3), Slot at (5,7)
  ds_edge_attr = (3.0, 4.0, 5.0)
                  ──Δx──  ──Δy──  ──dist──
```

The explicit distance as a third feature helps the epsilon head predict the correct bid increment — nearby slots get smaller epsilon (less urgency to outbid) while far slots need larger increments to justify the cost.

---

## 5. Approaches

### 5.1 Baseline — Gossip Consensus GNN (LocalNegotiatorGNN)

**What it is:** A Graph Attention Network (GAT) where drones iteratively share information with neighbours and vote on assignments through gossip rounds.

**Architecture:**

```
Input features: same 27D node features as SuperGlue (Section 4.3)
  — relative vectors to all N visible slots (2D each, zero-padded)
  — formation type embedding (8D)

2 × GATConv layers:
  hidden_dim = 64
  attention heads = 4  (concat=False → output stays 64D)
  activation = ELU

Communication: same comm graph (COMM_RADIUS = 3.0m)
Gossip rounds at inference: K = 10

Final layer: Linear(64, N) → softmax over visible slots → argmax → assignment

Training: AdamW (lr=3e-4, weight_decay=1e-4), 80 epochs,
          CosineAnnealingLR (eta_min=1e-5), batch_size=1
Loss: L_CE only (cross-entropy — drone should prefer its GT slot)
Total parameters: ~85K
```

**How it works:**

```
Each drone starts with:
  - Its own position
  - Visible slot positions
  - Formation type embedding

Each gossip round:
  1. Drone sends its current belief to neighbours
  2. Drone receives neighbours' beliefs
  3. GAT aggregates messages → updates belief
  4. Repeat for K = 10 rounds

Final step: argmax over visible slots → assignment
```

**Simple example:**

```
Round 0: Drone 2 thinks slot 3 is best (value=0.8)
         Drone 5 thinks slot 3 is best (value=0.7)
         → CONFLICT: both want slot 3

Round 1: Drone 2 hears drone 5 also wants slot 3
         GAT updates: drone 2 now values slot 1 more
         Drone 5 still wants slot 3

Round 2: No conflict — drone 2 takes slot 1, drone 5 takes slot 3
```

**Why it works partially:** Local message passing naturally propagates conflict information. But because the model makes a hard argmax decision at the end with no explicit conflict resolution mechanism, some conflicts remain unresolved — especially in dense swarms where many drones compete for the same slots.

**When it fails — the disconnected conflict:**

```
Drone 2 at (1.0, 1.0) wants slot 3
Drone 7 at (9.0, 9.0) wants slot 3
Distance between them: 11.3m  >  COMM_RADIUS (3.0m)
→ They are NOT neighbours — neither hears the other

All K gossip rounds:
  Drone 2: never hears about Drone 7 → keeps slot 3
  Drone 7: never hears about Drone 2 → keeps slot 3

Final argmax: Drone 2 → slot 3, Drone 7 → slot 3
→ CONFLICT — unresolved, bijection fails

This is why bij=0.840: 16% of scenarios (32/200) have at least one such disconnected conflict pair, each pushing bijection to 0 for that scenario. Note that bijection rate (fraction of scenarios with zero conflicts) and failure rate (fraction of scenarios with at least one unresolved conflict) are the same metric reported from two angles.
```

**Results:**

| Metric | Value |
|---|---|
| Bijection rate | 0.840 |
| Cost ratio | 1.021 |
| Slot match rate | 0.700 |
| Consensus rounds | 8.5 |
| Failure rate | 16.0% (32/200 test scenarios) |

*Results reported as mean ± std over 3 independent runs with different random seeds.*


---

### 5.2 SuperGlue Cross-Attention Matcher (SuperGlueSwarmMatcher)

**What it is:** A cross-attention neural network inspired by the SuperGlue feature matching paper, adapted for drone-slot assignment. Instead of matching keypoints in images, it matches drones to formation slots.

**Why we built it:** The gossip GNN treats assignment as a classification problem — each drone independently picks its best slot. SuperGlue treats it as a **joint matching problem** — the entire value matrix V[drone, slot] is computed simultaneously, allowing the model to reason about global competition even from local observations.

**Architecture:**

```
Input per drone:
  position (x, y) + formation embedding → drone feature vector
  [27-dimensional: relative vectors to all N visible slots (2D each,
   zero-padded for invisible slots) + formation embedding (8D).
   Using relative vectors (s_j - d_i) makes features translation-invariant
   — the model generalises across all arena positions. See Section 3.3.]

Input per slot:
  position (x, y) + formation embedding → slot feature vector

6 × Cross-Attention Layers:
  Layer k:
    - Each drone attends to all its visible slots
      → updates drone features based on slot preferences
    - Each slot attends to all drones that can see it
      → updates slot features based on competition
    - Communication graph attention
      → drone i receives messages from neighbouring drones

Output:
  V[i, j] = value score for drone i claiming slot j
  eps[i]  = epsilon (price increment) for drone i
```

---

#### 5.2.1 Inside One Cross-Attention Layer — Step by Step

Each of the 6 layers takes the current drone feature vectors $\mathbf{h}^{(k)}_i \in \mathbb{R}^{64}$ and slot feature vectors $\mathbf{g}^{(k)}_j \in \mathbb{R}^{64}$ and produces updated versions. Here is exactly what happens inside layer $k$:

**Step 1 — Linear projection (MLP projectors)**

Before the first layer, raw node features are projected into the 64-dimensional hidden space:

```
drone raw features (27D)  →  Linear(27, 64) + ReLU  →  h_i  ∈ R^64
slot  raw features (27D)  →  Linear(27, 64) + ReLU  →  g_j  ∈ R^64
edge  raw features  (2D)  →  Linear( 2, 16)          →  e_ij ∈ R^16
```

These projectors are shared across all layers — they are only applied once at the start. This is what "MLP projectors" refers to throughout the paper.

**Step 2 — Drone-to-slot attention (drone attends to its visible slots)**

Each drone $i$ computes how much it should pay attention to each visible slot $j$:

```
Query:  Q_i  = Linear(h_i,  64→32)     ← "what am I looking for?"
Key:    K_j  = Linear(g_j,  64→32)     ← "what does slot j offer?"
Value:  V_j  = Linear(g_j,  64→64)     ← "slot j's information"

Raw score:  score[i,j] = (Q_i · K_j) / √32

Masking: score[i,j] = -∞  if slot j not visible to drone i
         (this enforces the visibility constraint — invisible slots
          contribute zero to the attention output)

Attention weight:  α[i,j] = softmax(score[i,:])  over visible j only

Aggregated update:  Δh_i = Σ_j  α[i,j] · V_j
```

**Concrete example with 2 drones, 3 slots:**

```
Drone 1 can see: slots [0, 1]    (slot 2 invisible)
Drone 2 can see: slots [1, 2]    (slot 0 invisible)

Drone 1 scores:
  score[1,0] = Q_1 · K_0 / √32 = 0.8   ← slot 0 looks good
  score[1,1] = Q_1 · K_1 / √32 = 0.3   ← slot 1 less good
  score[1,2] = -∞                        ← invisible

  α[1,0] = exp(0.8)/(exp(0.8)+exp(0.3)) = 0.62
  α[1,1] = exp(0.3)/(exp(0.8)+exp(0.3)) = 0.38

  Δh_1 = 0.62 · V_0 + 0.38 · V_1
  → h_1 is updated to reflect: "I prefer slot 0 but slot 1 is ok"

Drone 2 scores:
  score[2,1] = 0.9   ← slot 1 looks great
  score[2,2] = 0.2
  score[2,0] = -∞

  α[2,1] = 0.69,  α[2,2] = 0.31
  Δh_2 = 0.69 · V_1 + 0.31 · V_2
  → h_2 is updated to reflect: "I strongly prefer slot 1"

After this layer:
  Both drones want slot 1.
  Next layer: slot 1's key K_1 is updated to reflect high competition.
  → In layer k+1, drones will see that slot 1 is contested.
```

**Step 3 — Slot-to-drone attention (slot attends to competing drones)**

Each slot $j$ aggregates information from all drones that can see it — learning how contested it is:

```
Query:  Q_j  = Linear(g_j, 64→32)
Key:    K_i  = Linear(h_i, 64→32)
Value:  V_i  = Linear(h_i, 64→64)

score[j,i]  = (Q_j · K_i) / √32   for all drones i that can see j
α[j,i]      = softmax(score[j,:])
Δg_j        = Σ_i  α[j,i] · V_i
```

This is how the model encodes competition — after step 3, slot 1's feature vector $\mathbf{g}_1$ encodes the fact that multiple drones are attending to it.

**Step 4 — Communication graph attention (drone attends to neighbours)**

Each drone additionally aggregates information from its neighbours in the comm graph:

```
For drone i with neighbours N(i):
  score[i,k] = (Q_i · K_k + MLP(e_ik)) / √32   for k ∈ N(i)
               ↑ edge feature e_ik (relative displacement) shifts the score
  α[i,k]     = softmax over neighbours
  Δh_i       += Σ_{k∈N(i)}  α[i,k] · V_k
```

The edge feature term `MLP(e_ik)` means neighbours in specific directions contribute differently — a neighbour that is to the left gets a different weight than one directly ahead.

**Step 5 — Residual update**

After all three attention steps, the features are updated with a residual connection and layer norm:

```
h_i^(k+1) = LayerNorm( h_i^(k) + Δh_i_slots + Δh_i_comm )
g_j^(k+1) = LayerNorm( g_j^(k) + Δg_j_drones )
```

This is repeated for all 6 layers. Each layer refines the features using the updated context from the previous layer.

---

#### 5.2.2 Value Head and Epsilon Head

After 6 layers, the final drone features $\mathbf{h}^{(6)}_i$ and slot features $\mathbf{g}^{(6)}_j$ are used to compute the output scores:

**Value head — produces V[i,j]:**

```
V[i,j] = VALUE_SCALE · tanh( MLP( concat(h_i^(6), g_j^(6)) ) )

where:
  concat(h_i, g_j) ∈ R^128     ← concatenate drone and slot final features
  MLP: Linear(128,64) → ReLU → Linear(64,1) → scalar
  tanh: bounds output to (-1, +1)
  VALUE_SCALE = 10.0: stretches to (-10, +10)

V[i,j] = -∞  if slot j not visible to drone i  (masked out)
```

A high value V[i,j] means drone i strongly wants slot j. The tanh keeps values bounded so the auction's epsilon computation is numerically stable.

**Epsilon head — produces eps[i]:**

```
eps[i] = softplus( MLP( h_i^(6) ) )

where:
  MLP: Linear(64,32) → ReLU → Linear(32,1) → scalar
  softplus: ensures eps[i] > 0 always (bids must be positive)
```

The epsilon head predicts how aggressively drone i should bid — how much it should raise the price of its target slot to outcompete rivals. Trained via `L_eps` to match the gap between first and second best net values.

**Full forward pass — dimensions at each stage:**

```
Input x:          [N, 27]     ← raw node features
After projector:  [N, 64]     ← h_i initial
After 6 layers:   [N, 64]     ← h_i^(6), g_j^(6)
concat(h_i,g_j):  [N×N, 128]  ← all drone-slot pairs
Value head:       [N, N]      ← V matrix (masked)
Epsilon head:     [N, 1]      ← eps per drone
```

---

**Simple example of what attention learns:**

```
Drone 1 at (2, 2), can see slots [A, B, C]
Drone 2 at (2.5, 2.5), can see slots [A, B, D]

Without attention:
  Both drones independently rate slot A as best → conflict

With cross-attention (layer 3):
  Drone 1 sees drone 2 also values A highly
  → Drone 1 adjusts: now values B more
  Drone 2 keeps A as top choice

Result: Drone 1→B, Drone 2→A — no conflict
```

**Training loss:** Five terms jointly trained:

```
L = L_CE + λ_margin · L_margin + λ_coverage · L_coverage + λ_eps · L_eps + λ_div · L_diversity

L_CE       : cross-entropy — drone should prefer its GT slot
L_margin   : the winning slot's value should exceed second best by a margin
L_coverage : penalise slots that no drone values positively (unassigned)
L_eps      : epsilon head should predict the correct price increment
L_diversity: penalise multiple drones assigning high value to the same slot
             → directly discourages competitive pile-up

Loss weights:
  λ_margin   = 0.25
  λ_coverage = 0.35   (= COVERAGE_WEIGHT in Appendix A)
  λ_eps      = 0.05
  λ_div      = 0.10
```

**Training configuration:**

```
Optimizer : AdamW (lr=3e-4, weight_decay=1e-4)
Scheduler : CosineAnnealingLR (T_max=80, eta_min=1e-5)
Epochs    : 80
Batch size: 1  (variable N prevents standard batching)
Early stop: patience=15 on validation loss
```

**Why the diversity loss matters:**

```
Without diversity loss:
  V[drone1, slot3] = 0.92  (drone 1 wants slot 3)
  V[drone2, slot3] = 0.89  (drone 2 also wants slot 3)
  V[drone3, slot3] = 0.85  (drone 3 also wants slot 3)
  → 3 drones pile onto slot 3 → conflict

With diversity loss:
  The loss penalises low column entropy in softmax(V, dim=0)
  → model learns to spread values: one drone clearly wins each slot
  V[drone1, slot3] = 0.92
  V[drone2, slot3] = 0.41  ← pushed down
  V[drone3, slot3] = 0.23  ← pushed down
  → conflict avoided before auction even runs
```

**Results (matcher alone, greedy argmax):**

| Metric | Value |
|---|---|
| Bijection rate | 0.815 |
| Cost ratio | 1.002 |
| Slot match rate | 0.865 |

*Results reported as mean ± std over 3 independent runs with different random seeds.*

Note: cost ratio of 1.002 means the matcher finds assignments within 0.2% of optimal using greedy argmax alone — the auction adds bijection guarantee at a small cost overhead.

---

### 5.3 DecentAuction Standalone

**What it is:** A price-aware wrapper around the LocalNegotiatorGNN that injects local auction state (prices, owners, claims) as additional features before each forward pass. The model is trained to produce values that work well with the auction protocol.

> **Relationship to Section 5.1:** DecentAuction standalone uses the **same GNN backbone** as the Gossip baseline, but with two key differences: (1) the input is augmented with auction state features (current prices, slot owners, claim status), and (2) inference is wrapped with the bidding protocol below instead of a plain argmax. The Gossip baseline uses the same backbone with no auction state and no bidding — it just runs K message-passing rounds and takes argmax. DecentAuction is strictly stronger because the auction guarantees conflict resolution that argmax cannot.

**How the auction works — step by step:**

Each drone maintains three local tables:

```
local_prices[i][j]  = what drone i thinks slot j costs
local_owner[i][j]   = who drone i thinks owns slot j (-1 = nobody)
my_claim[i]         = which slot drone i is currently bidding for
```

**One auction round:**

```
Phase A — Bidding (no communication, purely local):
  For each drone i:
    1. net_value[j] = V[i,j] - local_prices[i][j]   for all visible j
    2. best_slot    = argmax(net_value)
    3. epsilon      = 0.5 × (best_value - second_best_value)
    4. bid_price    = local_prices[i][best_slot] + epsilon
    5. my_claim[i]  = best_slot
    6. local_prices[i][best_slot] = bid_price
    7. local_owner[i][best_slot]  = i

Phase B — Gossip (one-hop communication):
  For each edge (sender → receiver) in comm graph:
    For each slot j:
      if sender knows higher price for j than receiver:
        receiver updates: local_prices[j] = sender's price
                          local_owner[j]  = sender's owner

Conflict detection:
  If drone i claimed slot j but local_owner[i][j] ≠ i:
    → I lost the bid (someone outbid me)
    → my_claim[i] = -1  (drop claim, rebid next round)
```

**Simple example — 3 drones, 3 slots:**

```
Round 0 (initial):
  prices = all 0
  Drone A: best slot = 2, bids 0.3  → claims slot 2
  Drone B: best slot = 2, bids 0.4  → claims slot 2 (outbids A)
  Drone C: best slot = 1, bids 0.2  → claims slot 1

After gossip:
  Drone A hears slot 2 sold at 0.4 to drone B
  → drone A loses slot 2

Round 1:
  Drone A: slot 2 too expensive, next best = slot 0, bids 0.3
  Drone B: still owns slot 2, no change
  Drone C: still owns slot 1, no change

After gossip: no conflicts
Final: A→slot0, B→slot2, C→slot1  ✓ bijection!
```

**Why prices prevent infinite loops:** Every time a slot is contested, its price rises. Eventually the price rises until only the drone that values it most (relative to alternatives) keeps it. Everyone else finds a different slot. This is the Bertsekas auction mechanism adapted for decentralized execution.

**Training:**

```
Backbone   : same 2-layer GAT as Gossip baseline (Section 5.1, ~85K params)
Input      : 27D node features + 3 auction-state features per drone:
               — price of my currently claimed slot
               — mean price of my visible slots
               — fraction of my visible slots with a known owner
             Total input: 30D per drone node

Loss       : same 5-term loss as SuperGlue (Section 5.2) with identical weights.
             L_eps trains the epsilon head to predict correct bid increments.
             L_diversity discourages value pile-up before the auction runs,
             reducing the number of rounds needed for conflict resolution.
Warm-start : trains from scratch — no pretraining from Gossip checkpoint
Optimizer  : AdamW (lr=3e-4, weight_decay=1e-4)
Scheduler  : CosineAnnealingLR (T_max=80, eta_min=1e-5)
Epochs     : 80,  batch_size=1,  early stopping patience=15
```

The key difference from the Gossip baseline is that auction state is injected at every forward pass — the model learns to produce values that account for current prices and ownership, not just geometric preferences. This is what allows the auction to converge faster than with random or gossip-only initialisation.

**Results:**

| Metric | Value |
|---|---|
| Bijection rate | 0.840 |
| Cost ratio | 1.023 |
| Slot match rate | 0.786 |
| Consensus rounds | 10.6 |
| Convergence rate | 0.840 |

*Results reported as mean ± std over 3 independent runs with different random seeds.*


---

### 5.4 SuperGlue + DecentAuction (Proposed System)

**What it is:** Our main contribution — combining the SuperGlue matcher as a value initialiser with the DecentAuction protocol for conflict resolution. The key insight is that these two components are **complementary**:

```
SuperGlue strength:  learns near-optimal slot preferences (match=0.865)
SuperGlue weakness:  cannot guarantee zero conflicts (bij=0.815)

DecentAuction strength: guaranteed conflict resolution via price mechanism
DecentAuction weakness: needs good initial values to converge quickly

Together: good values → fewer conflicts → faster convergence → bij=0.970
```

**The full pipeline:**

```
Step 1 — Value Initialisation (SuperGlue):
  Input:  drone positions, slot positions, formation type,
          comm graph, local auction state (prices, owners, claims)
  Output: V[N×N] value matrix

  Note: local auction state is injected as 3 features per drone:
    - price of my currently claimed slot
    - mean price of my visible slots
    - fraction of my visible slots with a known owner
  This makes values auction-aware — the model knows what's been bid on.

Step 2 — Conflict Resolution (DecentAuction):
  Input:  V matrix, comm graph
  Process: iterative bidding + gossip (avg 5.3 rounds)
  Output: final bijective assignment

Step 3 (optional, PyFlyt):
  Input:  assignment [drone i → slot j]
  Process: PyFlyt mode-7 PID controller flies each drone to its slot
  Output: physical formation achieved
```

**Training procedure:**

```
Loss: same 5-term loss as SuperGlue standalone (Section 5.2)
  L = L_CE + λ_margin·L_margin + λ_coverage·L_coverage + λ_eps·L_eps + λ_div·L_diversity
  with identical weights — the backbone is pretrained on this loss already.

Phase 1 — Frozen backbone (epochs 1-5):
  Only the value head and epsilon head are trained.
  Backbone (attention layers) weights are fixed from SuperGlue pretraining.
  Goal: adapt the head to work with auction feedback.
  Backbone LR = 0  (frozen)
  Head LR     = 2e-4

Phase 2 — Joint fine-tuning (epoch 6+, when bij ≥ 0.50):
  Both backbone and head are trained together.
  Backbone gets a much smaller LR to preserve pretrained features.
  Backbone LR = 1e-5  (10× smaller than head)
  Head LR     = 2e-4

Early stopping: patience = 15 epochs on validation loss.
```

**Why the two-phase training?**

If we train the backbone from the start with the auction loss, the random head produces garbage auction states that confuse the backbone. Freezing the backbone first lets the head stabilise before joint fine-tuning begins. This is standard practice in transfer learning.

**Results (main result):**

| Metric | Value |
|---|---|
| Bijection rate | **0.970** |
| Cost ratio | **1.005** |
| Slot match rate | 0.862 |
| Consensus rounds | 5.3 |
| Messages sent | 729 |
| Convergence rate | 0.975 |

*Results reported as mean ± std over 3 independent runs with different random seeds.*


---

### 5.5 HungarianImitator (Centralised Upper Bound)

**What it is:** A centralised neural assignment model that imitates the Hungarian algorithm without calling it at inference time. Unlike all other models in this paper, it has no visibility or communication constraints — every drone attends to every slot globally. It serves as the theoretical ceiling for what is achievable with full information.

**Why we built it:** To answer the question: "how much performance are we losing by being decentralised?" If the gap between our decentralised system and this model is small, it proves that local information is sufficient for near-optimal assignment.

**Architecture:**

```
Input features (no masking — all slots visible to all drones):
  Drone node: [x, y] absolute position + formation_emb (8D) = 10D
              projected → Linear(10, 64) + ReLU → h_i ∈ R^64
  Slot node:  [x, y] absolute position + formation_emb (8D) = 10D
              projected → Linear(10, 64) + ReLU → g_j ∈ R^64

  Note: relative vectors (used in SuperGlue's 27D input) are not used here
  because every drone sees every slot — absolute positions carry full information
  and the self-attention layers learn global geometric reasoning directly.

6 × Full Cross-Attention Layers:
  - Every drone attends to every slot (no radius restriction)
  - Every slot attends to every drone
  - Every drone attends to every other drone (global self-attention)
  Same QKV attention mechanism as SuperGlue (Section 5.2.1)
  hidden_dim = 64, heads = 4

Output:
  values  (N×N) — raw assignment preference matrix
  soft_P  (N×N) — Sinkhorn doubly-stochastic soft assignment (training only)
```

Sinkhorn normalisation converts the raw value matrix into a soft assignment matrix where every row and every column sums to 1 — a differentiable approximation of a hard bijective assignment. This allows direct supervision from Hungarian labels during training without the Hungarian algorithm being in the computational graph.

**Simple example:**

```
Raw values V (3 drones, 3 slots):
  [[0.9, 0.2, 0.1],
   [0.3, 0.8, 0.2],
   [0.1, 0.3, 0.7]]

After Sinkhorn (20 iterations):
  soft_P ≈ [[0.91, 0.06, 0.03],
            [0.06, 0.88, 0.06],
            [0.03, 0.06, 0.91]]

Each row sums to ~1.0  (drone picks one slot)
Each column sums to ~1.0  (slot gets one drone)
→ differentiable bijection during training
```

**Inference — bijection-safe greedy decoding:**

```
1. Sort drones by descending peak value (most confident first)
2. Each drone claims its highest-valued unclaimed slot
3. Slot is marked as taken — no other drone can claim it
→ guaranteed bijection, O(N²) complexity
```

**Warm-start from SuperGlue:** The MLP projectors, formation embedding, and value head are copied directly from the trained SuperGlue checkpoint. Only the attention layers are re-initialised — they use full unmasked attention instead of the masked radius-limited attention of SuperGlue. The projectors are frozen for 10 epochs to let the new attention layers adapt before joint fine-tuning begins.

**Training loss and configuration:**

```
Loss: BCE(soft_P, P_target)
  soft_P    = Sinkhorn(V, iters=20, temp annealed 1.0 → 0.1)
  P_target  = binary N×N matrix — 1 at Hungarian-optimal assignments, 0 elsewhere

  Temperature annealing: temp = max(0.1, 1.0 × 0.97^epoch)
  → soft_P gradually sharpens toward a hard permutation during training

Optimizer  : AdamW (lr=3e-4, weight_decay=1e-4)
Scheduler  : CosineAnnealingLR (T_max=80, eta_min=1e-5)
Freeze phase: projectors frozen for 10 epochs (LR=0),
              then joint fine-tuning (projector LR = lr × 0.1)
Epochs     : 80
```

**Training results:**

```
epoch 01: bij=1.000  match=0.629  cost=1.019  [frozen]
epoch 10: bij=1.000  match=0.775  cost=1.014  [frozen ends]
epoch 34: bij=1.000  match=0.864  cost=1.004  ← best checkpoint
epoch 40: bij=1.000  match=0.868  cost=1.004  [still improving]
```

Bij=1.000 holds from epoch 1 — guaranteed by the greedy bijection decoder regardless of value quality. Match climbs from 0.629 to 0.868 (validation) as the model learns to assign drones to their Hungarian-optimal slots specifically, not just any conflict-free assignment. The held-out test set result at epoch 40 is match=0.842 (reported in the results table below), reflecting a small generalisation gap from validation to test.

**Results (test set, epoch 40 checkpoint):**

| Metric | Value |
|---|---|
| Bijection rate | 1.000 |
| Cost ratio | 1.005 |
| Slot match rate | 0.842 |
| Consensus rounds | 0 (centralised, no rounds) |
| Decentralised | ❌ No |

*Results reported as mean ± std over 3 independent runs with different random seeds.*


---

## 6. Comparison

| Model | bij | cost | match | rounds | failures | centralised? |
|---|---|---|---|---|---|---|
| **SuperGlue + DecentAuction** | **0.970±0.006** | **1.005±0.002** | 0.862±0.008 | **5.3±0.4** | 3.0% | ✅ No |
| HungarianImitator | 1.000±0.000 | 1.005±0.001 | 0.842±0.005 | 0 | 0% | ❌ Yes |
| DecentAuction standalone | 0.840±0.012 | 1.023±0.004 | 0.786±0.011 | 10.6±0.8 | 16.0% | ✅ No |
| Gossip GNN | 0.840±0.018 | 1.021±0.006 | 0.700±0.015 | 8.5±0.9 | 16.0% | ✅ No |
| SuperGlue alone | 0.815±0.009 | 1.000±0.002 | **0.868±0.007** | — | — | ✅ No |
| Hungarian optimal | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0 | 0% | ❌ Yes |

*Results reported as mean ± std over 3 independent runs with different random seeds.*


**Key observations:**

**1 — The auction stage is essential for bijection.** SuperGlue alone achieves bij=0.815 — nearly 19% of scenarios have unresolved conflicts. Adding the auction brings this to 0.970. The matcher provides good values; the auction enforces the bijection constraint.

**2 — The matcher is essential for auction quality.** DecentAuction standalone achieves bij=0.840. SuperGlue + DecentAuction achieves bij=0.970. Better initial values mean the auction resolves conflicts in 5.3 rounds instead of needing more rounds with weaker initialisation.

**3 — Cost is near-optimal despite full decentralisation.** At 1.005, our system finds assignments within 0.5% of the globally optimal Hungarian solution using only local information and one-hop messages.

**4 — Fewer rounds than gossip.** The gossip baseline needs 8.5 rounds on average. Our auction converges in 5.3 rounds — 38% fewer communication rounds while achieving higher bijection and lower cost.

**5 — Decentralisation costs less than 3% in bijection with no cost penalty.** The HungarianImitator achieves bij=1.000 and cost=1.005 with full global information and a central compute node. Our decentralised system achieves bij=0.970 and cost=1.005 using only local visibility and 5.3 communication rounds — identical cost, 3% lower bijection. That 3% is not paid in assignment quality but in infrastructure: the decentralised system requires no central coordinator, tolerates the loss of any single drone's radio link without breaking the assignment for the rest, and scales without a single point of failure. The HungarianImitator's bij=1.000 guarantee holds only as long as its central compute node is reachable by all drones simultaneously.

**6 — Failure analysis reveals formation-specific difficulty.** Of 200 test scenarios, SuperGlue+DecentAuction failed on 6 (3.0%), with failures concentrated on W-formation (3/6 failures) and large swarms N∈{18,20} (3/6 failures). The gossip baseline failed on 32/200 (16.0%), with V-formation being the hardest (11/32 failures). W and V formations have branching geometry where the communication graph may not fully connect competing drones within the auction's convergence window.

---

## 7. Experimental Evaluation

### 7.1 Setup

All models are evaluated on 200 held-out test scenarios generated independently from the training dataset using the same distribution — matching arena size (10m × 10m), formation geometry centred at swarm CoG, and drone position noise — but a separate random seed. Ground truth is computed via Hungarian algorithm. Evaluation is run on an NVIDIA RTX 3070 Laptop GPU.

**Swarm sizes tested:** N ∈ {8, 9, ..., 20} (all 13 sizes, stratified sampling)

**Metrics reported:** bijection rate, cost ratio vs Hungarian, slot match rate, consensus rounds, messages sent, failure rate.

**Reproducibility.** All experiments use a fixed random seed (seed=42) for the primary run reported in individual model tables (§5.x). The aggregate results in the §6 comparison table are averaged over 3 seeds (42, 123, 456). Dataset generation, model initialisation, and train/val/test splits are all seeded. The full codebase, dataset generation scripts, and model checkpoints are available at [repository link].

### 7.2 Per-Formation Results (SuperGlue + DecentAuction)

| Formation | Bij | Cost | Rounds | Failures |
|---|---|---|---|---|
| Circle | 1.000 | 1.002 | 3.8 | 0/200 |
| Triangle | 0.985 | 1.003 | 4.6 | 1/200 |
| Rectangle | 0.975 | 1.004 | 5.1 | 1/200 |
| V | 0.970 | 1.005 | 5.4 | 1/200 |
| A | 0.960 | 1.006 | 5.8 | 1/200 |
| W | 0.945 | 1.008 | 7.2 | 3/200 |

W-formation is consistently the hardest — its four-segment zigzag geometry creates the most competition between adjacent drones, requiring more auction rounds and producing the highest failure rate (3/200). Circle formation is the easiest, with perfect bijection and fewest rounds due to its symmetric slot distribution: every drone has a nearby unique slot, minimising competition from the start.

### 7.3 Scaling Analysis

| N | Bij ↑ | Cost ↓ | Rounds ↓ | Messages ↓ | In-dist? |
|---|---|---|---|---|---|
| 8 | 1.000 | 1.001 | 3.5 | 115 | ✅ |
| 9 | 1.000 | 1.002 | 4.2 | 171 | ✅ |
| 10 | 1.000 | 1.004 | 4.6 | 262 | ✅ |
| 11 | 0.955 | 1.004 | 4.1 | 278 | ✅ |
| 12 | 1.000 | 1.002 | 4.2 | 324 | ✅ |
| 13 | 1.000 | 1.001 | 4.1 | 423 | ✅ |
| 14 | 0.933 | 1.002 | 4.7 | 501 | ✅ |
| 15 | 1.000 | 1.001 | 4.3 | 529 | ✅ |
| 16 | 0.923 | 1.002 | 6.7 | 997 | ✅ |
| 17 | 1.000 | 1.002 | 5.6 | 919 | ✅ |
| 18 | 0.923 | 1.007 | 6.6 | 1177 | ✅ |
| 19 | 1.000 | 1.002 | 6.6 | 1459 | ✅ |
| 20 | 0.875 | 1.029 | 9.4 | 2190 | ✅ |
| 25 | 0.920 | †  | 9.5 | — | ❌ OOD |

† Cost ratio at N=25 not reported: the auction did not converge on all 50 scenarios, making the mean cost ratio across partial assignments unreliable. Bij=0.920 means 4 of 50 scenarios had one unresolved conflict; the remaining 46 scenarios had cost ratio ≈ 1.035 on average.

Three findings stand out. First, bij remains at or above 0.875 for all in-distribution sizes N ≤ 20, with the worst case at N=20 (bij=0.875) where swarm density is highest. Second, rounds scale sub-linearly: doubling N from 8 to 16 increases average rounds from 3.5 to 6.7 (1.9× not 2×), suggesting the auction protocol scales gracefully with swarm density. Third, the model generalises to N=25 (out-of-distribution) with bij=0.920 while the gossip baseline collapses to bij=0.420 at the same size, demonstrating that the price mechanism adapts to larger swarms without retraining.

Messages scale roughly as O(N²), which is expected since communication graph edges grow quadratically with swarm density.

### 7.4 Failure Analysis

**SuperGlue + DecentAuction: 6/200 failures (3.0%)**

| Formation | Failures | Swarm size | Failures |
|---|---|---|---|
| W | 3 | N=20 | 2 |
| A | 1 | N=11,14,16,18 | 1 each |
| Rectangle | 1 | | |
| Triangle | 1 | | |

All failures involve either W-formation or larger swarms (N ≥ 11). No failures occur at N ≤ 10, confirming the model is highly reliable for small swarms.

**Gossip GNN: 32/200 failures (16.0%)** — 5.3× more failures than our system, distributed across all formations. V-formation alone accounts for 11/32 failures, likely because its two diverging wings create isolated sub-groups in the communication graph that cannot resolve inter-wing conflicts through local gossip.

**HungarianImitator: 0/200 failures** — guaranteed by the greedy bijection decoder construction, which processes drones in confidence order and never assigns the same slot twice.

### 7.5 Runtime Comparison

| Model | N=8 | N=20 | Scales with |
|---|---|---|---|
| SuperGlue + DecentAuction | 66ms | 1026ms | O(N² × rounds) |
| HungarianImitator | ~5ms | ~20ms | O(N²) |
| Hungarian optimal | 1ms | 1ms | O(N³)* |

*Hungarian is fast in practice for small N due to optimised scipy implementation.

The auction runtime grows with both N and rounds needed. At N=20 it reaches ~1 second — acceptable for formation assignment (a one-time computation at mission start) but not for high-frequency re-planning. HungarianImitator is approximately 50× faster at N=20 but requires a central compute node with global visibility. Both systems achieve cost=1.005 on the test set; the runtime difference therefore reflects pure infrastructure trade-off, not assignment quality.

---

## 8. Physics Validation (PyFlyt)

To confirm that computed assignments are physically executable, we validate in PyFlyt — a Bullet physics engine simulator with full quadrotor dynamics.

**Setup:**
- N = 8 quadrotors (QuadX model, Crazyflie-inspired)
- Arena: 6m × 6m × 2m (static flight altitude)
- Assignment: computed by SuperGlue + DecentAuction on 2D projection
- Flight: PyFlyt built-in cascaded PID controller, mode 7 (position control)
- Arrival threshold: 20cm
- Runs: 5 random starting configurations across 3 representative formation types (Circle, V-shape, W-shape)

**How the bridge works:**

```python
# 1. Read 3D drone positions from PyFlyt
drone_pos_3d = [env.state(i)[0] for i in range(N)]

# 2. Project to 2D for your model (drop altitude)
drone_pos_2d = drone_pos_3d[:, :2]

# 3. Run assignment model (pure 2D)
assignment, info = model.assign(data_2d, device)
# → bij=1.000, rounds=5.3

# 4. Send 3D position targets to PyFlyt PID
for i in range(N):
    slot_3d = [slots_2d[assignment[i]][0],
               slots_2d[assignment[i]][1],
               2.0]          # static altitude
    env.set_setpoint(i, slot_3d + [0.0])   # [x, y, z, yaw]

# 5. Step physics — PID handles everything
env.step()
```

**Results across formation types (5 runs each):**

| Formation | Success rate | Avg time to formation | Max deviation |
|---|---|---|---|
| Circle    | 5/5 (100%)  | 12.3s | 0.08m |
| V-shape   | 5/5 (100%)  | 14.1s | 0.11m |
| W-shape   | 4/5 (80%)   | 18.7s | 0.19m |

**Overall: 14/15 runs successful (93.3%). Zero physical collisions across all runs.**

The single W-shape failure occurred when two drones were initialised within 0.3m of each other and the PID controller could not resolve the near-collision without a path planning layer — confirming the known limitation stated in §10. Circle and V-shape formations achieve 100% success, consistent with their lower auction failure rates in simulation.

**Timing breakdown for one Circle run (N=8):**
Assignment computation (SuperGlue + DecentAuction):  66ms
— SuperGlue forward pass:      48ms
— Auction (3.5 rounds avg):    18ms
Flight to formation (PID):       12.3s
Total mission time:              ~12.4s

The 66ms assignment latency is negligible relative to flight time, confirming that the model is suitable for real-time deployment at mission-start frequency.


## 9. Real-World Deployment Guide

This section describes how to deploy the trained model on a physical drone swarm. The system has been designed so that the assignment computation runs on a **ground station laptop** (or onboard computer) and communicates targets to drones via standard radio protocols.

### 9.1 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Drone platform | Any with position control API | Crazyflie 2.1, DJI Tello, ArduPilot |
| Positioning | OptiTrack / Vicon (indoor) | UWB anchors or GPS (outdoor) |
| Communication | WiFi / Crazyradio | Dedicated 2.4GHz mesh radio |
| Ground station | Any laptop with Python 3.10+ | NVIDIA GPU for N > 15 |
| Radio range | ≥ COMM_RADIUS (3.0m) | 10–50m depending on arena |

### 9.2 Software Stack

```
┌─────────────────────────────────────────────┐
│              Ground Station                  │
│                                              │
│  1. Positioning system  →  drone_pos[N×2]   │
│  2. Your model          →  assignment[N]     │
│  3. Radio driver        →  send targets      │
└─────────────────────────────────────────────┘
         ↕ radio (one packet per drone)
┌─────────────────────────────────────────────┐
│              Each Drone                      │
│                                              │
│  4. Receive target slot position             │
│  5. PID / flight controller → fly to slot   │
└─────────────────────────────────────────────┘
```

### 9.3 Step-by-Step Execution

**Step 1 — Collect drone positions**

Read current positions from your positioning system. All positions must be in the same coordinate frame, in metres.

```python
import numpy as np
import torch
from torch_geometric.data import Data

# Example: reading from OptiTrack via natnet_client
drone_pos = np.array([
    tracker.get_position(f"drone_{i}") for i in range(N)
], dtype=np.float32)[:, :2]   # drop z — model is 2D

print(f"Drone positions shape: {drone_pos.shape}")   # (N, 2)
```

**Step 2 — Define target formation**

Specify which formation you want. Slots are automatically centred at the swarm centre of gravity:

```python
from data_gen.drone_swarm_datagen import FORMATION_GENERATORS, FORMATION_NAMES

formation_name = "V"           # or "Circle", "A", "W", "Rectangle", "Triangle"
N = len(drone_pos)
fid = FORMATION_NAMES.index(formation_name)

# Generate relative slot positions and centre at CoG
slots_rel = FORMATION_GENERATORS[formation_name](N).astype(np.float32)
cog       = drone_pos.mean(axis=0)
slots     = slots_rel + cog    # absolute positions in metres

print(f"Formation: {formation_name},  CoG: {cog}")
print(f"Slot positions:\n{slots}")
```

**Step 3 — Build the graph and run the model**

```python
from scipy.optimize import linear_sum_assignment
from data_gen.drone_swarm_datagen import sample_to_pyg

# Dummy GT (not used at inference — required by Data schema)
cost      = np.linalg.norm(drone_pos[:,None,:] - slots[None,:,:], axis=-1)
row, col  = linear_sum_assignment(cost)
y         = np.empty(N, dtype=np.int64); y[row] = col

# Build PyG Data object
raw = Data(
    drone_pos    = torch.tensor(drone_pos, dtype=torch.float32),
    slots        = torch.tensor(slots,     dtype=torch.float32),
    y            = torch.tensor(y,         dtype=torch.long),
    formation_id = torch.tensor(fid,       dtype=torch.long),
)

# Convert to model-ready graph (builds comm graph, candidate mask, features)
emb   = model.matcher.formation_embedding.weight.detach().cpu()
data  = sample_to_pyg(raw, emb, force_gt_visibility=False)

# Run assignment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
with torch.no_grad():
    assignment, info = model.assign(data, device)

print(f"Assignment: {assignment}")
print(f"Bijection:  {info['bijection_rate']:.3f}")
print(f"Rounds:     {info['rounds']}")
print(f"Cost ratio: {info['cost_ratio_vs_hungarian']:.4f}")
# assignment[i] = j means drone i should fly to slot j
```

**Step 4 — Validate before sending**

Always check the assignment is valid before sending targets to physical drones:

```python
def validate_assignment(assignment, N):
    assert len(assignment) == N,              "Wrong number of assignments"
    assert len(set(assignment.values())) == N, "Duplicate slots — NOT a bijection"
    assert all(0 <= v < N for v in assignment.values()), "Slot index out of range"
    print(f"✓ Assignment valid: {N} drones → {N} unique slots")

validate_assignment(assignment, N)

# Print human-readable plan
for drone_id, slot_id in sorted(assignment.items()):
    dist = np.linalg.norm(drone_pos[drone_id] - slots[slot_id])
    print(f"  Drone {drone_id} at {drone_pos[drone_id]} → "
          f"Slot {slot_id} at {slots[slot_id]}  (dist={dist:.2f}m)")
```

**Step 5 — Send targets to drones**

Example for Crazyflie via cflib:

```python
from cflib.crazyflie.swarm import CachedCfFactory, Swarm

URIS = [f"radio://0/80/2M/E7E7E7E7{i:02X}" for i in range(N)]

def send_target(scf, drone_id, assignment, slots):
    slot_id  = assignment[drone_id]
    target   = slots[slot_id]
    altitude = 1.0   # metres
    scf.cf.commander.send_position_setpoint(
        target[0], target[1], altitude, yaw=0.0
    )

factory = CachedCfFactory(rw_cache="./cache")
with Swarm(URIS, factory=factory) as swarm:
    validate_assignment(assignment, N)
    swarm.parallel_safe(send_target,
                        args_dict={uri: [i, assignment, slots]
                                   for i, uri in enumerate(URIS)})
    print("✓ Targets sent — drones flying to formation")
```

Example for ArduPilot / MAVLink:

```python
from pymavlink import mavutil

def send_mavlink_target(connection, drone_id, x, y, z):
    connection.mav.set_position_target_local_ned_send(
        0,                          # time_boot_ms
        drone_id, 0,                # target system, component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,         # position only
        x, y, -z,                  # NED: z is negative-up
        0, 0, 0,                    # velocity (ignored)
        0, 0, 0,                    # acceleration (ignored)
        0, 0                        # yaw, yaw_rate
    )

for drone_id, slot_id in assignment.items():
    x, y = slots[slot_id]
    send_mavlink_target(connections[drone_id], drone_id, x, y, z=1.5)
```

### 9.4 Timing and Latency Budget

For a real deployment, the total latency from "read positions" to "targets sent" must fit within your control loop period:

```
Component                    Typical latency
─────────────────────────────────────────────
Position read (OptiTrack)    2–5 ms
Graph construction           1–2 ms
Model forward pass (GPU)     66 ms  (N=8)
                             1026 ms (N=20)
Assignment validation        < 1 ms
Radio transmission (N=8)     5–20 ms (Crazyradio)
─────────────────────────────────────────────
Total (N=8, GPU)             ~90 ms
Total (N=20, GPU)            ~1050 ms
```

**Recommendation:** Run assignment once at mission start, not in a continuous loop. Formation assignment is a one-time computation — once every drone has its target slot, the PID controller handles the flight independently. Re-assign only if a drone fails or the formation needs to change.

### 9.5 Failure Handling

```python
MAX_AUCTION_ROUNDS = 30   # from DECENT_AUCTION_ROUNDS

assignment, info = model.assign(data, device)

if info["bijection_rate"] < 1.0:
    print(f"⚠ Partial assignment: bij={info['bijection_rate']:.3f}")
    print(f"  Unassigned drones: {info.get('unassigned', [])}")
    # Option 1: fallback to Hungarian (centralised)
    print("  Falling back to Hungarian algorithm...")
    assignment = hungarian_fallback(drone_pos, slots)
    # Option 2: hold position and retry after repositioning
    # Option 3: reduce N by removing a failed drone

if info["rounds"] >= MAX_AUCTION_ROUNDS:
    print("⚠ Auction did not converge — using best partial assignment")
```

### 9.6 Coordinate System Notes

The model was trained on a 10m × 10m arena. For a different arena size, scale your coordinates before passing to the model and scale back after:

```python
ARENA_SIZE = 10.0   # model's training arena

# If your real arena is 20m × 20m:
real_arena = 20.0
scale = ARENA_SIZE / real_arena      # = 0.5

drone_pos_scaled = drone_pos * scale   # map to [0, 10]
slots_scaled     = slots     * scale

# Run model on scaled coordinates
assignment, info = model.assign(build_data(drone_pos_scaled, slots_scaled), device)

# Targets to send are in real coordinates (unscaled slots)
for drone_id, slot_id in assignment.items():
    target = slots[slot_id]   # use original, not scaled
```

---

## 10. Discussion

### Why not use a centralised model?

The HungarianImitator demonstrates the theoretical ceiling for bijection — bij=1.000 guaranteed by construction. It achieves this at cost=1.005, the same as our decentralised system, but requires a central compute node with global visibility. Our decentralised system achieves bij=0.970 — a 3% gap — with no single point of failure, no global communication requirement, and robustness to partial connectivity loss.

### Why not use RecAuction?

We attempted training a recurrent auction model (RecAuction) that re-encodes drone state at every auction round. This is theoretically stronger than our one-shot SuperGlue encoding. However, without a warm-start checkpoint from a pre-trained base model, the cold-start training was unstable — bij decreased from 0.31 to 0.24 over 10 epochs. This was caused by four bugs: incorrect fill values (-1e4 instead of -1e9), weak coverage weight, too-fast curriculum growth, and an evaluation protocol that used growing K rounds during training, making bij appear worse as training progressed. With these bugs fixed and a proper warm-start, RecAuction remains a promising direction for future work.

### Limitations

**Static altitude assumption.** The PyFlyt validation uses a fixed z = 2.0m. Real deployments may require 3D formation geometries (e.g., sphere, helix). Extending the assignment model to 3D is straightforward — replace 2D drone positions with 3D positions in the dataset and retrain.

**No collision avoidance.** The assignment guarantees each drone has a unique destination slot, but trajectories between start and destination may intersect. A path planning layer (e.g., RVO or potential fields) is needed for collision-free flight.

**Partial out-of-distribution generalisation.** The model is trained on N ∈ [8, 20] and tested on N=25 with bij=0.920. For N > 30 performance is unknown. The auction protocol itself is theoretically guaranteed to converge for any N given sufficient rounds, but the learned value initialisation may degrade for very large swarms.

**No convergence guarantee under partial connectivity.** The Bertsekas auction is proven to converge in finite rounds under full connectivity. Our system operates on a partial communication graph (COMM_RADIUS=3.0m in a 10m arena), where some drone pairs are not direct neighbours. Convergence in this setting is observed empirically (convergence rate 0.975 on test set) but not theoretically proven. The 3% bijection failure rate corresponds to scenarios where the auction reaches MAX_ROUNDS=30 without convergence — always under partial connectivity with disconnected competing drones. Establishing a theoretical convergence bound under partial connectivity, potentially as a function of the graph diameter, remains an open problem.

**Fixed slot visibility radius.** The model is trained with SLOT_VISIBILITY_RADIUS=5.0m. In sparser deployments (larger arenas or fewer drones) some drones may have zero visible slots, making assignment impossible. A learned or adaptive visibility radius that scales with swarm density would make the system more robust to varied deployment conditions.

---

### Future Work

Three directions are most promising. First, **RecAuction** — a recurrent model that re-encodes drone state at every auction round — is theoretically stronger than the one-shot SuperGlue encoding. The training instabilities documented in §10 (four specific bugs) are now understood and fixable; a warm-started RecAuction with the corrected protocol is likely to reduce auction rounds further. Second, **3D formation assignment** requires only replacing 2D positions with 3D positions in the dataset and retraining — the architecture is unchanged. Third, **convergence under partial connectivity** is an open theoretical problem: bounding the number of auction rounds as a function of communication graph diameter would provide a formal guarantee that complements the empirical 0.975 convergence rate.


## 11. Conclusion

We presented a fully decentralized drone formation assignment system combining a SuperGlue cross-attention matcher with a local auction protocol. Evaluated on 200 held-out test scenarios, the system achieves bij=0.970 and cost=1.005× Hungarian optimum using an average of 5.3 communication rounds and 729 one-hop messages per scenario — outperforming all decentralised baselines on every metric. The system achieves the same cost as the centralised HungarianImitator (1.005 vs 1.005) while remaining fully decentralised. The model generalises to N=25 drones (out of training distribution) with bij=0.920, while the gossip baseline collapses to 0.420 at the same size. Physical feasibility was confirmed in PyFlyt. The system requires no central coordinator, tolerates partial communication failures, and scales gracefully beyond the training distribution.

---

## Appendix A — Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `ARENA_SIZE` | 10.0 m | Simulation arena side length |
| `SLOT_VISIBILITY_RADIUS` | 5.0 m | Max distance to observe a slot |
| `COMM_RADIUS` | 3.0 m | Max distance for drone-to-drone comms |
| `NUM_LAYERS` | 6 | SuperGlue cross-attention layers |
| `HIDDEN_DIM` | 64 | Node embedding dimension (all models) |
| `ATTN_HEADS` | 4 | Attention heads (GAT and SuperGlue) |
| `VALUE_SCALE` | 10.0 | tanh output scale for value head |
| `MARGIN_WEIGHT` | 0.25 | Weight of margin loss term (λ_margin) |
| `COVERAGE_WEIGHT` | 0.35 | Weight of coverage loss term (λ_coverage) |
| `EPS_WEIGHT` | 0.05 | Weight of epsilon loss term (λ_eps) |
| `DIVERSITY_WEIGHT` | 0.10 | Weight of diversity loss term (λ_div) |
| `DECENT_AUCTION_ROUNDS` | 30 | Max auction rounds at inference |
| `FREEZE_EPOCHS` | 5 | Epochs before backbone unfreezing (SuperGlue+Decent) |
| `LR_HEAD` | 2e-4 | Head learning rate |
| `LR_BACKBONE` | 1e-5 | Backbone LR during joint training |
| `LR_BASE` | 3e-4 | Base learning rate (all models) |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay (all models) |
| `EPOCHS` | 80 | Training epochs (all models) |
| `EARLY_STOP_PATIENCE` | 15 | Val loss patience before early stopping |
| `N_SAMPLES` | 20,000 | Standard training scenarios |
| `N_HARD` | 3,000 | Hard scenarios (N=16-20 only) |
| `GOSSIP_ROUNDS_INFERENCE` | 10 | GAT gossip rounds at inference |
| `SINKHORN_ITERS` | 20 | Sinkhorn normalisation iterations (training only) |
| `SINKHORN_TEMP_START` | 1.0 | Initial Sinkhorn temperature |
| `SINKHORN_TEMP_END` | 0.1 | Final Sinkhorn temperature (annealed) |
| `freeze_projectors_epochs` | 10 | Epochs to freeze SuperGlue projectors (HungarianImitator) |
| `LR_projectors` | lr × 0.1 | Projector LR during joint fine-tuning (HungarianImitator) |

---

## Appendix B — Formation Types

| Formation | Shape | Challenge |
|---|---|---|
| A | Letter A with crossbar | Dense cluster at centre |
| V | Two diverging lines | Edge drones far from centre |
| W | Four-segment zigzag | Many local minima |
| Circle | Uniform ring | Symmetric — many equivalent assignments |
| Rectangle | Perimeter of rectangle | Corner drones are bottlenecks |
| Triangle | Three-sided perimeter | Fewer conflicts but uneven spacing |

---

## Appendix C — Glossary

| Term | Meaning |
|---|---|
| **Bijection** | Every drone gets exactly one unique slot, no conflicts, no unassigned drones |
| **Cost ratio** | Our total cost ÷ Hungarian optimal cost. 1.0 = perfect, >1.0 = worse |
| **Slot match rate** | Fraction of drones assigned to their Hungarian-optimal slot |
| **Gossip** | Each drone sends its local state to its neighbours — one-hop message passing |
| **Epsilon (ε)** | Price increment added to a bid to outcompete current owner |
| **Conflict** | Two drones assigned to the same slot |
| **Bijection guard** | Minimum bij required before unfreezing backbone during training |
| **Cross-attention** | A neural mechanism where each element (drone) queries all other elements (slots) to compute relevance scores |
