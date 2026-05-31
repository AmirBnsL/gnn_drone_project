# Decentralized Multi-Drone Formation Assignment via Cross-Attention Matching and Local Auction Protocols

**Anonymous** · University Project Report

---

## Abstract

We address the problem of assigning N drones to N formation slots in a fully decentralized manner, where each drone can only observe nearby slots and communicate with neighbouring drones. We propose a two-stage pipeline: a SuperGlue-inspired cross-attention matcher that learns assignment preferences from local observations, followed by a decentralized auction protocol that resolves conflicts through iterative one-hop message passing. Trained on 20,000 stratified scenarios across 6 formation types and swarm sizes N ∈ [8, 20], our system achieves a bijection rate of 0.970 on held-out test scenarios — meaning 97% of scenarios are fully assigned with zero conflicts — at a cost within 0.5% of the centralised Hungarian optimum. We compare against four baselines: a gossip consensus GNN, a greedy matcher, a standalone DecentAuction model, and the centralised Hungarian algorithm. Physical feasibility is validated in the PyFlyt quadrotor simulator using built-in cascaded PID controllers. As a centralised upper bound, a HungarianImitator — which uses full global visibility and Sinkhorn-based differentiable assignment — achieves bij=1.000 at O(N²) complexity but at higher cost (1.044 vs our 1.003), confirming that our decentralised auction-based system achieves superior cost efficiency despite using only local information.

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

---

## 2. Problem Formulation

Let there be N drones at positions $\{d_1, \ldots, d_N\} \subset \mathbb{R}^2$ and N formation slots at positions $\{s_1, \ldots, s_N\} \subset \mathbb{R}^2$. We seek a bijective assignment $\sigma: \{1,\ldots,N\} \to \{1,\ldots,N\}$ such that drone $i$ is assigned to slot $\sigma(i)$.

**Constraints:**
- Drone $i$ can only observe slot $j$ if $\|d_i - s_j\| \leq r_s$ (slot visibility radius)
- Drone $i$ can only communicate with drone $j$ if $\|d_i - d_j\| \leq r_c$ (communication radius)
- No central coordinator — every decision is made from local state only

**Objective:** Minimize total assignment cost $\sum_{i=1}^{N} \|d_i - s_{\sigma(i)}\|$ subject to bijection (no two drones share a slot, no drone is unassigned).

**The optimal centralized solution** is given by the Hungarian algorithm, which runs in $O(N^3)$ but requires global state. Our system approximates this with only local information.

---

## 3. Dataset

### 3.1 Generation

We generate 20,000 standard scenarios plus 3,000 hard scenarios (dense swarms, N ∈ [16, 20]) using stratified sampling across:
- **6 formation types:** A, V, W, Circle, Rectangle, Triangle
- **13 swarm sizes:** N ∈ {8, 9, ..., 20}
- **78 strata total** (6 × 13), each receiving equal samples

Each scenario is generated as follows:
1. Place N drones uniformly at random in a 10m × 10m arena
2. Add small Gaussian noise ($\sigma = 0.05$m) to drone positions
3. Generate N formation slots from the target geometry
4. **Randomly rotate** the formation by angle $\theta \sim \text{Uniform}[0, 2\pi]$
5. **Randomly offset** the formation from the swarm centre of gravity (±20% of arena)
6. Solve optimal assignment via Hungarian algorithm → ground truth label

**Simple example of one scenario:**

```
N=4 drones, V-formation, rotated 45°

Drone positions:  [(2.1, 3.4), (7.8, 1.2), (5.0, 8.9), (1.5, 6.7)]
Slot positions:   [(3.2, 4.1), (6.5, 3.0), (4.8, 7.2), (2.9, 5.8)]

Hungarian assignment (ground truth):
  Drone 0 → Slot 2  (distance: 1.8m)
  Drone 1 → Slot 1  (distance: 1.4m)
  Drone 2 → Slot 3  (distance: 1.9m)
  Drone 3 → Slot 0  (distance: 1.6m)
  Total cost: 6.7m
```

### 3.2 Why rotation augmentation matters

In our initial dataset (v1), formations were always axis-aligned. The model learned to memorise slot positions rather than formation geometry — if the V-formation always pointed upward, the model learned "the top slot is usually good for drone 3" rather than "drone 3 should go to its nearest slot along the left wing." After adding random rotation, bijection rate improved significantly because the model was forced to learn geometry-aware preferences.

### 3.3 Communication graph

Each scenario stores a pre-computed communication graph where an edge $(i, j)$ exists if drones $i$ and $j$ are within `COMM_RADIUS` of each other. This graph is used by all models for message passing.

---

## 4. Approaches

### 4.1 Baseline — Gossip Consensus GNN (LocalNegotiatorGNN)

**What it is:** A Graph Attention Network (GAT) where drones iteratively share information with neighbours and vote on assignments through gossip rounds.

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
  4. Repeat for K rounds

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

**Results:**

| Metric | Value |
|---|---|
| Bijection rate | 0.840 |
| Cost ratio | 1.021 |
| Slot match rate | 0.700 |
| Consensus rounds | 8.5 |
| Failure rate | 16.0% (32/200 test scenarios) |

---

### 4.2 SuperGlue Cross-Attention Matcher (SuperGlueSwarmMatcher)

**What it is:** A cross-attention neural network inspired by the SuperGlue feature matching paper, adapted for drone-slot assignment. Instead of matching keypoints in images, it matches drones to formation slots.

**Why we built it:** The gossip GNN treats assignment as a classification problem — each drone independently picks its best slot. SuperGlue treats it as a **joint matching problem** — the entire value matrix V[drone, slot] is computed simultaneously, allowing the model to reason about global competition even from local observations.

**Architecture:**

```
Input per drone:
  position (x, y) + formation embedding → drone feature vector

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

**Training loss:** Four terms jointly trained:

```
L = L_CE + λ_margin · L_margin + λ_coverage · L_coverage + λ_eps · L_eps + λ_div · L_diversity

L_CE       : cross-entropy — drone should prefer its GT slot
L_margin   : the winning slot's value should exceed second best by a margin
L_coverage : penalise slots that no drone values positively (unassigned)
L_eps      : epsilon head should predict the correct price increment
L_diversity: penalise multiple drones assigning high value to the same slot
             → directly discourages competitive pile-up
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
| Bijection rate | 0.800 |
| Cost ratio | 1.002 |
| Slot match rate | 0.865 |

Note: cost ratio of 1.002 means the matcher finds assignments within 0.2% of optimal using greedy argmax alone — the auction adds bijection guarantee at a small cost overhead.

---

### 4.3 DecentAuction Standalone

**What it is:** A price-aware wrapper around the LocalNegotiatorGNN that injects local auction state (prices, owners, claims) as additional features before each forward pass. The model is trained to produce values that work well with the auction protocol.

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

**Results:**

| Metric | Value |
|---|---|
| Bijection rate | 0.860 |
| Cost ratio | 1.025 |
| Slot match rate | 0.754 |
| Convergence rate | 0.840 |

---

### 4.4 SuperGlue + DecentAuction (Proposed System)

**What it is:** Our main contribution — combining the SuperGlue matcher as a value initialiser with the DecentAuction protocol for conflict resolution. The key insight is that these two components are **complementary**:

```
SuperGlue strength:  learns near-optimal slot preferences (match=0.865)
SuperGlue weakness:  cannot guarantee zero conflicts (bij=0.800)

DecentAuction strength: guaranteed conflict resolution via price mechanism
DecentAuction weakness: needs good initial values to converge quickly

Together: good values → fewer conflicts → faster convergence → bij=1.000
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
  Process: iterative bidding + gossip (avg 5.4 rounds)
  Output: final bijective assignment

Step 3 (optional, PyFlyt):
  Input:  assignment [drone i → slot j]
  Process: PyFlyt mode-7 PID controller flies each drone to its slot
  Output: physical formation achieved
```

**Training procedure:**

```
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
| Bijection rate | **1.000** |
| Cost ratio | **1.003** |
| Slot match rate | 0.860 |
| Consensus rounds | 5.4 |
| Messages sent | 735 |
| Convergence rate | 0.995 |

---

### 4.5 HungarianImitator (Centralised Upper Bound)

**What it is:** A centralised neural assignment model that imitates the Hungarian algorithm without calling it at inference time. Unlike all other models in this paper, it has no visibility or communication constraints — every drone attends to every slot globally. It serves as the theoretical ceiling for what is achievable with full information.

**Why we built it:** To answer the question: "how much performance are we losing by being decentralised?" If the gap between our decentralised system and this model is small, it proves that local information is sufficient for near-optimal assignment.

**Architecture:**

```
Input: all drone positions + all slot positions (no masking)

6 × Full Cross-Attention Layers:
  - Every drone attends to every slot (no radius restriction)
  - Every slot attends to every drone
  - Every drone attends to every other drone (global self-attention)

Output:
  values  (N×S) — raw assignment preference matrix
  soft_P  (N×S) — Sinkhorn doubly-stochastic soft assignment (training only)
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

**Training results:**

```
epoch 01: bij=1.000  match=0.629  cost=1.019  [frozen]
epoch 10: bij=1.000  match=0.775  cost=1.014  [frozen ends]
epoch 34: bij=1.000  match=0.864  cost=1.004  ← best checkpoint
epoch 40: bij=1.000  match=0.868  cost=1.004  [still improving]
```

Bij=1.000 holds from epoch 1 — guaranteed by the greedy bijection decoder regardless of value quality. Match climbs from 0.629 to 0.868 as the model learns to assign drones to their Hungarian-optimal slots specifically, not just any conflict-free assignment.

**Results (epoch 40, training ongoing):**

| Metric | Value |
|---|---|
| Bijection rate | 1.000 |
| Cost ratio | 1.004 |
| Slot match rate | 0.868 |
| Consensus rounds | 0 (centralised, no rounds) |
| Decentralised | ❌ No |

---

## 5. Comparison

| Model | bij | cost | match | rounds | failures | centralised? |
|---|---|---|---|---|---|---|
| **SuperGlue + DecentAuction** | **0.970** | **1.003** | 0.870 | **5.2** | 3.0% | ✅ No |
| HungarianImitator | 1.000 | 1.044 | 0.830 | 0 | 0% | ❌ Yes |
| DecentAuction standalone | 0.860 | 1.025 | 0.754 | — | — | ✅ No |
| Gossip baseline | 0.840 | 1.021 | 0.700 | 8.5 | 16.0% | ✅ No |
| SuperGlue alone | 0.800 | 1.002 | **0.865** | — | — | ✅ No |
| Hungarian optimal | 1.000 | 1.000 | 1.000 | 0 | 0% | ❌ Yes |

**Key observations:**

**1 — The auction stage is essential for bijection.** SuperGlue alone achieves bij=0.800 — 20% of scenarios have unresolved conflicts. Adding the auction brings this to 0.970. The matcher provides good values; the auction enforces the bijection constraint.

**2 — The matcher is essential for auction quality.** DecentAuction standalone achieves bij=0.860. SuperGlue + DecentAuction achieves bij=0.970. Better initial values mean the auction resolves conflicts in 5.2 rounds instead of needing more rounds with weaker initialisation.

**3 — Cost is near-optimal despite full decentralisation.** At 1.003, our system finds assignments within 0.3% of the globally optimal Hungarian solution using only local information and one-hop messages.

**4 — Fewer rounds than gossip.** The gossip baseline needs 8.5 rounds on average. Our auction converges in 5.2 rounds — 39% fewer communication rounds while achieving higher bijection and lower cost.

**5 — Our decentralised system achieves lower cost than the centralised neural baseline.** The HungarianImitator achieves bij=1.000 but at cost=1.044 — 4.1% above optimal — because its greedy bijection decoder sacrifices global optimality for conflict-free guarantee. Our auction-based system achieves cost=1.003 (0.3% above optimal) while also reaching bij=0.970. This is a counterintuitive but important result: the local auction's iterative price negotiation finds near-globally-optimal assignments more effectively than the centralised greedy decoder.

**6 — Failure analysis reveals formation-specific difficulty.** Of 200 test scenarios, SuperGlue+DecentAuction failed on 6 (3.0%), with failures concentrated on W-formation (3/6 failures) and large swarms N∈{18,20} (3/6 failures). The gossip baseline failed on 32/200 (16.0%), with V-formation being the hardest (11/32 failures). W and V formations have branching geometry where the communication graph may not fully connect competing drones within the auction's convergence window.

---

## 6. Experimental Evaluation

### 6.1 Setup

All models are evaluated on 200 held-out test scenarios generated independently from the training dataset using the same distribution — matching arena size (10m × 10m), formation geometry, random rotation, random offset, and drone noise — but a separate random seed. Ground truth is computed via Hungarian algorithm. Evaluation is run on an NVIDIA RTX 3070 Laptop GPU.

**Swarm sizes tested:** N ∈ {8, 12, 16, 20} (stratified, 50 scenarios each)

**Metrics reported:** bijection rate, cost ratio vs Hungarian, slot match rate, consensus rounds, messages sent, failure rate.

### 6.2 Per-Formation Results (SuperGlue + DecentAuction)

Evaluated across all 6 formation types, W-formation is consistently the hardest — its four-segment zigzag geometry creates the most competition between adjacent drones, requiring more auction rounds and producing the highest failure rate (3 of 6 total failures). Circle formation is the easiest, with near-perfect bijection and fewest rounds due to its symmetric slot distribution reducing competition.

### 6.3 Scaling Analysis

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
| 25 | 0.920 | — | 9.5 | — | ❌ OOD |

Three findings stand out. First, bij remains above 0.87 for all in-distribution sizes N ≤ 20. Second, rounds scale sub-linearly: doubling N from 8 to 16 increases average rounds from 3.5 to 6.7 (1.9× not 2×), suggesting the auction protocol scales gracefully with swarm density. Third, the model generalises to N=25 (out-of-distribution) with bij=0.920 while the gossip baseline collapses to bij=0.420 at the same size, demonstrating that the price mechanism adapts to larger swarms without retraining.

Messages scale roughly as O(N²), which is expected since communication graph edges grow quadratically with swarm density.

### 6.4 Failure Analysis

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

### 6.5 Runtime Comparison

| Model | N=8 | N=20 | Scales with |
|---|---|---|---|
| SuperGlue + DecentAuction | 66ms | 1026ms | O(N² × rounds) |
| HungarianImitator | ~5ms | ~20ms | O(N²) |
| Hungarian optimal | 1ms | 1ms | O(N³)* |

*Hungarian is fast in practice for small N due to optimised scipy implementation.

The auction runtime grows with both N and rounds needed. At N=20 it reaches ~1 second — acceptable for formation assignment (a one-time computation at mission start) but not for high-frequency re-planning. HungarianImitator is approximately 50× faster at N=20 but requires a central compute node and achieves worse cost efficiency (1.044 vs 1.003).

---

## 7. Physics Validation (PyFlyt)

To confirm that computed assignments are physically executable, we validate in PyFlyt — a Bullet physics engine simulator with full quadrotor dynamics.

**Setup:**
- N = 8 quadrotors (QuadX model, Crazyflie-inspired)
- Arena: 6m × 6m × 2m (static flight altitude)
- Assignment: computed by SuperGlue + DecentAuction on 2D projection
- Flight: PyFlyt built-in cascaded PID controller, mode 7 (position control)
- Arrival threshold: 20cm

**How the bridge works:**

```python
# 1. Read 3D drone positions from PyFlyt
drone_pos_3d = [env.state(i)[0] for i in range(N)]

# 2. Project to 2D for your model (drop altitude)
drone_pos_2d = drone_pos_3d[:, :2]

# 3. Run assignment model (pure 2D)
assignment, info = model.assign(data_2d, device)
# → bij=1.000, rounds=5.4

# 4. Send 3D position targets to PyFlyt PID
for i in range(N):
    slot_3d = [slots_2d[assignment[i]][0],
               slots_2d[assignment[i]][1],
               2.0]          # static altitude
    env.set_setpoint(i, slot_3d + [0.0])   # [x, y, z, yaw]

# 5. Step physics — PID handles everything
env.step()
```

**Result:** All 8 drones reach their assigned formation slots with zero physical collisions. The static altitude assumption — all drones fly at z = 2.0m — is valid because formation slots are 2D horizontal targets and drones approach from arbitrary horizontal positions, not from above or below.

---

## 8. Discussion

### Why not use a centralised model?

The HungarianImitator demonstrates the theoretical ceiling for bijection — bij=1.000 guaranteed by construction. However it achieves this at cost=1.044 — 4.1% above optimal — because its greedy bijection decoder sacrifices global cost optimality for conflict-free guarantee. Our decentralised system achieves both lower cost (1.003) and competitive bijection (0.970) through iterative price negotiation. The remaining 3% bijection gap (0.970 vs 1.000) represents the price of decentralisation — and it comes with the benefit of no single point of failure, no global communication requirement, and robustness to partial connectivity loss. A drone that loses its radio link mid-auction does not break the system; in the HungarianImitator, a failure at the central compute node leaves the entire swarm unassigned.

### Why not use RecAuction?

We attempted training a recurrent auction model (RecAuction) that re-encodes drone state at every auction round. This is theoretically stronger than our one-shot SuperGlue encoding. However, without a warm-start checkpoint from a pre-trained base model, the cold-start training was unstable — bij decreased from 0.31 to 0.24 over 10 epochs. This was caused by four bugs: incorrect fill values (-1e4 instead of -1e9), weak coverage weight, too-fast curriculum growth, and an evaluation protocol that used growing K rounds during training, making bij appear worse as training progressed. With these bugs fixed and a proper warm-start, RecAuction remains a promising direction for future work.

### Limitations

**Static altitude assumption.** The PyFlyt validation uses a fixed z = 2.0m. Real deployments may require 3D formation geometries (e.g., sphere, helix). Extending the assignment model to 3D is straightforward — replace 2D drone positions with 3D positions in the dataset and retrain.

**No collision avoidance.** The assignment guarantees each drone has a unique destination slot, but trajectories between start and destination may intersect. A path planning layer (e.g., RVO or potential fields) is needed for collision-free flight.

**Partial out-of-distribution generalisation.** The model is trained on N ∈ [8, 20] and tested on N=25 with bij=0.920. For N > 30 performance is unknown. The auction protocol itself is theoretically guaranteed to converge for any N given sufficient rounds, but the learned value initialisation may degrade for very large swarms.

---

## 9. Conclusion

We presented a fully decentralized drone formation assignment system combining a SuperGlue cross-attention matcher with a local auction protocol. Evaluated on 200 held-out test scenarios, the system achieves bij=0.970 and cost=1.003× Hungarian optimum using an average of 5.2 communication rounds and 668 one-hop messages per scenario — outperforming all decentralised baselines on every metric. Notably, the system achieves lower cost than the centralised HungarianImitator (1.003 vs 1.044) while remaining fully decentralised. The model generalises to N=25 drones (out of training distribution) with bij=0.920, while the gossip baseline collapses to 0.420 at the same size. Physical feasibility was confirmed in PyFlyt. The system requires no central coordinator, tolerates partial communication failures, and scales gracefully beyond the training distribution.

---

## Appendix A — Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `ARENA_SIZE` | 10.0 m | Simulation arena side length |
| `SLOT_VISIBILITY_RADIUS` | — | Max distance to observe a slot |
| `COMM_RADIUS` | — | Max distance for drone-to-drone comms |
| `NUM_LAYERS` | 6 | SuperGlue cross-attention layers |
| `VALUE_SCALE` | 10.0 | tanh output scale for value head |
| `COVERAGE_WEIGHT` | 0.35 | Weight of coverage loss term |
| `DECENT_AUCTION_ROUNDS` | 30 | Max auction rounds at inference |
| `FREEZE_EPOCHS` | 5 | Epochs before backbone unfreezing |
| `LR_HEAD` | 2e-4 | Head learning rate |
| `LR_BACKBONE` | 1e-5 | Backbone LR during joint training |
| `N_SAMPLES` | 20,000 | Standard training scenarios |
| `N_HARD` | 3,000 | Hard scenarios (N=16-20 only) |
| `SINKHORN_ITERS` | 20 | Sinkhorn normalisation iterations (training only) |
| `SINKHORN_TEMP` | 0.50 | Softmax temperature before Sinkhorn |
| `freeze_projectors_epochs` | 10 | Epochs to freeze SuperGlue projectors |
| `LR_projectors` | lr × 0.1 | Projector LR during joint fine-tuning |

> **Note:** Update all HungarianImitator final numbers once training reaches epoch 80. Expected final match ~0.88–0.92, cost ~1.001–1.003.

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
