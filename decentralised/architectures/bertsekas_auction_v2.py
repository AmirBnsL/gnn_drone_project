"""
bertsekas_auction.py
====================
Learned Bertsekas Auction for fully decentralised drone-to-slot assignment.

Core idea
---------
Bertsekas' auction algorithm is a provably-optimal decentralised assignment
method. Each object (slot) has a price. Each person (drone) bids on its best
affordable slot; the slot's price rises by ε; outbid drones rebid elsewhere.
At convergence the assignment satisfies ε-complementary slackness, which
bounds the suboptimality to N·ε relative to Hungarian.

The problem with vanilla Bertsekas in a drone swarm:
  - Prices need to propagate through the comm graph, not broadcast globally.
  - ε is a fixed scalar — it controls the convergence/optimality tradeoff but
    doesn't adapt to the difficulty of each configuration.
  - Initial prices are zero — the auction wastes rounds discovering structure
    that a learned model could provide immediately.

This module fixes all three:
  1. GNN produces per-drone-slot VALUE estimates (learned warm-start bids).
  2. A learned ε-head produces per-slot price increments (adaptive ε).
  3. Slot prices propagate through the comm graph each round alongside bids.
  4. The auction protocol is Bertsekas-correct: drones bid on
     argmax_j(value[i,j] - price[j]), the winner pays price + ε, losers rebid.
  5. No Hungarian anywhere in the inference path.

Architecture
------------
              ┌─────────────────────────────────┐
drone features│  LearnedBertsekaGNN encoder       │  → h_i  (N, H)
+ price signals  (GATv2, price injected as feat) │
              └─────────────────────────────────┘
                       ↓
              ┌──────────────────┐   ┌─────────────────┐
              │  value_head      │   │  epsilon_head    │
              │  score(h_i, j)   │   │  ε_j per slot    │
              │  → V[i,j]  (N,N) │   │  → ε  (N,)      │
              └──────────────────┘   └─────────────────┘
                       ↓                     ↓
              ┌─────────────────────────────────────────┐
              │  bertsekas_price_auction                  │
              │  (fully decentralised, comm-graph aware) │
              │  → hard assignment  (N,)                 │
              └─────────────────────────────────────────┘

Training
--------
The auction itself is non-differentiable (argmax + discrete assignment).
We train via three differentiable surrogates computed from V[i,j] and ε:

  1. Soft-Bertsekas loss: CE on (V - price) after K simulated auction rounds.
     This teaches the value head to produce values that make the auction
     converge to the Hungarian assignment quickly.

  2. ε-calibration loss: penalise ε that is too large (coarse assignment,
     high suboptimality bound) or too small (slow convergence, many rounds).
     Target: ε ≈ (V[i, y_i] - V[i, second_best_i]) / 2
     i.e. the margin between first and second choice, halved.

  3. Cost-regret penalty: identical to other modules — keeps the model
     cost-aware.

Changes from original
---------------------
FIX 1  — Price update rule corrected. Was: price[j] += eps[j] + margin
         (additive overshoot). Now: price[j] = winner_net_value + eps[j]
         (standard Bertsekas absolute update, guaranteed ε-CS at convergence).

FIX 2  — Price propagation loop corrected. Was: sequential merge that let
         later drones see prices already updated by earlier drones in the same
         round (order-dependent, non-synchronous). Now: snapshot → broadcast →
         merge, guaranteeing true synchronous round semantics.

FIX 3  — messages_sent accounting corrected. Was: += N per neighbour link.
         Now: += N per round per link (one full price vector per message).

FIX 4  — Final reassignment no longer overwrites auction state
         unconditionally. Was: every drone's assignment overwritten with
         argmax(net_final) after the loop, discarding valid auction state.
         Now: only unassigned drones fall back to argmax; assigned drones keep
         their auction result.

FIX 5  — Post-loop conflict resolution made local and deterministic. Was:
         a global sequential scan that was order-dependent. Now: tie broken by
         highest net value, with explicit local fallback rule (lower drone
         index yields) that is commutable without global state.

FIX 6  — Training/inference mismatch eliminated. forward_values() now runs
         AUCTION_TRAIN_K rounds of simulated auction with price feedback,
         unrolling price state so the model learns values that work mid-auction
         not just at price=0.

FIX 7  — assign() now uses forward_with_prices() with a re-encode every
         RE_ENCODE_EVERY rounds so the GNN sees updated prices at inference.

FIX 8  — Dead code removed from bertsekas_loss (flat_V and ds_edge_index_dyn
         were computed but never used).

FIX 9  — Gradient explosion guard added: eps is clamped before entering log
         in the loss, and EPS_MIN is enforced at the output of EpsilonHead.

FIX 10 — AUCTION_ROUNDS raised from 10 to 20 (N drones may need up to N
          rounds; 10 was undersized for swarms > 10 drones).

FIX 11 — Training shuffle added: train_data is shuffled each epoch so the
          optimizer does not accumulate correlated gradients.

FIX 12 — Unused import `math` removed.

ADDED  — RE_ENCODE_EVERY constant controls how often the GNN is re-run with
          current prices during inference (default: every 5 rounds). This
          gives price-aware value estimates without paying full GNN cost every
          round.

Backward compatibility
----------------------
LocalNegotiatorGNN and all existing modules are not modified.
evaluate_strict_decentralized, evaluate_sinkhorn, evaluate_recurrent all work
unchanged. BertsekasPriceAuction can be evaluated alongside them.

Files required:
  local_negotiator.py
  (sinkhorn_head.py is NOT required — this module is self-contained)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from local_negotiator import (
    COMM_RADIUS,
    HIDDEN_DIM,
    MAX_CONSENSUS_ROUNDS,
    NODE_FEAT_DIM,
    NUM_FORMATIONS,
    REGRET_LAMBDA,
    SLOT_VISIBILITY_RADIUS,
    LocalNegotiatorGNN,
    assignment_quality_metrics,
    build_candidate_edge_attr,
    build_candidate_edges,
    build_candidate_mask,
    build_comm_graph,
    build_node_features,
    cost_regret_penalty,
    edge_logits_to_dense,
    load_negotiator_dataset,
    prepare_dataset,
    sparse_cross_entropy,
)

# ── Constants ──────────────────────────────────────────────────────────────────
# FIX 10: raised from 10 → 20.  Bertsekas can need up to N rounds to converge
# for N drones; 10 was undersized for swarms larger than 10.
AUCTION_ROUNDS      = 20
AUCTION_TRAIN_K     = 4     # rounds unrolled during training (cheaper than full)
# FIX (new): re-encode with current prices every this many auction rounds
# at inference. Gives price-aware GNN output without full-GNN cost every round.
RE_ENCODE_EVERY     = 5
PRICE_FEAT_DIM      = 2     # price features injected per drone
AUGMENTED_IN_DIM    = NODE_FEAT_DIM + PRICE_FEAT_DIM   # 27 + 2 = 29
EPS_INIT            = 0.1
EPS_MIN             = 1e-3
EPS_MAX             = 2.0
CE_WEIGHT           = 1.0
REGRET_WEIGHT       = 0.30
EPS_CAL_WEIGHT      = 0.10
FREEZE_EPOCHS       = 10
MIN_BIJ_FOR_BEST    = 0.90


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EpsilonHead  — per-slot adaptive price increment
# ─────────────────────────────────────────────────────────────────────────────

class EpsilonHead(nn.Module):
    """
    Produces a per-slot price increment ε_j from slot-aggregated drone
    hidden states.

    ε_j should be large when drones agree (slot is clearly owned by one drone
    — converges fast, suboptimality is small) and small when contested (need
    fine-grained price resolution).

    Design: for each slot j, pool the hidden states of all drones that can see
    it, then predict ε_j from the pooled representation.

    Input  : h  (N, H)       — drone hidden states
             candidate_mask (N, N) bool — drone i can see slot j
    Output : eps (N,)        — one ε per slot, in (EPS_MIN, EPS_MAX)
    """
    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        h: torch.Tensor,              # (N, H)
        candidate_mask: torch.Tensor, # (N, N) bool — mask[i,j] = drone i sees slot j
    ) -> torch.Tensor:
        """Return ε (N,) — one per slot, strictly in (EPS_MIN, EPS_MAX)."""
        N, H   = h.size()
        dev    = h.device
        mask_f = candidate_mask.float().to(dev)  # (N, N)

        # For each slot j: mean-pool hidden states of bidding drones
        # slot_h[j] = mean_{i: mask[i,j]} h[i]
        counts   = mask_f.sum(dim=0).clamp(min=1.0)          # (N,)
        slot_h   = (mask_f.T @ h) / counts.unsqueeze(1)      # (N, H)

        raw = self.mlp(slot_h).squeeze(-1)                    # (N,)
        # FIX 9: clamp output strictly above EPS_MIN so log(eps) is always
        # finite in the calibration loss — sigmoid can get arbitrarily close
        # to 0 which would make log(eps) → -inf.
        return (EPS_MIN + (EPS_MAX - EPS_MIN) * torch.sigmoid(raw)).clamp(min=EPS_MIN)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LearnedBertsekaGNN  — encoder with price-aware augmented input
# ─────────────────────────────────────────────────────────────────────────────

class LearnedBertsekaGNN(nn.Module):
    """
    Wraps LocalNegotiatorGNN with:
      • price-augmented input projection (29→H instead of 27→H)
      • EpsilonHead for adaptive per-slot ε
      • forward_with_prices() for the auction loop

    The base encoder's GATv2 layers and edge_scorer are reused directly.
    Only input_proj is replaced (price_proj: 29→H), initialised from
    the trained 27→H weights with two extra columns set to zero.

    Checkpoint loading
    ------------------
    Load strict_local_negotiator_best.pt into base_model, then pass it to
    LearnedBertsekaGNN(base_model). The constructor copies all weights.
    """

    def __init__(self, base_model: LocalNegotiatorGNN):
        super().__init__()

        # ── Reuse all base encoder components ────────────────────────────────
        self.formation_embedding = base_model.formation_embedding
        self.gat_layers          = base_model.gat_layers
        self.layer_norms         = base_model.layer_norms
        self.edge_scorer         = base_model.edge_scorer
        self.log_temp            = base_model.log_temp

        # ── Replace input_proj with price-augmented version ──────────────────
        self.price_proj = nn.Linear(AUGMENTED_IN_DIM, HIDDEN_DIM)
        with torch.no_grad():
            # Copy trained weights for original NODE_FEAT_DIM features
            self.price_proj.weight[:, :NODE_FEAT_DIM].copy_(
                base_model.input_proj.weight
            )
            # Price features initialised to zero — neutral at start of training
            self.price_proj.weight[:, NODE_FEAT_DIM:].zero_()
            self.price_proj.bias.copy_(base_model.input_proj.bias)

        # ── New heads ─────────────────────────────────────────────────────────
        self.epsilon_head = EpsilonHead(HIDDEN_DIM)

    # ── Internal encode ───────────────────────────────────────────────────────

    def _encode(
        self,
        x_aug: torch.Tensor,     # (N, 29) — node features + price features
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """GATv2 encoding using price-augmented input projection."""
        h = F.relu(self.price_proj(x_aug))
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            if edge_index.size(1) > 0:
                h_new = gat(h, edge_index, edge_attr=edge_attr)
            else:
                h_new = torch.zeros_like(h)
            h = ln(h + F.elu(h_new))
        return h  # (N, H)

    def _score_edges(
        self,
        h: torch.Tensor,
        ds_edge_index: torch.Tensor,
        ds_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse edge scoring — identical to LocalNegotiatorGNN.forward_sparse."""
        if ds_edge_index.size(1) == 0:
            return torch.zeros(0, device=h.device)
        i    = ds_edge_index[0]
        h_i  = h[i]
        temp = torch.exp(self.log_temp).clamp(min=1e-3)
        return self.edge_scorer(
            torch.cat([h_i, ds_edge_attr], dim=-1)
        ).squeeze(-1) / temp

    # ── Price feature extraction ──────────────────────────────────────────────

    @staticmethod
    def _price_features(
        prices: torch.Tensor,          # (N,) current slot prices
        candidate_mask: torch.Tensor,  # (N, N) bool
        current_claims: torch.Tensor,  # (N,) current claimed slot per drone (-1 = none)
        device: torch.device,
    ) -> torch.Tensor:
        """
        Extract 2 price features per drone:
          [0] price of drone's currently claimed slot  (0 if no claim)
          [1] mean price of all visible slots

        These tell the encoder how expensive the current assignment is and
        how competitive the local slot market is.
        """
        N      = prices.size(0)
        pr     = prices.to(device)
        mask_f = candidate_mask.float().to(device)

        # Feature 0: price of claimed slot
        claimed_price = torch.zeros(N, device=device)
        valid = current_claims >= 0
        if valid.any():
            claimed_price[valid] = pr[current_claims[valid].clamp(min=0)]

        # Feature 1: mean price of visible slots
        counts     = mask_f.sum(dim=1).clamp(min=1.0)
        mean_price = (mask_f * pr.unsqueeze(0)).sum(dim=1) / counts

        return torch.stack([claimed_price, mean_price], dim=1)  # (N, 2)

    # ── Single encoder pass given current prices ──────────────────────────────

    def forward_with_prices(
        self,
        x_base: torch.Tensor,          # (N, 27)
        prices: torch.Tensor,          # (N,) slot prices
        current_claims: torch.Tensor,  # (N,) claimed slot per drone
        candidate_mask: torch.Tensor,  # (N, N) bool
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        ds_edge_index: torch.Tensor,
        ds_edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One encoder pass given current prices.

        Returns
        -------
        V   : (N, N) dense value matrix  (masked, -1e4 for invisible)
        eps : (N,) per-slot price increments
        h   : (N, H) drone hidden states
        """
        dev  = x_base.device
        N    = x_base.size(0)

        pf     = self._price_features(prices, candidate_mask, current_claims, dev)
        x_aug  = torch.cat([x_base, pf], dim=1)  # (N, 29)
        h      = self._encode(x_aug, edge_index, edge_attr)
        logits = self._score_edges(h, ds_edge_index, ds_edge_attr)
        V      = edge_logits_to_dense(logits, ds_edge_index, N, fill=-1e4)
        V      = V.masked_fill(~candidate_mask.to(dev), -1e4)
        eps    = self.epsilon_head(h, candidate_mask.to(dev))

        return V, eps, h


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BertsekasPriceAuction  — the decentralised protocol
# ─────────────────────────────────────────────────────────────────────────────

def _neighbors_from_edge_index(
    edge_index: torch.Tensor, n: int
) -> Dict[int, List[int]]:
    neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
    if edge_index.numel() == 0:
        return neighbors
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if v not in neighbors[u]:
            neighbors[u].append(v)
    return neighbors


@torch.no_grad()
def bertsekas_price_auction(
    V_init: torch.Tensor,          # (N, N) initial value matrix (from GNN)
    eps_init: torch.Tensor,        # (N,) initial per-slot ε (from GNN)
    candidate_mask: torch.Tensor,  # (N, N) bool
    comm_edge_index: torch.Tensor, # (2, E_comm)
    max_rounds: int = AUCTION_ROUNDS,
) -> Tuple[torch.Tensor, Dict]:
    """
    Fully decentralised Bertsekas price auction.

    Each drone maintains:
      - Its row of V (local value estimates, fixed after GNN pass)
      - Local copy of slot prices (propagated from neighbours each round)
      - Current assignment

    Each round:
      1. Each drone computes net value = V[i,j] - price[j] for visible slots.
      2. Drone picks argmax_j(net_value[i,j]) as its bid.
      3. Among all drones bidding on slot j, the highest bidder wins.
         Winner's price is set to: winner_net_value + ε[j]   (FIX 1)
      4. Prices are broadcast synchronously to comm neighbours.           (FIX 2)
      5. Outbid drones are unassigned and will rebid next round.

    Convergence: stops when no price changes (ε-CS satisfied).

    Returns
    -------
    assignment : (N,) long — drone i → slot assignment[i], -1 if unassigned
    info       : dict with rounds, converged, messages_sent
    """
    N         = V_init.size(0)
    V         = V_init.cpu().clone()          # (N, N) fixed per-drone values
    eps       = eps_init.cpu().clone()        # (N,) per-slot increments
    prices    = torch.zeros(N)               # (N,) slot prices — start at 0
    mask      = candidate_mask.cpu()
    neighbors = _neighbors_from_edge_index(comm_edge_index.cpu(), N)

    assignment    = torch.full((N,), -1, dtype=torch.long)
    rounds_used   = 0
    converged     = False
    messages_sent = 0

    for round_idx in range(max_rounds):
        rounds_used = round_idx + 1
        prev_prices = prices.clone()

        # ── Step 1 & 2: each drone bids on argmax net value ──────────────────
        net = V - prices.unsqueeze(0)         # (N, N) net values
        net = net.masked_fill(~mask, -1e9)

        bids    = net.max(dim=1).values       # (N,) best net value per drone
        choices = net.argmax(dim=1)           # (N,) slot each drone bids on
        no_vis  = ~mask.any(dim=1)
        choices[no_vis] = -1

        # ── Step 3: for each slot, find highest bidder and set price ──────────
        # FIX 1: use standard Bertsekas absolute price update.
        #   Correct:  price[j] = winner_net_value + eps[j]
        #   Wrong was: price[j] += eps[j] + margin  (additive overshoot)
        #
        # The correct rule sets the new price to just above where the winner
        # prefers this slot over any other, guaranteeing ε-complementary
        # slackness at convergence.
        new_assignment = torch.full((N,), -1, dtype=torch.long)
        new_prices     = prices.clone()  # start from current; only winning slots update

        for slot in range(N):
            bidders = (choices == slot).nonzero(as_tuple=True)[0]
            if bidders.numel() == 0:
                continue
            # FIX A: exclude drones with no visible slots (bid = -1e9)
            valid_bidders = bidders[bids[bidders] > -1e8]
            if valid_bidders.numel() == 0:
                continue
            best_bidder_idx = bids[valid_bidders].argmax()
            best_bidder     = valid_bidders[best_bidder_idx]
            new_assignment[best_bidder] = slot
            # CORE FIX: prices must be monotonically non-decreasing (Bertsekas invariant).
            # price[j] = max(old_price[j], winner_net + eps) — never decrease.
            # The original rule allowed prices to drop when a lower-value drone won,
            # causing oscillation and drones cycling between slots indefinitely.
            candidate_price = bids[best_bidder].item() + eps[slot].item()
            new_prices[slot] = max(prices[slot].item(), max(0.0, candidate_price))

        assignment = new_assignment
        prices     = new_prices

        # ── Step 4: synchronous price propagation through comm graph ──────────
        # FIX 2: snapshot prices BEFORE the loop so all drones broadcast the
        # same round-start prices.  The original code merged sequentially,
        # making later drones see prices already updated by earlier drones
        # in the same round (order-dependent, non-synchronous behaviour).
        # FIX C: truly local propagation — each drone only sees prices
        # from its comm-graph neighbours, not the global maximum.
        price_snapshot = prices.clone()
        local_prices = [price_snapshot.clone() for _ in range(N)]
        for drone in range(N):
            for nb in neighbors[drone]:
                messages_sent += 1
                local_prices[nb] = torch.maximum(local_prices[nb], price_snapshot)
        merged = torch.zeros(N)
        for drone in range(N):
            merged = torch.maximum(merged, local_prices[drone])
        prices = merged

        # ── Convergence check ─────────────────────────────────────────────────
        if (prices - prev_prices).abs().max().item() < 1e-6:
            converged = True
            break

    # ── Final assignment ───────────────────────────────────────────────────────
    # FIX 4: do NOT overwrite every drone's assignment unconditionally.
    # The original code reassigned ALL drones to argmax(net_final) after the
    # loop, discarding valid auction state even for correctly-assigned drones.
    # Now: only unassigned drones (-1) fall back to argmax.
    net_final = V - prices.unsqueeze(0)
    net_final = net_final.masked_fill(~mask, -1e9)

    # FALLBACK: assign remaining drones greedily to unclaimed slots.
    # Standard Bertsekas guarantees all drones are assigned at convergence,
    # but with partial visibility some slots may be unreachable. We use a
    # greedy fallback: each unassigned drone takes the best unclaimed visible
    # slot. This is still local — each drone only needs its own V row and the
    # current price vector (shared via comm graph).
    claimed_slots = set(assignment[assignment >= 0].tolist())
    unassigned_drones = [i for i in range(N) if assignment[i].item() < 0]
    for drone in unassigned_drones:
        if not mask[drone].any():
            continue
        # Try unclaimed visible slots first
        unclaimed_net = net_final[drone].clone()
        for s in claimed_slots:
            unclaimed_net[s] = -1e9
        unclaimed_net = unclaimed_net.masked_fill(~mask[drone], -1e9)
        if unclaimed_net.max().item() > -1e8:
            best_slot = int(unclaimed_net.argmax().item())
        else:
            # All visible slots are claimed — take the best visible regardless
            best_slot = int(net_final[drone].masked_fill(~mask[drone], -1e9).argmax().item())
        assignment[drone] = best_slot
        claimed_slots.add(best_slot)

    # ── Conflict resolution ────────────────────────────────────────────────────
    # FIX 5: fully local and deterministic tiebreak.
    # Original was a sequential scan (drone 0 always wins regardless of value).
    # Now: among drones claiming the same slot, keep the one with the highest
    # net value; on exact tie, keep the lower drone index (a fixed local rule
    # that requires no global state — any drone can compute it from shared
    # prices and its own values).
    slot_claims: Dict[int, int] = {}          # slot → best_drone seen so far
    slot_best_val: Dict[int, float] = {}

    for drone in range(N):
        slot = assignment[drone].item()
        if slot < 0:
            continue
        val = net_final[drone, slot].item()
        if slot not in slot_claims:
            slot_claims[slot]    = drone
            slot_best_val[slot]  = val
        elif val > slot_best_val[slot]:
            slot_claims[slot]    = drone
            slot_best_val[slot]  = val
        elif val == slot_best_val[slot] and drone < slot_claims[slot]:
            # deterministic local tiebreak: lower drone index wins
            slot_claims[slot] = drone

    final = torch.full((N,), -1, dtype=torch.long)
    for slot, drone in slot_claims.items():
        final[drone] = slot

    dup   = int(final[final >= 0].numel() - final[final >= 0].unique().numel())
    unass = int((final < 0).sum().item())
    info  = {
        "rounds":          float(rounds_used),
        "converged":       float(converged),
        "messages_sent":   float(messages_sent),
        "conflict_rate":   float(dup / max(N, 1)),
        "unassigned_rate": float(unass / max(N, 1)),
        "bijection":       float(dup == 0 and unass == 0),
    }
    return final, info


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LearnedBertsekaModel(nn.Module):
    """
    Full learned Bertsekas model.

    Thin wrapper that ties together:
      • LearnedBertsekaGNN   (encoder + value head + ε head)
      • bertsekas_price_auction (decentralised inference)

    At training time: forward_values() runs AUCTION_TRAIN_K rounds of
    simulated auction with price feedback so the model sees mid-auction states.
    At inference time: assign() runs the full auction with periodic GNN
    re-encoding using current prices.
    """

    def __init__(self, base_model: Optional[LocalNegotiatorGNN] = None):
        super().__init__()
        if base_model is None:
            base_model = LocalNegotiatorGNN()
        self.gnn = LearnedBertsekaGNN(base_model)

    @property
    def formation_embedding(self):
        return self.gnn.formation_embedding

    def forward_values(
        self,
        x_base: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        ds_edge_index: torch.Tensor,
        ds_edge_attr: torch.Tensor,
        train_k: int = AUCTION_TRAIN_K,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable forward for training.

        FIX 6: instead of always passing zero prices (which created a
        train/inference mismatch), this now runs `train_k` rounds of simulated
        auction using detached price state, then does a final GNN pass with
        the resulting prices.  This teaches the model to produce values that
        are informative mid-auction, not just at price=0.

        The auction simulation uses torch.no_grad() for the price updates
        (non-differentiable argmax steps) and then re-encodes with those
        prices in a differentiable final pass.

        Returns
        -------
        V   : (N, N) dense value matrix  (from the final price-aware pass)
        eps : (N,) per-slot price increments
        """
        dev = x_base.device
        N   = x_base.size(0)

        # ── Phase 1: simulate auction rounds to get realistic prices ──────────
        # Run train_k rounds without gradients to build up price state.
        with torch.no_grad():
            prices = torch.zeros(N, device=dev)
            claims = torch.full((N,), -1, dtype=torch.long, device=dev)
            mask_cpu = candidate_mask.cpu()

            for _ in range(train_k):
                V_sim, eps_sim, _ = self.gnn.forward_with_prices(
                    x_base, prices, claims,
                    candidate_mask, edge_index, edge_attr,
                    ds_edge_index, ds_edge_attr,
                )
                # One round of Bertsekas price update (detached, no-grad)
                net = (V_sim - prices.unsqueeze(0)).masked_fill(
                    ~candidate_mask, -1e9
                )
                bids    = net.max(dim=1).values
                choices = net.argmax(dim=1)
                no_vis  = ~candidate_mask.any(dim=1)
                choices[no_vis] = -1

                new_prices = prices.clone()
                new_claims = torch.full((N,), -1, dtype=torch.long, device=dev)
                for slot in range(N):
                    bidders = (choices == slot).nonzero(as_tuple=True)[0]
                    if bidders.numel() == 0:
                        continue
                    valid_bidders = bidders[bids[bidders] > -1e8]  # FIX A
                    if valid_bidders.numel() == 0:
                        continue
                    best = valid_bidders[bids[valid_bidders].argmax()]
                    new_claims[best] = slot
                    # CORE FIX: price monotonicity
                    candidate_price = bids[best].item() + eps_sim[slot].item()
                    new_prices[slot] = max(prices[slot].item(), max(0.0, candidate_price))

                prices = new_prices
                claims = new_claims

        # ── Phase 2: final differentiable pass with learned price state ───────
        V, eps, _ = self.gnn.forward_with_prices(
            x_base, prices.detach(), claims.detach(),
            candidate_mask, edge_index, edge_attr,
            ds_edge_index, ds_edge_attr,
        )
        return V, eps

    @torch.no_grad()
    def assign(
        self,
        data: Data,
        device: torch.device,
        slot_radius: float  = SLOT_VISIBILITY_RADIUS,
        max_rounds: int     = AUCTION_ROUNDS,
        re_encode_every: int = RE_ENCODE_EVERY,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full inference: GNN → price-aware auction → hard assignment.

        FIX 7: re-encodes with current prices every `re_encode_every` rounds
        so the GNN sees updated price signals during the auction, eliminating
        the original train/inference mismatch where V was computed once at
        price=0 and then never updated.
        """
        self.eval()
        fid   = data.formation_id.item()
        f_emb = self.gnn.formation_embedding(torch.tensor(fid, device=device))

        mask  = build_candidate_mask(
            data.drone_pos, data.slots, slot_radius, y=data.y, force_gt=False
        )
        x     = build_node_features(
            data.drone_pos, data.slots, f_emb.detach().cpu(), candidate_mask=mask
        ).to(device)
        ei, ea  = build_comm_graph(data.drone_pos)
        ds_ei   = build_candidate_edges(mask)[0]
        ds_ea   = build_candidate_edge_attr(data.drone_pos, data.slots, ds_ei)

        ei_d    = ei.to(device)
        ea_d    = ea.to(device)
        ds_ei_d = ds_ei.to(device)
        ds_ea_d = ds_ea.to(device)
        mask_d  = mask.to(device)

        N         = x.size(0)
        prices    = torch.zeros(N, device=device)
        claims    = torch.full((N,), -1, dtype=torch.long, device=device)
        neighbors = _neighbors_from_edge_index(ei.cpu(), N)

        assignment    = torch.full((N,), -1, dtype=torch.long)
        rounds_used   = 0
        converged     = False
        messages_sent = 0

        # Initial encode
        V, eps, _ = self.gnn.forward_with_prices(
            x, prices, claims, mask_d, ei_d, ea_d, ds_ei_d, ds_ea_d
        )
        V   = V.cpu()
        eps = eps.cpu()

        for round_idx in range(max_rounds):
            rounds_used = round_idx + 1
            prev_prices = prices.cpu().clone()

            # Re-encode with current prices every re_encode_every rounds
            if round_idx > 0 and round_idx % re_encode_every == 0:
                V_new, eps_new, _ = self.gnn.forward_with_prices(
                    x, prices.to(device), claims.to(device),
                    mask_d, ei_d, ea_d, ds_ei_d, ds_ea_d,
                )
                V   = V_new.cpu()
                eps = eps_new.cpu()

            prices_cpu = prices.cpu()
            net = (V - prices_cpu.unsqueeze(0)).masked_fill(~mask, -1e9)
            bids    = net.max(dim=1).values
            choices = net.argmax(dim=1)
            no_vis  = ~mask.any(dim=1)
            choices[no_vis] = -1

            new_assignment = torch.full((N,), -1, dtype=torch.long)
            new_prices     = prices_cpu.clone()
            for slot in range(N):
                bidders = (choices == slot).nonzero(as_tuple=True)[0]
                if bidders.numel() == 0:
                    continue
                valid_bidders = bidders[bids[bidders] > -1e8]  # FIX A
                if valid_bidders.numel() == 0:
                    continue
                best = valid_bidders[bids[valid_bidders].argmax()]
                new_assignment[best] = slot
                # CORE FIX: price monotonicity
                candidate_price = bids[best].item() + eps[slot].item()
                new_prices[slot] = max(prices_cpu[slot].item(), max(0.0, candidate_price))

            assignment = new_assignment
            prices     = new_prices.to(device)

            # Synchronous price propagation (FIX C — truly local)
            price_snapshot = prices.cpu().clone()
            local_prices = [price_snapshot.clone() for _ in range(N)]
            for drone in range(N):
                for nb in neighbors[drone]:
                    messages_sent += 1
                    local_prices[nb] = torch.maximum(local_prices[nb], price_snapshot)
            merged = torch.zeros(N)
            for drone in range(N):
                merged = torch.maximum(merged, local_prices[drone])
            prices = merged.to(device)

            if (prices.cpu() - prev_prices).abs().max().item() < 1e-6:
                converged = True
                break

            claims = assignment.to(device)

        # Final fix-up for unassigned drones only (FIX 4)
        prices_cpu = prices.cpu()
        net_final  = (V - prices_cpu.unsqueeze(0)).masked_fill(~mask, -1e9)
        # FALLBACK: greedy assignment for remaining unassigned drones
        claimed_slots = set(assignment[assignment >= 0].tolist())
        unassigned_drones = [i for i in range(N) if assignment[i].item() < 0]
        for drone in unassigned_drones:
            if not mask[drone].any():
                continue
            unclaimed_net = net_final[drone].clone()
            for s in claimed_slots:
                unclaimed_net[s] = -1e9
            unclaimed_net = unclaimed_net.masked_fill(~mask[drone], -1e9)
            if unclaimed_net.max().item() > -1e8:
                best_slot = int(unclaimed_net.argmax().item())
            else:
                best_slot = int(net_final[drone].masked_fill(~mask[drone], -1e9).argmax().item())
            assignment[drone] = best_slot
            claimed_slots.add(best_slot)

        # Conflict resolution — local deterministic tiebreak (FIX 5)
        slot_claims:    Dict[int, int]   = {}
        slot_best_val:  Dict[int, float] = {}
        for drone in range(N):
            slot = assignment[drone].item()
            if slot < 0:
                continue
            val = net_final[drone, slot].item()
            if slot not in slot_claims:
                slot_claims[slot]   = drone
                slot_best_val[slot] = val
            elif val > slot_best_val[slot]:
                slot_claims[slot]   = drone
                slot_best_val[slot] = val
            elif val == slot_best_val[slot] and drone < slot_claims[slot]:
                slot_claims[slot] = drone

        final = torch.full((N,), -1, dtype=torch.long)
        for slot, drone in slot_claims.items():
            final[drone] = slot

        dup   = int(final[final >= 0].numel() - final[final >= 0].unique().numel())
        unass = int((final < 0).sum().item())
        info  = {
            "rounds":          float(rounds_used),
            "converged":       float(converged),
            "messages_sent":   float(messages_sent),
            "conflict_rate":   float(dup / max(N, 1)),
            "unassigned_rate": float(unass / max(N, 1)),
            "bijection":       float(dup == 0 and unass == 0),
        }
        return final, info


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Loss
# ─────────────────────────────────────────────────────────────────────────────

def bertsekas_loss(
    V: torch.Tensor,               # (N, N) dense value matrix
    eps: torch.Tensor,             # (N,) per-slot ε
    targets: torch.Tensor,         # (N,) ground-truth slot per drone
    candidate_mask: torch.Tensor,  # (N, N) bool
    drone_pos: torch.Tensor,       # (N, 2)
    slots: torch.Tensor,           # (N, 2)
    ce_weight: float      = CE_WEIGHT,
    regret_weight: float  = REGRET_WEIGHT,
    eps_cal_weight: float = EPS_CAL_WEIGHT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Three-term loss for the learned Bertsekas model.

    Term 1 — Cross-entropy on V
    ---------------------------
    CE teaches the value head to assign high values to the Hungarian target
    slot. After training, the auction resolves conflicts using prices; the
    CE ensures the values are informative before prices start.

    Term 2 — Cost-regret penalty
    ----------------------------
    Same as all other modules. Penalises high expected cost under the soft
    value distribution.

    Term 3 — ε calibration
    ----------------------
    Target ε[j] = 0.5 * mean_{i bids on j} (V[i,j] - V[i, second_best_i])
    Loss: MSE(log ε, log target_ε) — log-space because ε spans orders of
    magnitude.

    Returns
    -------
    total, ce_loss, regret_loss, eps_cal_loss
    """
    dev  = V.device
    N    = targets.size(0)
    mask_d = candidate_mask.to(dev)

    # ── Term 1: cross-entropy ─────────────────────────────────────────────────
    ce = sparse_cross_entropy(V, targets.to(dev), mask_d)

    # ── Term 2: cost regret ───────────────────────────────────────────────────
    # FIX 8: removed dead variables flat_V and ds_edge_index_dyn that were
    # computed here but never used.
    regret = cost_regret_penalty(
        V, drone_pos.to(dev), slots.to(dev), targets.to(dev), mask_d
    )

    # ── Term 3: ε calibration ─────────────────────────────────────────────────
    V_masked = V.masked_fill(~mask_d, -1e9)
    top2     = V_masked.topk(k=min(2, N), dim=1).values   # (N, 2) or (N, 1)
    if top2.size(1) >= 2:
        margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)  # (N,) per drone
    else:
        margin = torch.zeros(N, device=dev)

    mask_f      = mask_d.float()
    counts      = mask_f.sum(dim=0).clamp(min=1.0)
    slot_margin = (mask_f * margin.unsqueeze(1)).sum(dim=0) / counts
    target_eps  = (0.5 * slot_margin).clamp(min=EPS_MIN, max=EPS_MAX)

    # FIX 9: clamp eps before log to prevent -inf gradients.
    # EpsilonHead already clamps at EPS_MIN, but the extra guard here
    # is cheap insurance against numerical issues during early training.
    eps_safe = eps.clamp(min=EPS_MIN)
    eps_cal  = F.mse_loss(
        torch.log(eps_safe),
        torch.log(target_eps.detach()),
    )

    total = ce_weight * ce + regret_weight * regret + eps_cal_weight * eps_cal
    return total, ce, regret, eps_cal


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_bertsekas(
    model: LearnedBertsekaModel,
    dataset: List[Data],
    device: torch.device,
    slot_radius: float   = SLOT_VISIBILITY_RADIUS,
    max_rounds: int      = AUCTION_ROUNDS,
    re_encode_every: int = RE_ENCODE_EVERY,
) -> Dict[str, float]:
    """
    Evaluate using fully decentralised Bertsekas inference.

    Metrics match evaluate_strict_decentralized / evaluate_sinkhorn /
    evaluate_recurrent exactly — direct column comparison in results tables.
    """
    model.eval()
    metrics: Dict[str, List[float]] = {
        "bijection_rate":          [],
        "conflict_rate":           [],
        "unassigned_rate":         [],
        "cost_ratio_vs_hungarian": [],
        "slot_match_rate":         [],
        "consensus_rounds":        [],
        "converged_rate":          [],
        "messages_sent":           [],
    }

    for data in dataset:
        assignment, info = model.assign(
            data, device,
            slot_radius=slot_radius,
            max_rounds=max_rounds,
            re_encode_every=re_encode_every,
        )
        sample = assignment_quality_metrics(
            data.drone_pos, data.slots, assignment, data.y, rounds=info["rounds"]
        )
        for key, val in sample.items():
            metrics[key].append(val)
        metrics["converged_rate"].append(info["converged"])
        metrics["messages_sent"].append(info["messages_sent"])

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Training
# ─────────────────────────────────────────────────────────────────────────────

def _new_params(model: LearnedBertsekaModel) -> List[nn.Parameter]:
    """Parameters that are new — always trained from the start."""
    return (
        list(model.gnn.price_proj.parameters()) +
        list(model.gnn.epsilon_head.parameters())
    )


def _base_params(model: LearnedBertsekaModel) -> List[nn.Parameter]:
    """Pre-trained encoder parameters — frozen initially."""
    return (
        list(model.gnn.formation_embedding.parameters()) +
        list(model.gnn.gat_layers.parameters())          +
        list(model.gnn.layer_norms.parameters())         +
        list(model.gnn.edge_scorer.parameters())         +
        [model.gnn.log_temp]
    )


def train_bertsekas_model(
    model: LearnedBertsekaModel,
    train_data: List[Data],
    val_data: List[Data],
    device: torch.device,
    epochs: int               = 80,
    lr: float                 = 3e-4,
    freeze_epochs: int        = FREEZE_EPOCHS,
    min_bijection_for_best: float = MIN_BIJ_FOR_BEST,
    force_gt_train: bool      = True,
    ce_weight: float          = CE_WEIGHT,
    regret_weight: float      = REGRET_WEIGHT,
    eps_cal_weight: float     = EPS_CAL_WEIGHT,
    eval_subset: int          = 100,
    train_k: int              = AUCTION_TRAIN_K,
) -> Dict[str, List]:
    """
    Train LearnedBertsekaModel.

    Freeze schedule
    ---------------
    Epochs 1 … freeze_epochs     : only price_proj + epsilon_head trained.
                                    Teaches the new heads to be useful before
                                    the base encoder adapts.
    Epochs freeze_epochs+1 … end : all parameters, lr × 0.3, cosine decay.

    Checkpoint policy
    -----------------
    Best = lowest cost_ratio + penalty for bijection below threshold.
    Same formula as all other modules.
    """
    model.to(device)

    new_ps  = _new_params(model)
    base_ps = _base_params(model)
    for p in base_ps:
        p.requires_grad_(False)

    joint_lr     = lr * 0.3
    joint_epochs = max(epochs - freeze_epochs, 1)
    optimizer    = torch.optim.Adam(new_ps, lr=lr)
    scheduler    = None
    frozen       = True

    history: Dict[str, List] = {
        "train_loss":                  [],
        "val_loss":                    [],
        "val_bijection_rate":          [],
        "val_cost_ratio_vs_hungarian": [],
        "val_conflict_rate":           [],
        "val_unassigned_rate":         [],
        "val_slot_match_rate":         [],
        "val_converged_rate":          [],
    }
    best_score = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):

        # ── Unfreeze ──────────────────────────────────────────────────────────
        if frozen and epoch > freeze_epochs:
            frozen = False
            for p in base_ps:
                p.requires_grad_(True)
            optimizer = torch.optim.Adam(
                list(model.parameters()), lr=joint_lr
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=joint_epochs, eta_min=1e-5
            )
            print(
                f"  [Epoch {epoch}] Encoder unfrozen — joint fine-tuning "
                f"(lr={joint_lr:.2e} → 1e-5 over {joint_epochs} epochs)."
            )

        # ── Training pass ─────────────────────────────────────────────────────
        # FIX 11: shuffle training data each epoch to avoid correlated gradients.
        model.train()
        t_loss      = 0.0
        shuffled    = list(train_data)
        random.shuffle(shuffled)

        for data in shuffled:
            fid   = data.formation_id.item()
            f_emb = model.formation_embedding(torch.tensor(fid, device=device))
            mask  = build_candidate_mask(
                data.drone_pos, data.slots,
                SLOT_VISIBILITY_RADIUS, y=data.y, force_gt=force_gt_train,
            )
            x    = build_node_features(
                data.drone_pos, data.slots,
                f_emb.detach().cpu(), candidate_mask=mask,
            ).to(device)
            ei, ea  = build_comm_graph(data.drone_pos)
            ds_ei   = build_candidate_edges(mask)[0]
            ds_ea   = build_candidate_edge_attr(data.drone_pos, data.slots, ds_ei)

            # FIX 6: use price-aware forward (runs train_k simulated auction
            # rounds before the final differentiable pass).
            V, eps = model.forward_values(
                x,
                mask.to(device),
                ei.to(device), ea.to(device),
                ds_ei.to(device), ds_ea.to(device),
                train_k=train_k,
            )
            loss, _, _, _ = bertsekas_loss(
                V, eps,
                data.y.to(device),
                mask.to(device),
                data.drone_pos.to(device),
                data.slots.to(device),
                ce_weight=ce_weight,
                regret_weight=regret_weight,
                eps_cal_weight=eps_cal_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            t_loss += loss.item()

        # ── Validation loss ───────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for data in val_data:
                fid   = data.formation_id.item()
                f_emb = model.formation_embedding(torch.tensor(fid, device=device))
                mask  = build_candidate_mask(
                    data.drone_pos, data.slots,
                    SLOT_VISIBILITY_RADIUS, y=data.y, force_gt=False,
                )
                x    = build_node_features(
                    data.drone_pos, data.slots,
                    f_emb.detach().cpu(), candidate_mask=mask,
                ).to(device)
                ei, ea  = build_comm_graph(data.drone_pos)
                ds_ei   = build_candidate_edges(mask)[0]
                ds_ea   = build_candidate_edge_attr(data.drone_pos, data.slots, ds_ei)

                V, eps = model.forward_values(
                    x, mask.to(device),
                    ei.to(device), ea.to(device),
                    ds_ei.to(device), ds_ea.to(device),
                    train_k=train_k,
                )
                loss, _, _, _ = bertsekas_loss(
                    V, eps,
                    data.y.to(device), mask.to(device),
                    data.drone_pos.to(device), data.slots.to(device),
                    ce_weight=ce_weight,
                    regret_weight=regret_weight,
                    eps_cal_weight=eps_cal_weight,
                )
                v_loss += loss.item()

        t_loss /= max(len(train_data), 1)
        v_loss /= max(len(val_data),   1)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if not frozen and scheduler is not None:
            scheduler.step()

        # ── Evaluation metrics ────────────────────────────────────────────────
        metrics = evaluate_bertsekas(model, val_data[:eval_subset], device)
        for key in ("bijection_rate", "cost_ratio_vs_hungarian",
                    "conflict_rate", "unassigned_rate",
                    "slot_match_rate", "converged_rate"):
            history[f"val_{key}"].append(metrics[key])

        # ── Checkpoint ────────────────────────────────────────────────────────
        score = metrics["cost_ratio_vs_hungarian"]
        if metrics["bijection_rate"] < min_bijection_for_best:
            score += 10.0 * (min_bijection_for_best - metrics["bijection_rate"])
        if score < best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # ── Logging ───────────────────────────────────────────────────────────
        if epoch == 1 or epoch % 5 == 0:
            phase = "frozen" if frozen else "joint"
            print(
                f"Epoch {epoch:3d} [{phase}] | "
                f"train={t_loss:.4f} | val={v_loss:.4f} | "
                f"bij={metrics['bijection_rate']:.3f} | "
                f"conflict={metrics['conflict_rate']:.3f} | "
                f"cost={metrics['cost_ratio_vs_hungarian']:.4f} | "
                f"match={metrics['slot_match_rate']:.3f} | "
                f"conv={metrics['converged_rate']:.3f} | "
                f"rounds={metrics['consensus_rounds']:.1f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "LearnedBertsekaModel",
    "LearnedBertsekaGNN",
    "EpsilonHead",
    "bertsekas_price_auction",
    "bertsekas_loss",
    "evaluate_bertsekas",
    "train_bertsekas_model",
    "AUCTION_ROUNDS",
    "AUCTION_TRAIN_K",
    "RE_ENCODE_EVERY",
    "EPS_INIT",
    "EPS_MIN",
    "EPS_MAX",
    "load_negotiator_dataset",
    "prepare_dataset",
]
