"""
recurrent_consensus.py
======================
Idea 4: Unrolled Recurrent Consensus GNN for drone-to-slot assignment.

Wraps the existing LocalNegotiatorGNN + SinkhornHead into a K-round loop
where each round's soft assignment beliefs P_r feed back as additional node
features into round r+1. The model learns not just *what* assignment to make
but *how to converge* — early rounds form rough beliefs, later rounds refine
them using neighbour information.

Architecture
------------
Round r:
  1. Build node features: original x  +  belief_features(P_{r-1})  +  round_emb(r)
  2. encode_drones(x_r, comm_edge_index, comm_edge_attr)  → h_r   (N, H)
  3. forward_sparse(h_r, ds_edge_index, ds_edge_attr)      → logits_r (E,)
  4. SinkhornHead(dense(logits_r), mask)                   → P_r   (N, N)

Round 0 uses P_{-1} = uniform over visible slots (uninformed prior).

Belief features injected at each round (per drone i):
  • expected_dx, expected_dy  — soft expected slot position under P_{r-1}
  • top1_prob                 — max probability in P_{r-1}[i, :]
  • entropy                   — normalised entropy of P_{r-1}[i, :]
  → 4 features, concatenated with original x (27-d) and round_emb (8-d) = 39-d input

What this adds over Sinkhorn alone
------------------------------------
Sinkhorn is a static function of a single encoder pass — it resolves conflicts
in the *output* but the encoder cannot update its beliefs based on what other
drones are doing. Recurrent rounds let drone i see "my neighbours' round-1 P
heavily favours slot 3 — I should shift away from slot 3" before making a
final decision. This is the learned equivalent of a distributed auction.

Backward compatibility
----------------------
LocalNegotiatorGNN and SinkhornHead are not modified.
evaluate_strict_decentralized and train_strict_decentralized_model work unchanged.
The gossip inference path is untouched.

Files required in the same directory:
  local_negotiator.py   (fixed version, COMM_RADIUS=4.5)
  sinkhorn_head.py      (SinkhornHead + sinkhorn_assignment_loss)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data

from local_negotiator import (
    COMM_RADIUS,
    NUM_FORMATIONS,
    SLOT_VISIBILITY_RADIUS,
    MAX_CONSENSUS_ROUNDS,
    HIDDEN_DIM,
    NODE_FEAT_DIM,
    LocalNegotiatorGNN,
    assignment_quality_metrics,
    build_candidate_mask,
    build_candidate_edges,
    build_candidate_edge_attr,
    build_comm_graph,
    build_node_features,
    edge_logits_to_dense,
    load_negotiator_dataset,
    prepare_dataset,
)
from sinkhorn_head import (
    SinkhornHead,
    SINKHORN_ITERS,
    SINKHORN_CE_W,
    SINKHORN_REGRET_W,
    sinkhorn_assignment_loss,
)

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_ROUNDS       = 4     # K rounds unrolled during training
ROUND_EMB_DIM    = 8     # learnable round-index embedding dimension
BELIEF_FEAT_DIM  = 4     # features extracted from P_{r-1} per drone
# Total recurrent input dim = NODE_FEAT_DIM + BELIEF_FEAT_DIM + ROUND_EMB_DIM
#                           = 27 + 4 + 8 = 39
RECURRENT_IN_DIM = NODE_FEAT_DIM + BELIEF_FEAT_DIM + ROUND_EMB_DIM

# Loss weights per round — later rounds weighted higher (linear ramp)
# round 0 gets weight 1, round K-1 gets weight K; normalised to sum to 1
def _round_weights(K: int) -> List[float]:
    raw = [float(r + 1) for r in range(K)]
    total = sum(raw)
    return [w / total for w in raw]

# Freeze schedule
RECURRENT_FREEZE_EPOCHS = 10   # freeze base encoder; train only new components
MAX_ROUNDS_ANNEAL_START = 2    # start training with this many rounds
MAX_ROUNDS_ANNEAL_END   = NUM_ROUNDS  # anneal up to this by epoch 20


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RoundEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class RoundEmbedding(nn.Module):
    """
    Learnable round-index embedding.

    Maps an integer round index r ∈ {0, …, max_rounds-1} to a dense vector.
    This tells the GNN which round it's in so it can behave differently:
    early rounds should explore broadly, late rounds should commit.

    Supports up to max_rounds=12 (beyond MAX_CONSENSUS_ROUNDS).
    """
    def __init__(self, emb_dim: int = ROUND_EMB_DIM, max_rounds: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(max_rounds, emb_dim)

    def forward(self, r: int, n: int, device: torch.device) -> torch.Tensor:
        """Return (N, emb_dim) — same embedding broadcast to all drones."""
        idx = torch.tensor(r, dtype=torch.long, device=device)
        return self.embedding(idx).unsqueeze(0).expand(n, -1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Belief feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _belief_features(
    P: torch.Tensor,          # (N, N) doubly-stochastic
    slots: torch.Tensor,      # (N, 2)
    candidate_mask: torch.Tensor,  # (N, N) bool
) -> torch.Tensor:
    """
    Extract 4 belief features per drone from soft assignment matrix P.

    Features:
    ---------
    expected_dx, expected_dy : soft expected slot position relative to swarm CoG
        = sum_j P[i,j] * slots[j]  — tells the model where drone i "thinks"
          it's going, in a continuous way.

    top1_prob : max(P[i, :]) — confidence. Near 1 means drone i has committed
        to a slot. Near 1/N means it's still uncertain.

    entropy : -sum_j P[i,j] * log(P[i,j]+eps) / log(N_visible_i)
        Normalised entropy of drone i's belief. 0 = fully committed, 1 = uniform.
        This signals to neighbours whether drone i is a contested or settled agent.

    All four are scale-normalised so they don't dominate the original features.
    """
    dev = P.device
    N   = P.size(0)
    sl  = slots.to(dev)

    # Expected slot position (soft)
    expected_pos = P @ sl                          # (N, 2)
    cog          = sl.mean(dim=0, keepdim=True)
    expected_rel = expected_pos - cog              # (N, 2) — relative to formation centre

    # Top-1 probability (confidence)
    top1_prob = P.max(dim=1).values.unsqueeze(1)   # (N, 1)

    # Normalised entropy
    n_visible  = candidate_mask.float().sum(dim=1).clamp(min=1)  # (N,)
    log_P      = torch.log(P.clamp(min=1e-9))
    raw_ent    = -(P * log_P).sum(dim=1)           # (N,)
    norm_ent   = (raw_ent / torch.log(n_visible.clamp(min=2))).unsqueeze(1)  # (N, 1)
    norm_ent   = torch.nan_to_num(norm_ent, nan=0.0).clamp(0.0, 1.0)

    return torch.cat([expected_rel, top1_prob, norm_ent], dim=1)  # (N, 4)


def _uniform_belief(
    candidate_mask: torch.Tensor,  # (N, N) bool
    device: torch.device,
) -> torch.Tensor:
    """
    Initial belief P_0: uniform distribution over each drone's visible slots.
    Used as the prior before round 0 runs.
    """
    N = candidate_mask.size(0)
    mask_f = candidate_mask.float().to(device)
    n_vis  = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
    return mask_f / n_vis   # (N, N) — rows sum to 1, NOT doubly-stochastic yet


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RecurrentConsensusGNN
# ─────────────────────────────────────────────────────────────────────────────

class RecurrentConsensusGNN(nn.Module):
    """
    K-round recurrent consensus model.

    Wraps LocalNegotiatorGNN (encoder) + SinkhornHead (output) in a loop.
    Adds:
      • RoundEmbedding  — injects which round the model is in
      • belief_proj     — projects belief features (4-d) into hidden space
      • recurrent_proj  — fuses original x (27-d) + belief (4-d) + round (8-d)
                          into the hidden_dim expected by encode_drones

    The base encoder's input_proj (27→64) is replaced by recurrent_proj (39→64)
    for the recurrent forward pass. The original input_proj is preserved and
    used when loading the base checkpoint (the weights map to recurrent_proj
    via the first 27 columns, which are initialised from the checkpoint).

    Checkpoint loading
    ------------------
    Load strict_local_negotiator_best.pt into base_model first, then call
    RecurrentConsensusGNN(base_model, sinkhorn_head). The constructor copies
    all encoder weights and initialises recurrent_proj from input_proj.
    """

    def __init__(
        self,
        base_model: LocalNegotiatorGNN,
        sinkhorn_head: SinkhornHead,
        num_rounds: int = NUM_ROUNDS,
        round_emb_dim: int = ROUND_EMB_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.num_rounds    = num_rounds
        self.hidden_dim    = hidden_dim

        # ── Reuse base encoder components (weights loaded from checkpoint) ──
        self.formation_embedding = base_model.formation_embedding
        self.gat_layers          = base_model.gat_layers
        self.layer_norms         = base_model.layer_norms
        self.edge_scorer         = base_model.edge_scorer
        self.log_temp            = base_model.log_temp

        # ── New components (randomly initialised, trained from scratch) ─────
        self.round_embedding = RoundEmbedding(round_emb_dim, max_rounds=12)

        # Replaces input_proj (27→64) for the recurrent path (39→64)
        # Initialise the first 27 columns from the trained input_proj weights
        # so the encoder starts from a meaningful state, not random.
        recurrent_in = NODE_FEAT_DIM + BELIEF_FEAT_DIM + round_emb_dim
        self.recurrent_proj = nn.Linear(recurrent_in, hidden_dim)
        with torch.no_grad():
            # Copy trained weights for original features
            self.recurrent_proj.weight[:, :NODE_FEAT_DIM].copy_(
                base_model.input_proj.weight
            )
            # New dimensions (belief + round) initialised small
            nn.init.xavier_uniform_(
                self.recurrent_proj.weight[:, NODE_FEAT_DIM:],
                gain=0.1,
            )
            self.recurrent_proj.bias.copy_(base_model.input_proj.bias)

        self.sinkhorn_head = sinkhorn_head

    # ── Internal encode (replaces encode_drones for recurrent path) ──────────

    def _encode(
        self,
        x_r: torch.Tensor,       # (N, 39) — augmented features for round r
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """GATv2 encoding with recurrent_proj instead of input_proj."""
        h = F.relu(self.recurrent_proj(x_r))
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            if edge_index.size(1) > 0:
                h_new = gat(h, edge_index, edge_attr=edge_attr)
            else:
                h_new = torch.zeros_like(h)
            h = ln(h + F.elu(h_new))
        return h   # (N, H)

    def _score_edges(
        self,
        h: torch.Tensor,
        ds_edge_index: torch.Tensor,
        ds_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Same as forward_sparse — reused directly."""
        if ds_edge_index.size(1) == 0:
            return torch.zeros(0, device=h.device)
        i      = ds_edge_index[0]
        h_i    = h[i]
        temp   = torch.exp(self.log_temp).clamp(min=1e-3)
        return self.edge_scorer(
            torch.cat([h_i, ds_edge_attr], dim=-1)
        ).squeeze(-1) / temp

    # ── Single round ──────────────────────────────────────────────────────────

    def _one_round(
        self,
        x_base: torch.Tensor,        # (N, 27) — original node features
        P_prev: torch.Tensor,         # (N, N)  — belief from previous round
        slots: torch.Tensor,          # (N, 2)
        candidate_mask: torch.Tensor, # (N, N) bool
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        ds_edge_index: torch.Tensor,
        ds_edge_attr: torch.Tensor,
        round_idx: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One recurrent round. Returns (P_r, edge_logits_r).
        """
        N = x_base.size(0)

        # Extract belief features from previous round's P
        b_feats  = _belief_features(P_prev, slots, candidate_mask)  # (N, 4)

        # Round embedding
        r_emb    = self.round_embedding(round_idx, N, device)        # (N, 8)

        # Augmented input for this round
        x_r      = torch.cat([x_base, b_feats, r_emb], dim=1)       # (N, 39)

        # Encode + score
        h_r      = self._encode(x_r, edge_index, edge_attr)
        logits_r = self._score_edges(h_r, ds_edge_index, ds_edge_attr)

        # Dense logits → Sinkhorn
        n        = candidate_mask.size(0)
        dense_r  = edge_logits_to_dense(logits_r, ds_edge_index, n, fill=-1e4)
        dense_r  = dense_r.masked_fill(~candidate_mask, -1e4)
        P_r      = self.sinkhorn_head(dense_r, candidate_mask)

        return P_r, logits_r

    # ── Multi-round forward (training) ────────────────────────────────────────

    def forward_recurrent(
        self,
        x_base: torch.Tensor,
        slots: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        ds_edge_index: torch.Tensor,
        ds_edge_attr: torch.Tensor,
        K: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Unroll K rounds. Returns all intermediate P and logits for per-round loss.

        Returns
        -------
        Ps      : list of K tensors, each (N, N) doubly-stochastic
        logits  : list of K tensors, each (E,) sparse edge logits
        """
        if K is None:
            K = self.num_rounds
        device = x_base.device

        # Prior: uniform over visible slots (uninformed)
        P_prev = _uniform_belief(candidate_mask, device)

        Ps, logits_list = [], []
        for r in range(K):
            # During training, randomly reset belief to uniform with p=0.2
            # Forces each round to be independently capable, prevents
            # memorising specific belief trajectories
            if self.training and r > 0 and torch.rand(1).item() < 0.2:
                P_input = _uniform_belief(candidate_mask, device)
            else:
                P_input = P_prev

            P_r, logits_r = self._one_round(
                x_base, P_input, slots, candidate_mask,
                edge_index, edge_attr, ds_edge_index, ds_edge_attr,
                round_idx=r, device=device,
            )
            Ps.append(P_r)
            logits_list.append(logits_r)
            P_prev = P_r.detach()  # stop gradient between rounds (memory efficient)

        return Ps, logits_list

    # ── Single-round forward (inference, early-stop) ──────────────────────────

    @torch.no_grad()
    def forward_inference(
        self,
        x_base: torch.Tensor,
        slots: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        ds_edge_index: torch.Tensor,
        ds_edge_attr: torch.Tensor,
        max_rounds: int = NUM_ROUNDS,
        early_stop: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        Inference with optional early stopping.

        Stops when the hard assignment (argmax of P) doesn't change between
        consecutive rounds. Returns final P and number of rounds used.

        Returns
        -------
        P_final : (N, N) doubly-stochastic
        rounds_used : int
        """
        device  = x_base.device
        P_prev  = _uniform_belief(candidate_mask, device)
        prev_assignment = None

        for r in range(max_rounds):
            P_r, _ = self._one_round(
                x_base, P_prev, slots, candidate_mask,
                edge_index, edge_attr, ds_edge_index, ds_edge_attr,
                round_idx=r, device=device,
            )
            curr_assignment = P_r.argmax(dim=1)

            if early_stop and prev_assignment is not None:
                if (curr_assignment == prev_assignment).all():
                    return P_r, r + 1   # converged

            P_prev          = P_r
            prev_assignment = curr_assignment

        return P_r, max_rounds


# ─────────────────────────────────────────────────────────────────────────────
# 4.  recurrent_consensus_loss
# ─────────────────────────────────────────────────────────────────────────────

def recurrent_consensus_loss(
    Ps: List[torch.Tensor],
    logits_list: List[torch.Tensor],
    ds_edge_index: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    ce_weight: float     = SINKHORN_CE_W,
    regret_weight: float = SINKHORN_REGRET_W,
) -> Tuple[torch.Tensor, List[float], List[float]]:

    K       = len(Ps)
    weights = _round_weights(K)
    total   = torch.tensor(0.0, device=Ps[0].device)  # removed requires_grad=False
    ce_log, regret_log = [], []

    for r, (P_r, logits_r, w) in enumerate(zip(Ps, logits_list, weights)):
        loss_r, ce_r, regret_r = sinkhorn_assignment_loss(
            P_r, logits_r, ds_edge_index,
            targets, candidate_mask, drone_pos, slots,
            ce_weight=ce_weight, regret_weight=regret_weight,
        )
        total = total + w * loss_r
        ce_log.append(ce_r.item())
        regret_log.append(regret_r.item())

    # KL consistency — penalise large belief jumps between rounds
    consist = torch.tensor(0.0, device=Ps[0].device)
    for r in range(1, len(Ps)):
        p_prev = Ps[r-1].clamp(min=1e-9)
        p_curr = Ps[r].clamp(min=1e-9)
        kl     = (p_prev * (p_prev.log() - p_curr.log())).sum(dim=1)
        visible = candidate_mask.to(Ps[0].device).any(dim=1)
        if visible.any():
            consist = consist + kl[visible].mean()
    consist = consist / max(len(Ps) - 1, 1)
    total   = total + 0.05 * consist

    return total, ce_log, regret_log


# ─────────────────────────────────────────────────────────────────────────────
# 5.  evaluate_recurrent
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_recurrent(
    model: RecurrentConsensusGNN,
    dataset: List[Data],
    device: torch.device,
    slot_radius: float = SLOT_VISIBILITY_RADIUS,
    max_rounds: int    = NUM_ROUNDS,
    early_stop: bool   = True,
) -> Dict[str, float]:
    """
    Evaluate using recurrent inference with early stopping.

    Metrics returned match evaluate_strict_decentralized and evaluate_sinkhorn
    so all three can be compared in the same table.

    Extra metric: avg_rounds_used — average number of rounds before convergence.
    Ideally this is well below max_rounds for easy configurations and close to
    max_rounds only for hard symmetric cases.
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
        "avg_rounds_used":         [],
    }

    for data in dataset:
        fid   = data.formation_id.item()
        f_emb = model.formation_embedding(torch.tensor(fid, device=device))
        dp, sl, y = data.drone_pos, data.slots, data.y
        N     = dp.size(0)

        # Strict visibility — no force_gt at inference
        mask  = build_candidate_mask(dp, sl, slot_radius, y=y, force_gt=False)
        x     = build_node_features(dp, sl, f_emb.detach().cpu(),
                                     candidate_mask=mask).to(device)
        ei, ea = build_comm_graph(dp)
        ds_ei  = build_candidate_edges(mask)[0]
        ds_ea  = build_candidate_edge_attr(dp, sl, ds_ei)

        P_final, rounds_used = model.forward_inference(
            x,
            sl.to(device),
            mask.to(device),
            ei.to(device),
            ea.to(device),
            ds_ei.to(device),
            ds_ea.to(device),
            max_rounds=max_rounds,
            early_stop=early_stop,
        )

        # Hungarian rounding — guaranteed bijection
        P_cpu   = P_final.cpu()
        row_ind, col_ind = linear_sum_assignment(-P_cpu.numpy())
        assignment = torch.full((N,), -1, dtype=torch.long)
        assignment[row_ind] = torch.tensor(col_ind, dtype=torch.long)

        sample_m = assignment_quality_metrics(dp, sl, assignment, y,
                                               rounds=float(rounds_used))
        for key, val in sample_m.items():
            metrics[key].append(val)
        metrics["converged_rate"].append(
            1.0 if rounds_used < max_rounds else 0.0
        )
        metrics["avg_rounds_used"].append(float(rounds_used))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  train_recurrent_model
# ─────────────────────────────────────────────────────────────────────────────

def _new_params(model: RecurrentConsensusGNN) -> List[nn.Parameter]:
    """Parameters that are new (not from base checkpoint) — always trained."""
    return (
        list(model.round_embedding.parameters()) +
        list(model.recurrent_proj.parameters())  +
        list(model.sinkhorn_head.parameters())
    )


def _base_encoder_params(model: RecurrentConsensusGNN) -> List[nn.Parameter]:
    """Pre-trained encoder parameters — frozen initially."""
    return (
        list(model.formation_embedding.parameters()) +
        list(model.gat_layers.parameters())          +
        list(model.layer_norms.parameters())         +
        list(model.edge_scorer.parameters())         +
        [model.log_temp]
    )


def _current_K(epoch: int,
               start: int = MAX_ROUNDS_ANNEAL_START,
               end: int   = MAX_ROUNDS_ANNEAL_END,
               anneal_end_epoch: int = 20) -> int:
    """
    Linearly anneal number of training rounds from start → end over epochs.
    Training with K=2 first is faster and lets the model learn the feedback
    loop structure before being asked to unroll 4 rounds.
    """
    if epoch >= anneal_end_epoch:
        return end
    t = (epoch - 1) / max(anneal_end_epoch - 1, 1)
    return min(end, max(start, round(start + t * (end - start))))


def train_recurrent_model(
    model: RecurrentConsensusGNN,
    train_data: List[Data],
    val_data: List[Data],
    device: torch.device,
    epochs: int             = 80,
    lr: float               = 3e-4,
    freeze_epochs: int      = RECURRENT_FREEZE_EPOCHS,
    min_bijection_for_best: float = 0.90,
    force_gt_train: bool    = True,
    ce_weight: float        = SINKHORN_CE_W,
    regret_weight: float    = SINKHORN_REGRET_W,
) -> Dict[str, List]:
    """
    Train RecurrentConsensusGNN with:
      - K annealed from 2 → NUM_ROUNDS over first 20 epochs
      - Encoder frozen for freeze_epochs, then joint fine-tuning
      - Per-round weighted loss (later rounds weighted higher)
      - Early-stop evaluation to measure avg rounds used

    Checkpoint saved when cost_ratio is best AND bijection ≥ min_bijection_for_best.
    """
    model.to(device)

    # Phase 1: only new components
    new_ps  = _new_params(model)
    base_ps = _base_encoder_params(model)
    for p in base_ps:
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(new_ps, lr=lr)
    frozen    = True

    history: Dict[str, List] = {
        "train_loss":                  [],
        "val_loss":                    [],
        "val_bijection_rate":          [],
        "val_cost_ratio_vs_hungarian": [],
        "val_conflict_rate":           [],
        "val_unassigned_rate":         [],
        "val_slot_match_rate":         [],
        "val_avg_rounds_used":         [],
        "K_per_epoch":                 [],
    }
    best_score    = float("inf")
    best_model_st = None
    best_sink_st  = None

    for epoch in range(1, epochs + 1):

        # ── Round annealing ───────────────────────────────────────────────────
        K = _current_K(epoch)
        history["K_per_epoch"].append(K)

        # ── Unfreeze encoder ──────────────────────────────────────────────────
        if frozen and epoch > freeze_epochs:
            frozen = False
            for p in base_ps:
                p.requires_grad_(True)
            optimizer = torch.optim.Adam(
                list(model.parameters()), lr=lr * 0.3,
            )
            print(f"  [Epoch {epoch}] Encoder unfrozen — joint fine-tuning (K={K}).")

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0

        for data in train_data:
            fid   = data.formation_id.item()
            f_emb = model.formation_embedding(
                torch.tensor(fid, device=device)
            )
            mask  = build_candidate_mask(
                data.drone_pos, data.slots,
                SLOT_VISIBILITY_RADIUS, y=data.y, force_gt=force_gt_train,
            )
            x     = build_node_features(
                data.drone_pos, data.slots,
                f_emb.detach().cpu(), candidate_mask=mask,
            ).to(device)
            ei, ea = data.edge_index.to(device), data.edge_attr.to(device)
            ds_ei  = build_candidate_edges(mask)[0].to(device)
            ds_ea  = build_candidate_edge_attr(
                data.drone_pos, data.slots, ds_ei.cpu()
            ).to(device)
            mask_d = mask.to(device)

            Ps, logits_list = model.forward_recurrent(
                x, data.slots.to(device), mask_d,
                ei, ea, ds_ei, ds_ea, K=K,
            )
            loss, _, _ = recurrent_consensus_loss(
                Ps, logits_list, ds_ei,
                data.y.to(device), mask_d,
                data.drone_pos.to(device), data.slots.to(device),
                ce_weight=ce_weight, regret_weight=regret_weight,
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
                f_emb = model.formation_embedding(
                    torch.tensor(fid, device=device)
                )
                mask  = build_candidate_mask(
                    data.drone_pos, data.slots,
                    SLOT_VISIBILITY_RADIUS, y=data.y, force_gt=False,
                )
                x     = build_node_features(
                    data.drone_pos, data.slots,
                    f_emb.detach().cpu(), candidate_mask=mask,
                ).to(device)
                ei, ea = data.edge_index.to(device), data.edge_attr.to(device)
                ds_ei  = build_candidate_edges(mask)[0].to(device)
                ds_ea  = build_candidate_edge_attr(
                    data.drone_pos, data.slots, ds_ei.cpu()
                ).to(device)
                mask_d = mask.to(device)

                Ps, logits_list = model.forward_recurrent(
                    x, data.slots.to(device), mask_d,
                    ei, ea, ds_ei, ds_ea, K=K,
                )
                loss, _, _ = recurrent_consensus_loss(
                    Ps, logits_list, ds_ei,
                    data.y.to(device), mask_d,
                    data.drone_pos.to(device), data.slots.to(device),
                    ce_weight=ce_weight, regret_weight=regret_weight,
                )
                v_loss += loss.item()

        t_loss /= max(len(train_data), 1)
        v_loss /= max(len(val_data),   1)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        # ── Evaluation metrics (early-stop recurrent inference) ───────────────
        metrics = evaluate_recurrent(
            model, val_data[:100], device, max_rounds=K, early_stop=True
        )
        for key in ("bijection_rate", "cost_ratio_vs_hungarian",
                    "conflict_rate", "unassigned_rate", "slot_match_rate"):
            history[f"val_{key}"].append(metrics[key])
        history["val_avg_rounds_used"].append(metrics["avg_rounds_used"])

        # ── Checkpoint ────────────────────────────────────────────────────────
        score = metrics["cost_ratio_vs_hungarian"]
        if metrics["bijection_rate"] < min_bijection_for_best:
            score += 10.0 * (min_bijection_for_best - metrics["bijection_rate"])
        if score < best_score:
            best_score    = score
            best_model_st = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}

        # ── Logging ───────────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            phase = "frozen" if frozen else "joint"
            print(
                f"Epoch {epoch:3d} [{phase}|K={K}] | "
                f"train={t_loss:.4f} | val={v_loss:.4f} | "
                f"bij={metrics['bijection_rate']:.3f} | "
                f"cost={metrics['cost_ratio_vs_hungarian']:.4f} | "
                f"match={metrics['slot_match_rate']:.3f} | "
                f"rounds={metrics['avg_rounds_used']:.1f}"
            )

    if best_model_st is not None:
        model.load_state_dict(best_model_st)
    return history


