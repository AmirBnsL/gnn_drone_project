"""
hungarian_imitator.py
=====================
Centralised O(N²) Hungarian imitator for drone-to-slot assignment.

Design goals
------------
  * Match Hungarian assignment quality through imitation learning.
  * O(N²) forward pass — one attention layer over (drone, slot) pairs,
    no O(N³) sorting or Hungarian call at inference time.
  * Sinkhorn differentiable assignment during training for soft supervision;
    greedy hardening at inference (same as SuperGlue path but bijection-aware).
  * Shares the SuperGlue backbone so the pre-trained checkpoint can warm-start
    this model with zero extra code.

Architecture summary
--------------------
  1. Encode drones and slots with the same MLP projectors as SuperGlue.
  2. Run NUM_LAYERS of full (unmasked) cross-attention between ALL drones and
     ALL slots — no communication-radius restriction, which is the key
     difference from the decentralised models.
  3. Build the N×S value matrix from pair features (same head as SuperGlue).
  4. At training time, apply Sinkhorn normalisation to get a soft doubly-
     stochastic assignment P, then supervise with Hungarian labels via:
       - Cross-entropy on rows of P vs. ground-truth column indices.
       - Sinkhorn assignment cost vs. Hungarian cost (regret).
       - Margin loss to sharpen the winning slot's value.
  5. At inference, run bijection-safe greedy decoding on the raw value matrix
     (no Sinkhorn needed — values are already well-separated after training).

Complexity
----------
  Transformer: O(N·S·D)  — N drones, S slots, D hidden dim.
  Sinkhorn (train only): O(N·S·T)  — T Sinkhorn iterations (~20).
  Greedy decode (inference): O(N²) worst case, O(N log N) typical.
  Compare: Hungarian O(N³), decent_auction O(N²·R·E) with R rounds and E
  message passes per round.

No decentralised state, no auction rounds, no communication graph needed.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from local_negotiator_v2 import (
    NUM_FORMATIONS,
    SLOT_VISIBILITY_RADIUS,
    assignment_quality_metrics,
    build_candidate_mask,
    load_negotiator_dataset,
    prepare_dataset,
)
from superglue_negotiator import (
    FeedForwardBlock,
    MLP,
    SuperGlueSwarmMatcher,
    VALUE_SCALE,
    FORMATION_EMB_DIM,
    HIDDEN_DIM,
    NUM_HEADS,
    NUM_LAYERS,
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

SINKHORN_ITERS: int = 20          # Sinkhorn normalisation iterations (train)
SINKHORN_TEMP: float = 0.50        # Softmax temperature before Sinkhorn
# FIX: raised from 0.05 to 0.50. With VALUE_SCALE=50, temp=0.05 produced
# log_alpha in [-1000,+1000] causing Sinkhorn log_r/log_c to overflow,
# making soft_P rows sum to 0.74–1.52 instead of 1.0. At temp=0.5 the
# range is [-100,+100] — still sharp but numerically stable.
CE_WEIGHT: float = 1.0             # Cross-entropy on soft assignment rows
REGRET_WEIGHT: float = 0.30        # Expected-cost vs optimal-cost penalty
MARGIN_WEIGHT: float = 0.10        # Hard-margin on value matrix
COVERAGE_WEIGHT: float = 0.05      # Penalty for negative peak values
MARGIN: float = 0.20               # Minimum gap between best and 2nd-best value
MIN_BIJ_FOR_BEST: float = 0.90     # Bijection threshold for checkpoint saving
EPS_CLIP: float = 1e-6             # Numerical floor inside Sinkhorn log


# ---------------------------------------------------------------------------
# Full cross-attention layer (no mask, no comm-graph)
# ---------------------------------------------------------------------------

class FullCrossAttention(nn.Module):
    """
    Standard multi-head cross-attention without any masking.

    Unlike the decentralised models, every drone attends to every slot and
    vice-versa.  This gives O(N·S) complexity per layer — cheaper than the
    O(N²·S) of the SuperGlue masked variant on sparse graphs, and it lets the
    model see the global assignment context.

    q: (Nq, D), k/v: (Nk, D) → output: (Nq, D)
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        nq, nk = q.size(0), k.size(0)
        if nq == 0 or nk == 0:
            return q

        # (heads, Nq, head_dim)
        qh = self.q_proj(q).view(nq, self.heads, self.head_dim).transpose(0, 1)
        kh = self.k_proj(k).view(nk, self.heads, self.head_dim).transpose(0, 1)
        vh = self.v_proj(v).view(nk, self.heads, self.head_dim).transpose(0, 1)

        # (heads, Nq, Nk)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)

        # (Nq, dim)
        msg = torch.matmul(attn, vh).transpose(0, 1).reshape(nq, self.dim)
        return self.norm(q + self.out_proj(msg))


class HungarianImitatorLayer(nn.Module):
    """
    One encoder layer: drone self-attention + slot self-attention +
    full bidirectional cross-attention + feed-forward blocks.

    Drone self-attention: every drone sees every other drone — full global
    context (no comm-radius restriction).
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.drone_self = FullCrossAttention(dim, heads)
        self.slot_self = FullCrossAttention(dim, heads)
        self.drone_cross = FullCrossAttention(dim, heads)
        self.slot_cross = FullCrossAttention(dim, heads)
        self.drone_ff = FeedForwardBlock(dim)
        self.slot_ff = FeedForwardBlock(dim)

    def forward(
        self,
        drone_h: torch.Tensor,
        slot_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Full self-attention (no mask)
        drone_h = self.drone_self(drone_h, drone_h, drone_h)
        slot_h = self.slot_self(slot_h, slot_h, slot_h)
        # Full bidirectional cross-attention
        drone_h = self.drone_cross(drone_h, slot_h, slot_h)
        slot_h = self.slot_cross(slot_h, drone_h, drone_h)
        return self.drone_ff(drone_h), self.slot_ff(slot_h)


# ---------------------------------------------------------------------------
# Sinkhorn normalisation
# ---------------------------------------------------------------------------

def sinkhorn(
    log_alpha: torch.Tensor,
    n_iters: int = SINKHORN_ITERS,
    eps: float = EPS_CLIP,
) -> torch.Tensor:
    """
    Differentiable Sinkhorn normalisation in log-space.

    Input:  log_alpha  (N, S) — log of un-normalised assignment scores.
    Output: P          (N, S) — doubly stochastic soft assignment matrix.

    Log-space arithmetic avoids numerical underflow for large N.
    """
    log_r = torch.zeros(log_alpha.size(0), 1, device=log_alpha.device)
    log_c = torch.zeros(1, log_alpha.size(1), device=log_alpha.device)

    # Log-space Sinkhorn: alternating row/column normalisation
    for _ in range(n_iters):
        log_r = -torch.logsumexp(log_alpha + log_c, dim=1, keepdim=True)
        log_c = -torch.logsumexp(log_alpha + log_r, dim=0, keepdim=True)

    return torch.exp(log_alpha + log_r + log_c).clamp(min=0.0, max=1.0)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HungarianImitator(nn.Module):
    """
    Centralised O(N²) Hungarian imitator.

    Can be initialised cold or warm-started from a SuperGlueSwarmMatcher
    checkpoint.  When warm-started, the MLP projectors and value_head weights
    are copied directly; only the attention layers differ (full vs masked) so
    those are re-initialised.

    Parameters
    ----------
    hidden_dim : int
        Width of all internal representations.
    num_heads : int
        Number of attention heads (must divide hidden_dim).
    num_layers : int
        Number of encoder layers.
    num_formations : int
        Number of formation types for the embedding table.
    formation_emb_dim : int
        Dimension of the formation embedding.
    sinkhorn_iters : int
        Sinkhorn iterations during training forward pass.
    sinkhorn_temp : float
        Softmax temperature applied to values before Sinkhorn.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        num_formations: int = NUM_FORMATIONS,
        formation_emb_dim: int = FORMATION_EMB_DIM,
        sinkhorn_iters: int = SINKHORN_ITERS,
        sinkhorn_temp: float = SINKHORN_TEMP,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_temp = sinkhorn_temp

        self.formation_embedding = nn.Embedding(num_formations, formation_emb_dim)
        self.drone_mlp = MLP(2 + formation_emb_dim, hidden_dim, hidden_dim)
        self.slot_mlp = MLP(2 + formation_emb_dim, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [HungarianImitatorLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        # Same value head as SuperGlue: pair features → scalar value
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Weight transfer from SuperGlueSwarmMatcher
    # ------------------------------------------------------------------

    @classmethod
    def from_superglue(
        cls,
        matcher: SuperGlueSwarmMatcher,
        sinkhorn_iters: int = SINKHORN_ITERS,
        sinkhorn_temp: float = SINKHORN_TEMP,
    ) -> "HungarianImitator":
        """
        Build a HungarianImitator and copy compatible weights from a
        SuperGlueSwarmMatcher checkpoint.

        Copied layers (exact weight transfer):
          - formation_embedding
          - drone_mlp, slot_mlp
          - value_head

        NOT copied (incompatible — full vs masked attention):
          - layers (re-initialised randomly)
          - eps_head (not used in this model)
        """
        model = cls(
            hidden_dim=matcher.hidden_dim,
            num_heads=matcher.layers[0].drone_self.heads,
            num_layers=len(matcher.layers),
            num_formations=matcher.formation_embedding.num_embeddings,
            formation_emb_dim=matcher.formation_embedding.embedding_dim,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_temp=sinkhorn_temp,
        )
        model.formation_embedding.load_state_dict(
            matcher.formation_embedding.state_dict()
        )
        model.drone_mlp.load_state_dict(matcher.drone_mlp.state_dict())
        model.slot_mlp.load_state_dict(matcher.slot_mlp.state_dict())
        model.value_head.load_state_dict(matcher.value_head.state_dict())
        return model

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        drone_pos: torch.Tensor,
        slots: torch.Tensor,
        formation_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        drone_pos    : (N, 2)  — drone 2-D positions.
        slots        : (S, 2)  — slot 2-D positions.
        formation_id : (,) or (N,) — integer formation index.

        Returns
        -------
        values  : (N, S)  — raw value matrix (used for greedy decode).
        soft_P  : (N, S)  — Sinkhorn soft assignment (used for training loss).
        """
        device = drone_pos.device
        n_drones = drone_pos.size(0)
        n_slots = slots.size(0)

        fid = formation_id.to(device).long().view(-1)[0]
        f_emb = self.formation_embedding(fid).unsqueeze(0)  # (1, formation_emb_dim)

        # Input projections — identical to SuperGlue
        drone_in = torch.cat([drone_pos, f_emb.expand(n_drones, -1)], dim=1)
        slot_in = torch.cat([slots, f_emb.expand(n_slots, -1)], dim=1)
        drone_h = self.drone_mlp(drone_in)   # (N, D)
        slot_h = self.slot_mlp(slot_in)       # (S, D)

        # Full cross-attention encoder
        for layer in self.layers:
            drone_h, slot_h = layer(drone_h, slot_h)

        # Pair feature matrix — identical to SuperGlue value head input
        d = drone_h.unsqueeze(1).expand(n_drones, n_slots, -1)   # (N, S, D)
        s = slot_h.unsqueeze(0).expand(n_drones, n_slots, -1)    # (N, S, D)
        rel = slots.unsqueeze(0) - drone_pos.unsqueeze(1)         # (N, S, 2)
        dist = rel.norm(dim=-1, keepdim=True)                     # (N, S, 1)
        pair = torch.cat([d, s, rel, dist], dim=-1)               # (N, S, 2D+3)

        values = VALUE_SCALE * torch.tanh(self.value_head(pair).squeeze(-1))  # (N, S)

        log_alpha = values / self.sinkhorn_temp
        soft_P = sinkhorn(log_alpha, n_iters=self.sinkhorn_iters)

        return values, soft_P


# ---------------------------------------------------------------------------
# Loss functions  (all vectorised — no per-drone Python loops)
# ---------------------------------------------------------------------------

def _valid_drone_mask(targets: torch.Tensor, n_slots: int, device: torch.device) -> torch.Tensor:
    """Boolean mask of drones whose GT slot index is within range."""
    tgt = targets.to(device).long()
    return (tgt >= 0) & (tgt < n_slots)


def hungarian_ce_loss(
    soft_P: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy on rows of the Sinkhorn soft assignment matrix.

    Treats each row of soft_P as a probability distribution over slots and
    applies NLL against the ground-truth column index.  This is the primary
    imitation signal — it directly teaches the model to put probability mass
    on the Hungarian-optimal slot.

    soft_P  : (N, S) — doubly stochastic matrix from sinkhorn().
    targets : (N,)   — ground-truth slot index per drone (from data.y).
    """
    device = soft_P.device
    tgt = targets.to(device).long()
    valid = _valid_drone_mask(tgt, soft_P.size(1), device)
    if not bool(valid.any()):
        return soft_P.sum() * 0.0
    # NLL on soft probabilities — clamp to avoid log(0)
    log_p = torch.log(soft_P[valid].clamp(min=EPS_CLIP))
    return F.nll_loss(log_p, tgt[valid])


def hungarian_regret_loss(
    values: torch.Tensor,
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Expected assignment cost under a softmax policy minus the optimal cost.

    Encourages the model to not just predict the right slot, but to predict
    values that reflect actual spatial distances — preventing the model from
    learning degenerate high-confidence predictions on wrong slots.
    """
    device = values.device
    tgt = targets.to(device).long()
    valid = _valid_drone_mask(tgt, values.size(1), device)
    if not bool(valid.any()):
        return values.sum() * 0.0

    drone_pos = drone_pos.to(device).float()
    slots = slots.to(device).float()
    probs = torch.softmax(values[valid], dim=1)
    dist = torch.cdist(drone_pos[valid], slots)
    expected_cost = (probs * dist).sum(dim=1)
    optimal_cost = dist[torch.arange(valid.sum(), device=device), tgt[valid]]
    return F.relu(expected_cost - optimal_cost).mean()


def hungarian_margin_loss(
    values: torch.Tensor,
    targets: torch.Tensor,
    margin: float = MARGIN,
) -> torch.Tensor:
    """
    Vectorised margin loss: the GT slot's value must exceed the best
    competitor by at least `margin`.

    No per-drone Python loop — operates in a single masked matmul pass.
    """
    device = values.device
    tgt = targets.to(device).long()
    n = tgt.numel()
    valid = _valid_drone_mask(tgt, values.size(1), device)
    if not bool(valid.any()):
        return values.sum() * 0.0

    row_idx = torch.arange(n, device=device)[valid]
    target_vals = values[row_idx, tgt[valid]]

    # Best competitor: mask out the target slot, then take max
    others = values[valid].clone()
    others[torch.arange(valid.sum(), device=device), tgt[valid]] = -1e9
    best_other = others.max(dim=1).values

    # Exclude rows where there is no competitor (single-slot problem)
    has_competitor = best_other > -1e8
    if not bool(has_competitor.any()):
        return values.sum() * 0.0

    loss = F.relu(margin - (target_vals[has_competitor] - best_other[has_competitor]))
    return loss.mean()


def hungarian_coverage_loss(values: torch.Tensor) -> torch.Tensor:
    """
    Penalise negative peak values — every drone should have at least one
    slot with positive value so greedy decoding does not leave drones unassigned.
    """
    peak = values.max(dim=1).values
    return F.relu(-peak).mean()


def hungarian_imitator_loss(
    values: torch.Tensor,
    soft_P: torch.Tensor,
    data: Data,
    ce_weight: float = CE_WEIGHT,
    regret_weight: float = REGRET_WEIGHT,
    margin_weight: float = MARGIN_WEIGHT,
    coverage_weight: float = COVERAGE_WEIGHT,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Combined loss for Hungarian imitation.

    Returns
    -------
    total : scalar loss tensor (differentiable).
    parts : dict of named sub-losses (for logging).
    """
    targets = data.y.to(values.device).long()

    ce = hungarian_ce_loss(soft_P, targets)
    regret = hungarian_regret_loss(values, data.drone_pos, data.slots, targets)
    margin = hungarian_margin_loss(values, targets)
    coverage = hungarian_coverage_loss(values)

    total = (
        ce_weight * ce
        + regret_weight * regret
        + margin_weight * margin
        + coverage_weight * coverage
    )
    return total, {"ce": ce, "regret": regret, "margin": margin, "coverage": coverage}


# ---------------------------------------------------------------------------
# Greedy bijection-safe assignment (inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_bijection_assignment(values: torch.Tensor) -> torch.Tensor:
    """
    O(N²) worst-case greedy assignment that guarantees bijection.

    Drones are processed in descending order of their peak value (most
    confident first).  Each drone takes its highest-valued slot that has
    not yet been taken.

    This is equivalent to the SuperGlue greedy decode but without a mask
    parameter since the centralised model sees all slots.

    Returns assignment : (N,) long tensor, value ≥ 0 for assigned drones.
    """
    n, s = values.size()
    score = values.cpu().float().clone()
    assignment = torch.full((n,), -1, dtype=torch.long)
    taken_mask = torch.zeros(s, dtype=torch.bool)
    # Sort drones by descending confidence
    order = score.max(dim=1).values.argsort(descending=True).tolist()
    for drone in order:
        if taken_mask.all():
            break
        row = score[drone].masked_fill(taken_mask, -1e9)
        if row.max().item() <= -1e8:
            continue
        slot = int(row.argmax().item())
        assignment[drone] = slot
        taken_mask[slot] = True
    return assignment


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_hungarian_imitator(
    model: HungarianImitator,
    dataset: List[Data],
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate assignment quality metrics over a dataset.

    Returns a dict with the same keys as the other models in this codebase
    so results are directly comparable in the notebook comparison tables.
    """
    model.eval()
    metrics: Dict[str, List[float]] = {
        "bijection_rate": [],
        "conflict_rate": [],
        "unassigned_rate": [],
        "cost_ratio_vs_hungarian": [],
        "slot_match_rate": [],
        "consensus_rounds": [],   # always 0 — centralised, no rounds
    }
    for data in dataset:
        values, _ = model(
            data.drone_pos.to(device).float(),
            data.slots.to(device).float(),
            data.formation_id.to(device),
        )
        assignment = greedy_bijection_assignment(values)
        sample = assignment_quality_metrics(
            data.drone_pos, data.slots, assignment, data.y, rounds=0.0
        )
        for key, value in sample.items():
            metrics[key].append(value)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_hungarian_imitator(
    model: HungarianImitator,
    train_data: List[Data],
    val_data: List[Data],
    device: torch.device,
    epochs: int = 80,
    lr: float = 3e-4,
    min_bijection_for_best: float = MIN_BIJ_FOR_BEST,
    eval_subset: int = 200,
    ckpt_path: Optional[str] = "hungarian_imitator_best.pt",
    # Freeze embedding + MLP projectors for the first N epochs when
    # warm-starting from SuperGlue to let the new attention layers stabilise.
    freeze_projectors_epochs: int = 0,
) -> Dict[str, List[float]]:
    """
    Train the HungarianImitator with supervised imitation of Hungarian labels.

    Training protocol
    -----------------
    1. (Optional) Freeze formation_embedding, drone_mlp, slot_mlp, value_head
       for the first `freeze_projectors_epochs` epochs.  Useful when warm-
       starting from SuperGlue — lets the new full-attention layers adapt
       before the shared weights start moving.
    2. Unfreeze everything and fine-tune jointly with a lower LR on the
       projector group (0.1×) to prevent catastrophic forgetting.
    3. Save best checkpoint by composite score:
         score = cost_ratio_vs_hungarian
                 + 10 × max(0, min_bij_for_best − bijection_rate)
       so we only prefer a lower cost ratio if bijection is already high.

    Parameters
    ----------
    model                   : HungarianImitator to train.
    train_data / val_data   : lists of PyG Data objects.
    device                  : torch device.
    epochs                  : total training epochs.
    lr                      : peak learning rate.
    min_bijection_for_best  : bijection threshold for best-model selection.
    eval_subset             : number of val samples used per epoch.
    ckpt_path               : where to save the best checkpoint (or None).
    freeze_projectors_epochs: how many epochs to freeze projectors (0 = never).

    Returns
    -------
    history : dict of per-epoch metric lists for plotting.
    """
    model.to(device)

    # Separate projector params (potentially warm-started) from fresh
    # attention-layer params so we can apply different LRs.
    projector_params: List[nn.Parameter] = (
        list(model.formation_embedding.parameters())
        + list(model.drone_mlp.parameters())
        + list(model.slot_mlp.parameters())
        + list(model.value_head.parameters())
    )
    attention_params: List[nn.Parameter] = list(model.layers.parameters())

    frozen = freeze_projectors_epochs > 0
    if frozen:
        for p in projector_params:
            p.requires_grad_(False)

    # Frozen phase: only train attention layers
    optimizer = torch.optim.AdamW(attention_params, lr=lr, weight_decay=1e-4)
    # Cosine schedule spans the whole training run regardless of freeze phase
    frozen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-5
    )
    joint_scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_ce": [],
        "train_regret": [],
        "train_margin": [],
        "val_loss": [],
        "val_bijection_rate": [],
        "val_cost_ratio_vs_hungarian": [],
        "val_slot_match_rate": [],
        "val_conflict_rate": [],
        "val_unassigned_rate": [],
        "lr": [],
        "phase": [],
    }
    best_score = float("inf")
    best_state: Optional[Dict] = None

    for epoch in range(1, epochs + 1):

        # ---- Training ----
        model.train()
        random.shuffle(train_data)
        ep_loss, ep_ce, ep_regret, ep_margin = [], [], [], []

        for data in train_data:
            values, soft_P = model(
                data.drone_pos.to(device).float(),
                data.slots.to(device).float(),
                data.formation_id.to(device),
            )
            loss, parts = hungarian_imitator_loss(values, soft_P, data)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            ep_loss.append(float(loss.item()))
            ep_ce.append(float(parts["ce"].item()))
            ep_regret.append(float(parts["regret"].item()))
            ep_margin.append(float(parts["margin"].item()))

        # ---- Scheduler step ----
        if frozen:
            frozen_scheduler.step()
        elif joint_scheduler is not None:
            joint_scheduler.step()

        history["train_loss"].append(float(np.mean(ep_loss)) if ep_loss else 0.0)
        history["train_ce"].append(float(np.mean(ep_ce)) if ep_ce else 0.0)
        history["train_regret"].append(float(np.mean(ep_regret)) if ep_regret else 0.0)
        history["train_margin"].append(float(np.mean(ep_margin)) if ep_margin else 0.0)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["phase"].append(0.0 if frozen else 1.0)

        # ---- Validation loss ----
        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for data in val_data[:eval_subset]:
                values, soft_P = model(
                    data.drone_pos.to(device).float(),
                    data.slots.to(device).float(),
                    data.formation_id.to(device),
                )
                val_loss, _ = hungarian_imitator_loss(values, soft_P, data)
                val_losses.append(float(val_loss.item()))

        # ---- Validation assignment metrics ----
        val_metrics = evaluate_hungarian_imitator(model, val_data[:eval_subset], device)
        history["val_loss"].append(float(np.mean(val_losses)) if val_losses else 0.0)
        for key in (
            "bijection_rate",
            "cost_ratio_vs_hungarian",
            "slot_match_rate",
            "conflict_rate",
            "unassigned_rate",
        ):
            history[f"val_{key}"].append(val_metrics[key])

        # ---- Best model selection ----
        score = val_metrics["cost_ratio_vs_hungarian"]
        if val_metrics["bijection_rate"] < min_bijection_for_best:
            score += 10.0 * (min_bijection_for_best - val_metrics["bijection_rate"])
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "metrics": val_metrics,
                        "epoch": epoch,
                    },
                    ckpt_path,
                )

        # ---- Unfreeze projectors after freeze_projectors_epochs ----
        if frozen and epoch > freeze_projectors_epochs:
            for p in projector_params:
                p.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": attention_params, "lr": lr},
                    # Projectors (possibly warm-started) get a lower LR
                    # to avoid undoing the SuperGlue pre-training.
                    {"params": projector_params, "lr": lr * 0.1},
                ],
                weight_decay=1e-4,
            )
            joint_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(epochs - epoch, 1),
                eta_min=1e-5,
            )
            frozen = False
            print(
                f"  [Epoch {epoch}] Projectors unfrozen — joint fine-tuning starts."
            )

        phase = "frozen" if frozen else "joint"
        print(
            f"epoch {epoch:03d} [{phase}] "
            f"train={history['train_loss'][-1]:.4f} "
            f"(ce={history['train_ce'][-1]:.3f} "
            f"reg={history['train_regret'][-1]:.3f} "
            f"mrg={history['train_margin'][-1]:.3f}) "
            f"val={history['val_loss'][-1]:.4f} "
            f"bij={val_metrics['bijection_rate']:.3f} "
            f"cost={val_metrics['cost_ratio_vs_hungarian']:.4f} "
            f"match={val_metrics['slot_match_rate']:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    DATASET = "./dataset/negotiator_dataset_v1.pt"
    MATCHER_CKPT = "superglue_negotiator_best.pt"
    OUT = "hungarian_imitator_best.pt"

    raw = load_negotiator_dataset(DATASET)

    # Build model — warm-start from SuperGlue if checkpoint exists
    if os.path.isfile(MATCHER_CKPT):
        print(f"Warm-starting from {MATCHER_CKPT}")
        ckpt = torch.load(MATCHER_CKPT, map_location="cpu", weights_only=False)
        matcher = SuperGlueSwarmMatcher()
        matcher.load_state_dict(ckpt["model_state_dict"])
        model = HungarianImitator.from_superglue(matcher)
        freeze_epochs = 10   # let attention layers adapt first
    else:
        print("No SuperGlue checkpoint found — training from scratch.")
        model = HungarianImitator()
        freeze_epochs = 0

    train_data, val_data, test_data = prepare_dataset(
        raw,
        model.formation_embedding.weight.detach().cpu(),
        seed=seed,
        force_gt_visibility=False,
    )
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_hungarian_imitator(
        model,
        train_data[:7000],
        val_data[:500],
        device,
        epochs=80,
        lr=3e-4,
        freeze_projectors_epochs=freeze_epochs,
        eval_subset=200,
        ckpt_path=OUT,
    )

    test_metrics = evaluate_hungarian_imitator(model, test_data[:500], device)
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
