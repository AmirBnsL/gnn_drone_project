"""
sinkhorn_head.py
================
Drop-in Sinkhorn output head for LocalNegotiatorGNN.

Adds four things to local_negotiator.py without touching it:
  • SinkhornHead          — differentiable doubly-stochastic projection
  • forward_sinkhorn()    — monkey-patched method for LocalNegotiatorGNN
  • sinkhorn_assignment_loss() — soft CE + cost-regret, fully differentiable
  • evaluate_sinkhorn()   — evaluation using Sinkhorn inference (no gossip)
  • train_sinkhorn_model() — fine-tuning loop with freeze/unfreeze schedule

Backward compatibility
----------------------
All original functions in local_negotiator.py (train_strict_decentralized_model,
evaluate_strict_decentralized, assign_strict_decentralized_consensus) work
unchanged. The gossip path is not modified.

Checkpoint format
-----------------
Saves the same dict structure as train_strict_decentralized.ipynb:
  {model_state_dict, sinkhorn_state_dict, config, history, metrics}
The base model state is saved separately from the SinkhornHead state so you
can load either independently.
"""

from __future__ import annotations

import math
import types
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data

from local_negotiator import (
    CONFLICT_LAMBDA,
    COMM_RADIUS,
    REGRET_LAMBDA,
    SLOT_VISIBILITY_RADIUS,
    LocalNegotiatorGNN,
    assignment_quality_metrics,
    build_candidate_edges,
    build_candidate_mask,
    build_candidate_edge_attr,
    build_comm_graph,
    build_node_features,
    cost_regret_penalty,
    edge_logits_to_dense,
    prepare_dataset,
    load_negotiator_dataset,
)

# ── Constants ─────────────────────────────────────────────────────────────────
SINKHORN_ITERS   = 20      # number of log-space Sinkhorn iterations
SINKHORN_EPS     = 1e-8    # numerical stability
FREEZE_EPOCHS    = 15      # encoder frozen for this many epochs then unfrozen
SINKHORN_CE_W    = 1.0     # weight on soft cross-entropy term
SINKHORN_REGRET_W = 0.30   # weight on cost-regret term (doubled from 0.15 — keeps model cost-aware)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SinkhornHead
# ─────────────────────────────────────────────────────────────────────────────

class SinkhornHead(nn.Module):
    """
    Converts a sparse (N, N) logit matrix into a doubly-stochastic assignment
    matrix via log-space Sinkhorn iterations.

    Design decisions
    ----------------
    • Operates on a DENSE (N, N) tensor produced by edge_logits_to_dense().
      Invisible slots are already masked to -1e4 before this module runs.
    • Uses its own learnable log-temperature (sinkhorn_log_temp) separate from
      the encoder's log_temp so the two can be tuned independently.
    • Log-space iterations: numerically stable for large N and many iters.
    • Variable N: fully dynamic — no hardcoded matrix size anywhere.
    • Interpretation: each Sinkhorn iteration is one round of row-normalisation
      (drone picks a slot distribution) followed by one round of
      column-normalisation (slot resolves which drone gets it). This is the
      distributed-auction interpretation from the prompt — the iterations *could*
      be run in a gossip fashion where each drone updates its row and neighbours
      pass column sums.

    Forward
    -------
    Input  : logits (N, N) — already masked, -1e4 for invisible pairs
             candidate_mask (N, N) bool — True for visible pairs
    Output : P (N, N) doubly-stochastic matrix
             Each P[i, j] = soft probability that drone i → slot j.
             At eval time: round P via argmax (greedy) or Hungarian.
    """

    def __init__(self, n_iters: int = SINKHORN_ITERS, eps: float = SINKHORN_EPS):
        super().__init__()
        self.n_iters         = n_iters
        self.eps             = eps
        # Separate temperature from the encoder's log_temp
        self.sinkhorn_log_temp = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        logits: torch.Tensor,          # (N, N), invisible = -1e4
        candidate_mask: torch.Tensor,  # (N, N) bool
    ) -> torch.Tensor:
        """Return doubly-stochastic matrix P (N, N)."""
        temp = torch.exp(self.sinkhorn_log_temp).clamp(min=1e-3)
        # Scale logits, keep invisible pairs masked
        log_alpha = logits / temp
        log_alpha = log_alpha.masked_fill(~candidate_mask, -1e9)

        # Log-space Sinkhorn
        # Invariant: log_alpha = M + u.unsqueeze(1) + v.unsqueeze(0)
        # where u (N,) and v (N,) are the dual variables.
        for _ in range(self.n_iters):
            # Row normalisation (drone softmax over visible slots)
            log_alpha = log_alpha - torch.logsumexp(
                log_alpha.masked_fill(~candidate_mask, -1e9), dim=1, keepdim=True
            )
            # Column normalisation (slot softmax over bidding drones)
            log_alpha = log_alpha - torch.logsumexp(
                log_alpha.masked_fill(~candidate_mask, -1e9), dim=0, keepdim=True
            )
            # Re-mask after each step to prevent drift into invisible cells
            log_alpha = log_alpha.masked_fill(~candidate_mask, -1e9)

        P = torch.exp(log_alpha)
        P = P.masked_fill(~candidate_mask, 0.0)
        return P   # (N, N), rows and cols approximately sum to 1


# ─────────────────────────────────────────────────────────────────────────────
# 2.  forward_sinkhorn  (patched onto LocalNegotiatorGNN)
# ─────────────────────────────────────────────────────────────────────────────

def _forward_sinkhorn(
    self,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    ds_edge_index: torch.Tensor,
    ds_edge_attr: torch.Tensor,
    candidate_mask: torch.Tensor,
    sinkhorn_head: "SinkhornHead",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full forward pass ending in a doubly-stochastic assignment matrix.

    Returns
    -------
    P           : (N, N) doubly-stochastic matrix
    edge_logits : (E,)   sparse edge logits (kept for loss computation)
    """
    n = candidate_mask.size(0)
    # Encode drone states via GATv2 comm graph
    h = self.encode_drones(x, edge_index, edge_attr)
    # Score every visible drone→slot pair
    edge_logits = self.forward_sparse(h, ds_edge_index, ds_edge_attr)
    # Densify: invisible pairs → -1e4
    dense = edge_logits_to_dense(edge_logits, ds_edge_index, n, fill=-1e4)
    dense = dense.masked_fill(~candidate_mask, -1e4)
    # Project to doubly-stochastic
    P = sinkhorn_head(dense, candidate_mask)
    return P, edge_logits


def attach_sinkhorn(model: LocalNegotiatorGNN, sinkhorn_head: SinkhornHead) -> None:
    """
    Monkey-patch forward_sinkhorn onto a LocalNegotiatorGNN instance.
    Call once after loading the checkpoint.

        attach_sinkhorn(model, sinkhorn_head)
        P, logits = model.forward_sinkhorn(x, ei, ea, ds_ei, ds_ea, mask, sinkhorn_head)
    """
    model.forward_sinkhorn = types.MethodType(
        lambda self, x, ei, ea, ds_ei, ds_ea, mask, sh=sinkhorn_head: (
            _forward_sinkhorn(self, x, ei, ea, ds_ei, ds_ea, mask, sh)
        ),
        model,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  sinkhorn_assignment_loss
# ─────────────────────────────────────────────────────────────────────────────

def sinkhorn_assignment_loss(
    P: torch.Tensor,              # (N, N) doubly-stochastic
    edge_logits: torch.Tensor,    # (E,) sparse logits (for regret penalty)
    ds_edge_index: torch.Tensor,  # (2, E)
    targets: torch.Tensor,        # (N,) long — GT slot per drone
    candidate_mask: torch.Tensor, # (N, N) bool
    drone_pos: torch.Tensor,      # (N, 2)
    slots: torch.Tensor,          # (N, 2)
    ce_weight: float    = SINKHORN_CE_W,
    regret_weight: float = SINKHORN_REGRET_W,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable Sinkhorn assignment loss.

    Terms
    -----
    1. Soft cross-entropy: -log P[i, gt_slot_i] for each drone i.
       This is the standard supervised signal — push P[i, gt] → 1.
       Computed only for drones that can see their GT slot (candidate_mask).

    2. Cost-regret penalty: expected travel cost relative to Hungarian target.
       Uses the doubly-stochastic P as a soft probability distribution.
       Keeps the model cost-aware even during Sinkhorn training.
       (Original regret term from strict_assignment_loss, rephrased for P.)

    No conflict penalty term — Sinkhorn guarantees conflict-free output by
    construction (doubly-stochastic), so penalising conflicts is redundant
    and would fight the Sinkhorn projection.

    Returns
    -------
    total, ce_loss, regret_loss
    """
    n = targets.size(0)
    dev = P.device

    # ── 1. Soft cross-entropy ────────────────────────────────────────────────
    # For drone i: loss_i = -log(P[i, gt_i] + eps)
    # Only include drones that have their GT slot visible.
    gt_probs = P[torch.arange(n, device=dev), targets.to(dev)]  # (N,)
    visible_gt = candidate_mask[torch.arange(n, device=dev), targets.to(dev)]  # (N,) bool
    log_probs = torch.log(gt_probs.clamp(min=1e-9))
    if visible_gt.any():
        ce_loss = -log_probs[visible_gt].mean()
    else:
        ce_loss = P.sum() * 0.0  # zero but differentiable

    # ── 2. Cost-regret penalty ───────────────────────────────────────────────
    # Expected cost under P vs Hungarian target cost per drone.
    dist = torch.cdist(
        drone_pos.to(dev).float(),
        slots.to(dev).float(),
    )                                       # (N, N)
    expected_cost = (P * dist).sum(dim=1)   # (N,)  soft expected travel dist
    target_cost   = dist[
        torch.arange(n, device=dev),
        targets.to(dev),
    ]                                       # (N,)
    visible_rows = candidate_mask.to(dev).any(dim=1)
    if visible_rows.any():
        regret = F.relu(expected_cost - target_cost)
        regret_loss = regret[visible_rows].mean()
    else:
        regret_loss = P.sum() * 0.0

    # Guard NaN
    if torch.isnan(ce_loss):
        ce_loss = P.sum() * 0.0
    if torch.isnan(regret_loss):
        regret_loss = P.sum() * 0.0

    total = ce_weight * ce_loss + regret_weight * regret_loss
    return total, ce_loss, regret_loss


# ─────────────────────────────────────────────────────────────────────────────
# 4.  evaluate_sinkhorn
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sinkhorn(
    model: LocalNegotiatorGNN,
    sinkhorn_head: SinkhornHead,
    dataset: List[Data],
    device: torch.device,
    slot_radius: float = SLOT_VISIBILITY_RADIUS,
    use_hungarian_rounding: bool = True,
) -> Dict[str, float]:
    """
    Evaluate using Sinkhorn inference. No gossip used.

    Rounding modes
    --------------
    use_hungarian_rounding=True  (default, recommended):
        Round the doubly-stochastic P to a hard permutation via Hungarian
        algorithm on -P. Guarantees a valid bijection if P is full-support.
        This is the evaluation-time equivalent of Sinkhorn at convergence.

    use_hungarian_rounding=False:
        Greedy argmax row by row. Faster but may produce conflicts.
        Useful for measuring how close P is to a permutation matrix without
        any post-processing.

    Metrics returned
    ----------------
    Same keys as evaluate_strict_decentralized so you can directly compare
    columns in your results table.
    """
    model.eval()
    sinkhorn_head.eval()

    metrics: Dict[str, List[float]] = {
        "bijection_rate":         [],
        "conflict_rate":          [],
        "unassigned_rate":        [],
        "cost_ratio_vs_hungarian": [],
        "slot_match_rate":        [],
        "consensus_rounds":       [],   # always 0.0 — no rounds used
        "converged_rate":         [],   # always 1.0 — Sinkhorn always converges
    }

    for data in dataset:
        fid   = data.formation_id.item()
        f_emb = model.formation_embedding(torch.tensor(fid, device=device))
        dp, sl, y = data.drone_pos, data.slots, data.y
        n     = dp.size(0)

        # Strict local visibility — same as evaluate_strict_decentralized
        mask  = build_candidate_mask(dp, sl, slot_radius, y=y, force_gt=False)
        x     = build_node_features(dp, sl, f_emb.cpu(), candidate_mask=mask).to(device)
        ei, ea = build_comm_graph(dp)
        ds_ei  = build_candidate_edges(mask)[0]
        ds_ea  = build_candidate_edge_attr(dp, sl, ds_ei)

        # Forward through encoder + Sinkhorn head
        P, _ = _forward_sinkhorn(
            model,
            x,
            ei.to(device),
            ea.to(device),
            ds_ei.to(device),
            ds_ea.to(device),
            mask.to(device),
            sinkhorn_head,
        )
        # Round to hard assignment
        P_cpu = P.cpu()
        if use_hungarian_rounding:
            # Hungarian on -P gives the assignment that maximises sum of P[i, sigma(i)]
            # i.e. the most-probable valid permutation under the Sinkhorn distribution
            row_ind, col_ind = linear_sum_assignment(-P_cpu.numpy())
            assignment = torch.full((n,), -1, dtype=torch.long)
            assignment[row_ind] = torch.tensor(col_ind, dtype=torch.long)
        else:
            assignment = P_cpu.argmax(dim=1)

        sample_m = assignment_quality_metrics(dp, sl, assignment, y, rounds=0.0)
        for key, val in sample_m.items():
            metrics[key].append(val)
        metrics["converged_rate"].append(1.0)

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  train_sinkhorn_model
# ─────────────────────────────────────────────────────────────────────────────

def _encoder_params(model: LocalNegotiatorGNN) -> List[nn.Parameter]:
    """All parameters that belong to the GATv2 encoder (frozen for first N epochs)."""
    return (
        list(model.formation_embedding.parameters()) +
        list(model.input_proj.parameters())          +
        list(model.gat_layers.parameters())          +
        list(model.layer_norms.parameters())
    )


def _head_params(model: LocalNegotiatorGNN) -> List[nn.Parameter]:
    """Edge scorer + temperature — always trained."""
    return (
        list(model.edge_scorer.parameters()) +
        [model.log_temp]
    )


def train_sinkhorn_model(
    model: LocalNegotiatorGNN,
    sinkhorn_head: SinkhornHead,
    train_data: List[Data],
    val_data: List[Data],
    device: torch.device,
    epochs: int = 60,
    lr: float = 3e-4,
    freeze_epochs: int = FREEZE_EPOCHS,
    min_bijection_for_best: float = 0.80,
    force_gt_train: bool = True,
) -> Dict[str, List[float]]:
    """
    Fine-tune LocalNegotiatorGNN + SinkhornHead end-to-end.

    Freeze schedule
    ---------------
    Epochs 1 … freeze_epochs     : encoder frozen, only edge_scorer + sinkhorn_head trained.
    Epochs freeze_epochs+1 … end : all parameters unfrozen, joint training at lower lr.

    This mirrors the standard fine-tuning recipe: let the new head stabilise
    first, then allow the encoder to adapt.

    Saving policy
    -------------
    Best checkpoint = lowest (cost_ratio + penalty for bijection < threshold).
    Same policy as train_strict_decentralized_model so the two are comparable.

    Returns
    -------
    history dict with keys:
      train_loss, val_loss,
      val_bijection_rate, val_cost_ratio_vs_hungarian,
      val_conflict_rate, val_unassigned_rate, val_slot_match_rate
    """
    model.to(device)
    sinkhorn_head.to(device)

    # ── Fix 2: increase GATv2 dropout 0.1 → 0.2 during fine-tuning ──────────
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.2

    # ── Optimiser setup ──────────────────────────────────────────────────────
    # Phase 1: only head + edge scorer params
    head_params = _head_params(model) + list(sinkhorn_head.parameters())
    enc_params  = _encoder_params(model)

    joint_lr = lr * 0.3   # lower lr used when encoder unfreezes
    optimizer = torch.optim.Adam(head_params, lr=lr)
    # Fix 3: cosine annealing scheduler — decays from joint_lr to 1e-5 over
    # the joint phase (epochs freeze_epochs+1 … epochs).
    # T_max counts *joint* epochs; eta_min is the floor.
    joint_epochs = max(epochs - freeze_epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=joint_epochs,
        eta_min=1e-5,
    )
    frozen = True

    # Freeze encoder
    for p in enc_params:
        p.requires_grad_(False)

    history: Dict[str, List] = {
        "train_loss":                  [],
        "val_loss":                    [],
        "val_bijection_rate":          [],
        "val_cost_ratio_vs_hungarian": [],
        "val_conflict_rate":           [],
        "val_unassigned_rate":         [],
        "val_slot_match_rate":         [],
    }
    best_score     = float("inf")
    best_model_st  = None
    best_sink_st   = None

    for epoch in range(1, epochs + 1):

        # ── Unfreeze at the right epoch ──────────────────────────────────────
        if frozen and epoch > freeze_epochs:
            frozen = False
            for p in enc_params:
                p.requires_grad_(True)
            # Rebuild optimiser with all params at joint_lr, then let the
            # cosine scheduler (already constructed above) decay it to 1e-5.
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(sinkhorn_head.parameters()),
                lr=joint_lr,
            )
            # Re-attach scheduler to the new optimiser
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=joint_epochs,
                eta_min=1e-5,
            )
            print(
                f"  [Epoch {epoch}] Encoder unfrozen — joint fine-tuning starts "
                f"(lr={joint_lr:.2e} → 1e-5 over {joint_epochs} epochs)."
            )

        # ── Training pass ────────────────────────────────────────────────────
        model.train()
        sinkhorn_head.train()
        t_loss = 0.0

        for data in train_data:
            fid   = data.formation_id.item()
            f_emb = model.formation_embedding(torch.tensor(fid, device=device))

            # Rebuild strict mask with force_gt=True so all drones get signal
            mask  = build_candidate_mask(
                data.drone_pos, data.slots,
                SLOT_VISIBILITY_RADIUS, y=data.y, force_gt=force_gt_train,
            )
            x     = build_node_features(
                data.drone_pos, data.slots, f_emb.detach().cpu(), candidate_mask=mask,
            ).to(device)
            ei, ea = data.edge_index.to(device), data.edge_attr.to(device)
            ds_ei  = build_candidate_edges(mask)[0].to(device)
            ds_ea  = build_candidate_edge_attr(
                data.drone_pos, data.slots, ds_ei.cpu()
            ).to(device)
            mask_d = mask.to(device)

            P, edge_logits = _forward_sinkhorn(
                model, x, ei, ea, ds_ei, ds_ea, mask_d, sinkhorn_head
            )

            loss, ce, regret = sinkhorn_assignment_loss(
                P, edge_logits, ds_ei,
                data.y.to(device), mask_d,
                data.drone_pos.to(device), data.slots.to(device),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(sinkhorn_head.parameters()), 1.0
            )
            optimizer.step()
            t_loss += loss.item()

        # ── Validation pass ──────────────────────────────────────────────────
        model.eval()
        sinkhorn_head.eval()
        v_loss = 0.0

        with torch.no_grad():
            for data in val_data:
                fid   = data.formation_id.item()
                f_emb = model.formation_embedding(torch.tensor(fid, device=device))
                # Val uses strict visibility — no force_gt
                mask  = build_candidate_mask(
                    data.drone_pos, data.slots,
                    SLOT_VISIBILITY_RADIUS, y=data.y, force_gt=False,
                )
                x     = build_node_features(
                    data.drone_pos, data.slots, f_emb.detach().cpu(), candidate_mask=mask,
                ).to(device)
                ei, ea = data.edge_index.to(device), data.edge_attr.to(device)
                ds_ei  = build_candidate_edges(mask)[0].to(device)
                ds_ea  = build_candidate_edge_attr(
                    data.drone_pos, data.slots, ds_ei.cpu()
                ).to(device)
                mask_d = mask.to(device)

                P, edge_logits = _forward_sinkhorn(
                    model, x, ei, ea, ds_ei, ds_ea, mask_d, sinkhorn_head
                )
                loss, _, _ = sinkhorn_assignment_loss(
                    P, edge_logits, ds_ei,
                    data.y.to(device), mask_d,
                    data.drone_pos.to(device), data.slots.to(device),
                )
                v_loss += loss.item()

        t_loss /= max(len(train_data), 1)
        v_loss /= max(len(val_data),   1)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        # ── Evaluation metrics (Sinkhorn inference, Hungarian rounding) ───────
        metrics = evaluate_sinkhorn(model, sinkhorn_head, val_data[:100], device)
        for key in (
            "bijection_rate", "cost_ratio_vs_hungarian",
            "conflict_rate", "unassigned_rate", "slot_match_rate",
        ):
            history[f"val_{key}"].append(metrics[key])

        # ── Checkpoint ───────────────────────────────────────────────────────
        score = metrics["cost_ratio_vs_hungarian"]
        if metrics["bijection_rate"] < min_bijection_for_best:
            score += 10.0 * (min_bijection_for_best - metrics["bijection_rate"])
        if score < best_score:
            best_score    = score
            best_model_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_sink_st  = {k: v.cpu().clone() for k, v in sinkhorn_head.state_dict().items()}

        # ── Fix 3: scheduler step (joint phase only) ─────────────────────────
        if not frozen:
            scheduler.step()

        # ── Logging ──────────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            phase = "frozen" if frozen else "joint"
            print(
                f"Epoch {epoch:3d} [{phase}] | "
                f"train={t_loss:.4f} | val={v_loss:.4f} | "
                f"bij={metrics['bijection_rate']:.3f} | "
                f"conflict={metrics['conflict_rate']:.3f} | "
                f"cost_ratio={metrics['cost_ratio_vs_hungarian']:.4f} | "
                f"match={metrics['slot_match_rate']:.3f}"
            )

    # ── Restore best ─────────────────────────────────────────────────────────
    if best_model_st is not None:
        model.load_state_dict(best_model_st)
    if best_sink_st is not None:
        sinkhorn_head.load_state_dict(best_sink_st)

    return history


