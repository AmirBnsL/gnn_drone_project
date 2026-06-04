"""
superglue_negotiator.py
=======================
SuperGlue-style cross-attentional matcher for drone-to-slot assignment.

The model is meant to replace the LocalNegotiatorGNN encoder as a stronger
initializer for auction-style solvers.  It predicts:

  * V:   dense drone-slot value matrix, masked by candidate visibility;
  * eps: positive per-slot auction step sizes.

Hungarian labels are used only for supervised training through data.y.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import values
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


HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 6
FORMATION_EMB_DIM = 16
VALUE_SCALE = 10.0
EPS_MIN = 1e-3
EPS_MAX = 1.0
CE_WEIGHT = 1.0
REGRET_WEIGHT = 0.40   # was 0.25 — push harder on wrong assignments
MARGIN_WEIGHT = 0.20   # was 0.10 — enforce larger gap between gt slot and competitors
EPS_WEIGHT = 0.05
COVERAGE_WEIGHT = 0.50 # was 0.35 — every drone must have at least one visible slot
MARGIN = 0.20
MIN_BIJ_FOR_BEST = 0.90


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class MaskedMultiheadAttention(nn.Module):
    """
    Batch-free multi-head attention for variable-size PyG samples.

    q: (Nq, D), k/v: (Nk, D), mask: optional bool (Nq, Nk), True = allowed.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        nq = q.size(0)
        nk = k.size(0)
        if nq == 0 or nk == 0:
            return q

        qh = self.q_proj(q).view(nq, self.heads, self.head_dim).transpose(0, 1)
        kh = self.k_proj(k).view(nk, self.heads, self.head_dim).transpose(0, 1)
        vh = self.v_proj(v).view(nk, self.heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask_h = mask.to(q.device).unsqueeze(0)
            scores = scores.masked_fill(~mask_h, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        msg = torch.matmul(attn, vh).transpose(0, 1).reshape(nq, self.dim)
        return self.norm(q + self.out_proj(msg))


def edge_index_to_dense_mask(edge_index: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    mask = torch.eye(n, dtype=torch.bool, device=device)
    if edge_index.numel() > 0:
        src = edge_index[0].to(device)
        dst = edge_index[1].to(device)
        valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
        mask[src[valid], dst[valid]] = True
    return mask


class SuperGlueLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.drone_self = MaskedMultiheadAttention(dim, heads)
        self.slot_self = MaskedMultiheadAttention(dim, heads)
        self.drone_cross = MaskedMultiheadAttention(dim, heads)
        self.slot_cross = MaskedMultiheadAttention(dim, heads)
        self.drone_ff = FeedForwardBlock(dim)
        self.slot_ff = FeedForwardBlock(dim)

    def forward(
        self,
        drone_h: torch.Tensor,
        slot_h: torch.Tensor,
        drone_self_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_slots = slot_h.size(0)
        slot_self_mask = torch.ones(n_slots, n_slots, dtype=torch.bool, device=slot_h.device)

        drone_h = self.drone_self(drone_h, drone_h, drone_h, drone_self_mask)
        slot_h = self.slot_self(slot_h, slot_h, slot_h, slot_self_mask)

        drone_h = self.drone_cross(drone_h, slot_h, slot_h, candidate_mask)
        slot_h = self.slot_cross(slot_h, drone_h, drone_h, candidate_mask.transpose(0, 1))

        return self.drone_ff(drone_h), self.slot_ff(slot_h)


class SuperGlueSwarmMatcher(nn.Module):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        num_formations: int = NUM_FORMATIONS,
        formation_emb_dim: int = FORMATION_EMB_DIM,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.formation_embedding = nn.Embedding(num_formations, formation_emb_dim)
        self.price_feat_dim = 3
        self.drone_mlp = MLP(2 + formation_emb_dim + self.price_feat_dim, hidden_dim, hidden_dim)
        # Initialise price feature weights to zero so that when local_prices=None
        # (zero features padded), behaviour is identical to the original model.
        # This means a checkpoint trained without price features can be loaded and
        # fine-tuned — the price feature columns start neutral.
        with torch.no_grad():
            self.drone_mlp.net[0].weight[:, 2 + formation_emb_dim:].zero_()
        self.slot_mlp = MLP(2 + formation_emb_dim, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [SuperGlueLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.eps_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        drone_pos: torch.Tensor,
        slots: torch.Tensor,
        formation_id: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_index: torch.Tensor,
        local_prices: Optional[torch.Tensor] = None,   # (N, N) each row is drone i's price view
        local_owner: Optional[torch.Tensor] = None,    # (N, N) long
        claims: Optional[torch.Tensor] = None,         # (N,) long
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = drone_pos.device
        n_drones = drone_pos.size(0)
        fid = formation_id.to(device).long().view(-1)[0]
        f_emb = self.formation_embedding(fid).unsqueeze(0)

        n_slots = slots.size(0)
        drone_in = torch.cat([drone_pos, f_emb.expand(n_drones, -1)], dim=1)

        if local_prices is not None and local_owner is not None and claims is not None:
            # Compute 3 purely local price features per drone.
            # Each drone reads only its own row — no cross-drone reads.
            lp = local_prices.to(device).float()          # (N, N)
            lo = local_owner.to(device)                    # (N, N) long
            cl = claims.to(device)                         # (N,) long
            mask_bool = candidate_mask.to(device).bool()
            mask_f = mask_bool.float()

            # Feature 0: price of claimed slot (0 if no claim)
            claimed_price = torch.zeros(n_drones, device=device)
            valid_claim = cl >= 0
            if valid_claim.any():
                rows = torch.arange(n_drones, device=device)[valid_claim]
                claimed_price[valid_claim] = lp[rows, cl[valid_claim]]

            # Feature 1: mean price of visible slots in drone's local view
            counts = mask_f.sum(dim=1).clamp(min=1.0)
            mean_price = (lp * mask_f).sum(dim=1) / counts

            # Feature 2: fraction of visible slots with a known owner
            owned_frac = ((lo >= 0) & mask_bool).float().sum(dim=1) / counts

            price_feats = torch.stack([claimed_price, mean_price, owned_frac], dim=1)  # (N, 3)
            drone_in = torch.cat([drone_in, price_feats], dim=1)                        # (N, 2+emb+3)
        else:
            # No local state: pad with zeros so drone_mlp always receives the
            # same input width regardless of whether state was provided.
            price_feats = torch.zeros(n_drones, self.price_feat_dim, device=device)
            drone_in = torch.cat([drone_in, price_feats], dim=1)

        slot_in = torch.cat([slots, f_emb.expand(n_slots, -1)], dim=1)
        drone_h = self.drone_mlp(drone_in)
        slot_h = self.slot_mlp(slot_in)

        drone_self_mask = edge_index_to_dense_mask(edge_index, n_drones, device)
        cross_mask = candidate_mask.to(device).bool()

        for layer in self.layers:
            drone_h, slot_h = layer(drone_h, slot_h, drone_self_mask, cross_mask)

        d = drone_h.unsqueeze(1).expand(n_drones, n_slots, -1)
        s = slot_h.unsqueeze(0).expand(n_drones, n_slots, -1)
        rel = slots.unsqueeze(0) - drone_pos.unsqueeze(1)
        dist = rel.norm(dim=-1, keepdim=True)
        pair = torch.cat([d, s, rel, dist], dim=-1)

        values = VALUE_SCALE * torch.tanh(self.value_head(pair).squeeze(-1))
        values = values.masked_fill(~cross_mask, -1e9)

        raw_eps = self.eps_head(slot_h).squeeze(-1)
        eps = EPS_MIN + (EPS_MAX - EPS_MIN) * torch.sigmoid(raw_eps)
        return values, eps


def masked_row_ce(values: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Vectorised: mask invisible slots with -inf so softmax ignores them,
    # then run a single batched F.cross_entropy — one autograd node, ~N× faster.
    device = values.device
    mask_b = mask.to(device).bool()
    tgt = targets.to(device).long()
    # Only include drones that have at least one visible slot AND whose GT slot
    # is visible (otherwise the CE target is undefined).
    row_idx = torch.arange(tgt.numel(), device=device)
    valid = mask_b.any(dim=1) & mask_b[row_idx, tgt]
    if not bool(valid.any()):
        return values.sum() * 0.0
    logits = values[valid].masked_fill(~mask_b[valid], -1e9)
    return F.cross_entropy(logits, tgt[valid])


def soft_regret_loss(
    values: torch.Tensor,
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    masked = values.masked_fill(~mask.to(values.device), -1e9)
    probs = torch.softmax(masked, dim=1)
    probs = torch.nan_to_num(probs, nan=0.0)
    dist = torch.cdist(drone_pos.to(values.device), slots.to(values.device))
    expected = (probs * dist).sum(dim=1)
    opt = dist[torch.arange(targets.numel(), device=values.device), targets.to(values.device)]
    visible = mask.to(values.device).any(dim=1)
    if not bool(visible.any()):
        return values.sum() * 0.0
    return F.relu(expected - opt).masked_select(visible).mean()


def margin_loss(values: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, margin: float = MARGIN) -> torch.Tensor:
    # Vectorised: compute target value and best-other value in parallel.
    device = values.device
    mask_b = mask.to(device).bool()
    tgt = targets.to(device).long()
    row_idx = torch.arange(tgt.numel(), device=device)
    # Only drones whose GT slot is visible
    valid = mask_b[row_idx, tgt]
    if not bool(valid.any()):
        return values.sum() * 0.0
    masked = values.masked_fill(~mask_b, -1e9)
    target_vals = masked[row_idx[valid], tgt[valid]]
    # Mask out the target slot to find best competitor
    others = masked[valid].clone()
    others[torch.arange(valid.sum(), device=device), tgt[valid]] = -1e9
    best_other = others.max(dim=1).values
    # Exclude rows where there is no visible competitor
    has_competitor = best_other > -1e8
    if not bool(has_competitor.any()):
        return values.sum() * 0.0
    loss = F.relu(margin - (target_vals[has_competitor] - best_other[has_competitor]))
    return loss.mean()


def epsilon_margin_loss(values: torch.Tensor, eps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask.to(values.device), -1e9)
    k = min(2, values.size(1))
    if k < 2:
        return eps.sum() * 0.0
    top2 = masked.topk(k=2, dim=1).values
    drone_margin = (top2[:, 0] - top2[:, 1]).clamp(min=EPS_MIN, max=EPS_MAX)
    mask_f = mask.to(values.device).float()
    # For each slot j, only use margins from drones whose top choice IS slot j.
    # This gives the correct ε target: the price increment needed to resolve
    # competition among drones that actually want this slot.
    top_choice = values.masked_fill(~mask.to(values.device), -1e9).argmax(dim=1)  # (N_drones,)
    top_choice_mask = torch.zeros_like(mask_f)  # (N_drones, N_slots)
    top_choice_mask.scatter_(1, top_choice.unsqueeze(1), 1.0)
    top_choice_mask = top_choice_mask * mask_f  # only for visible slots
    counts = top_choice_mask.sum(dim=0).clamp(min=1.0)
    slot_target = ((top_choice_mask * drone_margin.unsqueeze(1)).sum(dim=0) / counts).detach()
    slot_target = slot_target.clamp(min=EPS_MIN, max=EPS_MAX)
    return F.mse_loss(torch.log(eps.clamp(min=EPS_MIN)), torch.log(slot_target))


def coverage_loss(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_d = mask.to(values.device).bool()
    visible_rows = mask_d.any(dim=1)
    if not bool(visible_rows.any()):
        return values.sum() * 0.0
    v_max = values.masked_fill(~mask_d, -1e9).max(dim=1).values
    return F.relu(-v_max).masked_select(visible_rows).mean()


def superglue_loss(
    values: torch.Tensor,
    eps: torch.Tensor,
    data: Data,
    candidate_mask: torch.Tensor,
    ce_weight: float = CE_WEIGHT,
    regret_weight: float = REGRET_WEIGHT,
    margin_weight: float = MARGIN_WEIGHT,
    eps_weight: float = EPS_WEIGHT,
    coverage_weight: float = COVERAGE_WEIGHT,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    targets = data.y.to(values.device).long()
    mask = candidate_mask.to(values.device).bool()
    ce = masked_row_ce(values, targets, mask)
    regret = soft_regret_loss(values, data.drone_pos, data.slots, targets, mask)
    marg = margin_loss(values, targets, mask)
    eps_l = epsilon_margin_loss(values, eps, mask)
    cov = coverage_loss(values, mask)

    # Diversity loss: penalise column collapse — encourages the model to spread
    # probability mass across drones per slot, not assign all drones to one slot.
    # col_probs[i, j] = softmax over drones for slot j (viewed from slot's perspective).
    col_probs = torch.softmax(values.masked_fill(~mask, -1e9), dim=0)
    col_entropy = -(col_probs * (col_probs + 1e-9).log()).sum(dim=0)  # (S,)
    max_ent = torch.log(torch.tensor(values.size(0), dtype=torch.float, device=values.device))
    diversity_loss = (max_ent - col_entropy).clamp(min=0.0).mean()

    total = (
        ce_weight * ce
        + regret_weight * regret
        + margin_weight * marg
        + eps_weight * eps_l
        + coverage_weight * cov
        + 0.10 * diversity_loss
    )
    # Column contention penalty: each slot should attract exactly one drone.
    # sum of softmax probabilities over column j should be close to 1.
    row_masked = values.masked_fill(~candidate_mask.to(values.device), -1e9)
    soft_assign = torch.softmax(row_masked, dim=1)          # [N, N] row-normalized
    col_load = soft_assign.sum(dim=0)                       # [N] ideal = 1.0
    contention_loss = F.mse_loss(col_load,
                                torch.ones_like(col_load))
    total = total + 0.30 * contention_loss
    return total, {"ce": ce, "regret": regret, "margin": marg, "eps": eps_l, "coverage": cov}


@torch.no_grad()
def greedy_assignment(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n = values.size(0)
    score = values.cpu().masked_fill(~mask.cpu(), -1e9)
    assignment = torch.full((n,), -1, dtype=torch.long)
    taken = set()
    order = torch.argsort(score.max(dim=1).values, descending=True).tolist()
    for drone in order:
        row = score[drone].clone()
        for slot in taken:
            row[slot] = -1e9
        if row.max().item() <= -1e8:
            continue
        slot = int(row.argmax().item())
        assignment[drone] = slot
        taken.add(slot)
    return assignment


@torch.no_grad()
def sinkhorn_round_assignment(
    values: torch.Tensor,
    mask: torch.Tensor,
    iters: int = 20,
) -> torch.Tensor:
    device = values.device
    mask_d = mask.to(device).bool()
    log_alpha = values.masked_fill(~mask_d, -1e9)

    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(
            log_alpha.masked_fill(~mask_d, -1e9), dim=1, keepdim=True
        )
        log_alpha = log_alpha - torch.logsumexp(
            log_alpha.masked_fill(~mask_d, -1e9), dim=0, keepdim=True
        )
        log_alpha = log_alpha.masked_fill(~mask_d, -1e9)

    probs = torch.exp(log_alpha).masked_fill(~mask_d, 0.0)
    assignment = probs.argmax(dim=1).cpu()
    visible = mask_d.any(dim=1).cpu()
    assignment = assignment.masked_fill(~visible, -1)
    return assignment.long()


@torch.no_grad()
def evaluate_superglue_matcher(
    model: SuperGlueSwarmMatcher,
    dataset: List[Data],
    device: torch.device,
    slot_radius: float = SLOT_VISIBILITY_RADIUS,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, List[float]] = {
        "bijection_rate": [],
        "conflict_rate": [],
        "unassigned_rate": [],
        "cost_ratio_vs_hungarian": [],
        "slot_match_rate": [],
        "consensus_rounds": [],
        "mean_eps": [],
    }
    for data in dataset:
        mask = getattr(data, "candidate_mask", None)
        if mask is None:
            mask = build_candidate_mask(data.drone_pos, data.slots, slot_radius, y=data.y, force_gt=False)
        values, eps = model(
            data.drone_pos.to(device).float(),
            data.slots.to(device).float(),
            data.formation_id.to(device),
            mask.to(device),
            data.edge_index.to(device),
        )
        assignment = sinkhorn_round_assignment(values, mask)
        sample = assignment_quality_metrics(data.drone_pos, data.slots, assignment, data.y, rounds=0.0)
        for key, value in sample.items():
            metrics[key].append(value)
        metrics["mean_eps"].append(float(eps.mean().item()))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def train_superglue_matcher(
    model: SuperGlueSwarmMatcher,
    train_data: List[Data],
    val_data: List[Data],
    device: torch.device,
    epochs: int = 80,
    lr: float = 3e-4,
    min_bijection_for_best: float = MIN_BIJ_FOR_BEST,
    force_gt_train: bool = True,
    eval_subset: int = 200,
    ckpt_path: Optional[str] = "superglue_negotiator_best.pt",
    patience: int = 15,
) -> Dict[str, List[float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_bijection_rate": [],
        "val_cost_ratio_vs_hungarian": [],
        "val_slot_match_rate": [],
        "val_mean_eps": [],
        "lr": [],
    }
    best_score = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        losses: List[float] = []

        for data in train_data:
            mask = build_candidate_mask(
                data.drone_pos,
                data.slots,
                SLOT_VISIBILITY_RADIUS,
                y=data.y,
                force_gt=force_gt_train,
            )
            values, eps = model(
                data.drone_pos.to(device).float(),
                data.slots.to(device).float(),
                data.formation_id.to(device),
                mask.to(device),
                data.edge_index.to(device),
            )
            loss, _ = superglue_loss(values, eps, data, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        scheduler.step()
        history["train_loss"].append(float(np.mean(losses)) if losses else 0.0)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for data in val_data[:eval_subset]:
                mask = build_candidate_mask(
                    data.drone_pos,
                    data.slots,
                    SLOT_VISIBILITY_RADIUS,
                    y=data.y,
                    force_gt=False,
                )
                values, eps = model(
                    data.drone_pos.to(device).float(),
                    data.slots.to(device).float(),
                    data.formation_id.to(device),
                    mask.to(device),
                    data.edge_index.to(device),
                )
                val_loss, _ = superglue_loss(values, eps, data, mask)
                val_losses.append(float(val_loss.item()))

        val_metrics = evaluate_superglue_matcher(model, val_data[:eval_subset], device)
        history["val_loss"].append(float(np.mean(val_losses)) if val_losses else 0.0)
        for key in ("bijection_rate", "cost_ratio_vs_hungarian", "slot_match_rate", "mean_eps"):
            history[f"val_{key}"].append(val_metrics[key])
        # Early stopping: stop when val_loss has not improved for `patience` consecutive epochs.
        # All history arrays are appended before this check so they stay in sync on break.
        if len(history["val_loss"]) > patience:
            recent_best = min(history["val_loss"][:-patience])
            if all(v >= recent_best for v in history["val_loss"][-patience:]):
                print(f"  Early stopping at epoch {epoch} — val loss has not improved for {patience} epochs.")
                break

        score = val_metrics["cost_ratio_vs_hungarian"]
        if val_metrics["bijection_rate"] < min_bijection_for_best:
            score += 10.0 * (min_bijection_for_best - val_metrics["bijection_rate"])
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            if ckpt_path:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "metrics": val_metrics,
                        "epoch": epoch,
                    },
                    ckpt_path,
                )

        print(
            f"epoch {epoch:03d} "
            f"train={history['train_loss'][-1]:.4f} "
            f"val={history['val_loss'][-1]:.4f} "
            f"bij={val_metrics['bijection_rate']:.3f} "
            f"cost={val_metrics['cost_ratio_vs_hungarian']:.4f} "
            f"match={val_metrics['slot_match_rate']:.3f} "
            f"eps={val_metrics['mean_eps']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "./dataset/negotiator_dataset_v1.pt"
    raw = load_negotiator_dataset(dataset_path)

    temp_model = SuperGlueSwarmMatcher()
    train_data, val_data, test_data = prepare_dataset(
        raw,
        temp_model.formation_embedding.weight.detach().cpu(),
        seed=seed,
        force_gt_visibility=False,
    )
    model = temp_model.to(device)
    history = train_superglue_matcher(
        model,
        train_data[:7000],
        val_data[:500],
        device,
        epochs=80,
        lr=3e-4,
        ckpt_path="superglue_negotiator_best.pt",
    )
    test_metrics = evaluate_superglue_matcher(model, test_data[:500], device)
    print("test:", test_metrics)
