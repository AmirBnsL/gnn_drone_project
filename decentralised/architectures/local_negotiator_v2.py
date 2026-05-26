"""
Decentralized Hungarian-style drone-to-slot assignment.

Each drone scores only locally visible slots, then a multi-hop gossip protocol
resolves conflicts without a central Hungarian solver at inference time.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

# ── Default hyperparameters ───────────────────────────────────────────────────
NUM_FORMATIONS = 6
FORMATION_EMB_DIM = 16
TOP_K_SLOTS = 3
TOP_K_FEAT_DIM = TOP_K_SLOTS * 3
NODE_FEAT_DIM = 2 + FORMATION_EMB_DIM + TOP_K_FEAT_DIM
EDGE_FEAT_DIM = 2
HIDDEN_DIM = 64
NUM_GAT_HEADS = 4
NUM_GNN_LAYERS = 3
COMM_RADIUS = 4.5  # raised from 3.0 — at 3.0, ~6% of drones had zero neighbours in a
# 10m arena with N∈[8,20], making conflict resolution via gossip impossible for them.
SLOT_VISIBILITY_RADIUS = 5.0
MAX_GOSSIP_ROUNDS = 8
GOSSIP_FLOOD_STEPS = 5
MAX_CONSENSUS_ROUNDS = 12
CONFLICT_LAMBDA = 0.5
REGRET_LAMBDA = 0.15
FORCE_GT_VISIBILITY_TRAIN = True
STRICT_FORCE_GT_VISIBILITY_TRAIN = False


def hungarian_assignment(drone_pos: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    cost = torch.cdist(drone_pos, slots).numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = np.empty(drone_pos.size(0), dtype=np.int64)
    assignment[row_ind] = col_ind
    return torch.tensor(assignment, dtype=torch.long)


def build_candidate_mask(
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    radius: float = SLOT_VISIBILITY_RADIUS,
    y: Optional[torch.Tensor] = None,
    force_gt: bool = FORCE_GT_VISIBILITY_TRAIN,
) -> torch.Tensor:
    """mask[i, j] == True if drone i can bid on slot j."""
    dist = torch.cdist(drone_pos, slots)
    mask = dist <= radius
    if y is not None and force_gt:
        mask[torch.arange(mask.size(0)), y] = True
    return mask


def build_comm_graph(
    drone_pos: torch.Tensor,
    comm_radius: float = COMM_RADIUS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = drone_pos.size(0)
    rows, cols, attrs = [], [], []
    connected: Set[int] = set()
    for i in range(n):
        for j in range(i + 1, n):
            diff = drone_pos[j] - drone_pos[i]
            if diff.norm().item() <= comm_radius:
                rows.extend([i, j])
                cols.extend([j, i])
                attrs.extend([diff, -diff])
                connected.add(i)
                connected.add(j)

    # BUG 1 FIX: drones with zero neighbours inside comm_radius cannot receive
    # any gossip and cannot resolve conflicts. Add a self-loop (zero displacement)
    # so the GAT layer still produces a valid output for them. This preserves the
    # decentralised constraint — a self-loop carries no information from others.
    zero = torch.zeros(2, dtype=drone_pos.dtype)
    for i in range(n):
        if i not in connected:
            rows.extend([i, i])
            cols.extend([i, i])
            attrs.extend([zero, zero])

    if not rows:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 2), dtype=torch.float),
        )
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.stack(attrs, dim=0)
    return edge_index, edge_attr


def build_candidate_edges(
    candidate_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drone→slot candidate edges: (2, E) with rows [drone_i, slot_j], rel (dx, dy, dist)."""
    rows, cols, attrs = [], [], []
    n = candidate_mask.size(0)
    for i in range(n):
        for j in range(n):
            if candidate_mask[i, j].item():
                rows.append(i)
                cols.append(j)
    if not rows:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 3), dtype=torch.float),
        )
    drone_idx = torch.tensor(rows, dtype=torch.long)
    slot_idx = torch.tensor(cols, dtype=torch.long)
    return torch.stack([drone_idx, slot_idx], dim=0), drone_idx  # slot_idx stored separately below


def build_candidate_edge_attr(
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    ds_edge_index: torch.Tensor,
) -> torch.Tensor:
    if ds_edge_index.size(1) == 0:
        return torch.zeros((0, 3), dtype=torch.float)
    i = ds_edge_index[0]
    j = ds_edge_index[1]
    diff = slots[j] - drone_pos[i]
    dist = diff.norm(dim=-1, keepdim=True)
    return torch.cat([diff, dist], dim=-1)


def build_node_features(
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    formation_embedding: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    top_k: int = TOP_K_SLOTS,
) -> torch.Tensor:
    n = drone_pos.size(0)
    cog = drone_pos.mean(dim=0, keepdim=True)
    pos_to_com = drone_pos - cog
    emb_broadcast = formation_embedding.unsqueeze(0).expand(n, -1)

    diff = slots.unsqueeze(0) - drone_pos.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    if candidate_mask is not None:
        dist = dist.masked_fill(~candidate_mask, 1e6)

    k = min(top_k, dist.size(1))
    _, topk_idx = torch.topk(dist, k=k, dim=1, largest=False)
    topk_dx = diff[:, :, 0].gather(1, topk_idx)
    topk_dy = diff[:, :, 1].gather(1, topk_idx)
    topk_dist = dist.gather(1, topk_idx)
    top3_slots_info = torch.stack([topk_dx, topk_dy, topk_dist], dim=2).reshape(n, -1)
    if top3_slots_info.size(1) < TOP_K_FEAT_DIM:
        pad = torch.zeros(n, TOP_K_FEAT_DIM - top3_slots_info.size(1))
        top3_slots_info = torch.cat([top3_slots_info, pad], dim=1)

    return torch.cat([pos_to_com, emb_broadcast, top3_slots_info[:, :TOP_K_FEAT_DIM]], dim=1)


def edge_logits_to_dense(
    edge_logits: torch.Tensor,
    ds_edge_index: torch.Tensor,
    n: int,
    fill: float = -1e4,
) -> torch.Tensor:
    logits = torch.full((n, n), fill, dtype=edge_logits.dtype, device=edge_logits.device)
    if ds_edge_index.size(1) > 0:
        logits[ds_edge_index[0], ds_edge_index[1]] = edge_logits
    return logits


def sample_to_pyg(
    raw: Data,
    formation_emb_weight: torch.Tensor,
    comm_radius: float = COMM_RADIUS,
    slot_radius: float = SLOT_VISIBILITY_RADIUS,
    force_gt_visibility: bool = FORCE_GT_VISIBILITY_TRAIN,
) -> Data:
    drone_pos = raw.drone_pos.float()
    slots = raw.slots.float()
    y = raw.y.long()
    fid = int(raw.formation_id.item()) if raw.formation_id.dim() == 0 else int(raw.formation_id)
    n = drone_pos.size(0)

    candidate_mask = build_candidate_mask(
        drone_pos, slots, slot_radius, y=y, force_gt=force_gt_visibility
    )
    f_emb = formation_emb_weight[fid].detach().cpu()
    x = build_node_features(drone_pos, slots, f_emb, candidate_mask=candidate_mask)
    edge_index, edge_attr = build_comm_graph(drone_pos, comm_radius)
    ds_edge_index = build_candidate_edges(candidate_mask)[0]
    ds_edge_attr = build_candidate_edge_attr(drone_pos, slots, ds_edge_index)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        drone_pos=drone_pos,
        slots=slots,
        formation_id=torch.tensor(fid, dtype=torch.long),
        candidate_mask=candidate_mask,
        ds_edge_index=ds_edge_index,
        ds_edge_attr=ds_edge_attr,
        num_nodes=n,
    )


def load_negotiator_dataset(
    path: str = "./dataset/negotiator_dataset_v1.pt",
) -> List[Data]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run data_gen/drone_swarm_datagen.ipynb first."
        )
    return torch.load(path, weights_only=False)


def prepare_dataset(
    raw_list: List[Data],
    formation_emb_weight: torch.Tensor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    force_gt_visibility: bool = FORCE_GT_VISIBILITY_TRAIN,
) -> Tuple[List[Data], List[Data], List[Data]]:
    rng = random.Random(seed)
    indices = list(range(len(raw_list)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def convert(idxs):
        return [
            sample_to_pyg(
                raw_list[i],
                formation_emb_weight,
                force_gt_visibility=force_gt_visibility,
            )
            for i in idxs
        ]

    return convert(train_idx), convert(val_idx), convert(test_idx)


class LocalNegotiatorGNN(nn.Module):
    """GAT on comm graph + sparse drone→slot edge MLP scoring."""

    def __init__(
        self,
        num_formations: int = NUM_FORMATIONS,
        formation_emb_dim: int = FORMATION_EMB_DIM,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_GAT_HEADS,
        num_layers: int = NUM_GNN_LAYERS,
        ds_edge_feat_dim: int = 3,
    ):
        super().__init__()
        self.formation_embedding = nn.Embedding(num_formations, formation_emb_dim)
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        head_dim = hidden_dim // num_heads
        self.gat_layers = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    edge_dim=edge_feat_dim,
                    concat=True,
                    dropout=0.1,
                    add_self_loops=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim + ds_edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_temp = nn.Parameter(torch.zeros(1))

    def encode_drones(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            if edge_index.size(1) > 0:
                h_new = gat(h, edge_index, edge_attr=edge_attr)
            else:
                h_new = torch.zeros_like(h)
            h = ln(h + F.elu(h_new))
        return h

    def forward_sparse(
        self,
        drone_hidden: torch.Tensor,
        ds_edge_index: torch.Tensor,
        ds_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if ds_edge_index.size(1) == 0:
            return torch.zeros(0, device=drone_hidden.device)
        i = ds_edge_index[0]
        h_i = drone_hidden[i]
        temp = torch.exp(self.log_temp).clamp(min=1e-3)
        edge_in = torch.cat([h_i, ds_edge_attr], dim=-1)
        return self.edge_scorer(edge_in).squeeze(-1) / temp

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        ds_edge_index: torch.Tensor,
        ds_edge_attr: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        return_dense: bool = False,
    ):
        h = self.encode_drones(x, edge_index, edge_attr)
        edge_logits = self.forward_sparse(h, ds_edge_index, ds_edge_attr)
        if return_dense and candidate_mask is not None:
            n = candidate_mask.size(0)
            return edge_logits_to_dense(edge_logits, ds_edge_index, n), edge_logits
        return edge_logits


def conflict_penalty(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=logits.device)
    if candidate_mask is not None:
        probs = F.softmax(logits.masked_fill(~candidate_mask, -1e4), dim=1)
    else:
        probs = F.softmax(logits, dim=1)
    probs = torch.nan_to_num(probs, nan=0.0)
    src, dst = edge_index[0], edge_index[1]
    sim = (probs[src] * probs[dst]).sum(dim=1)
    return sim.mean()


def sparse_cross_entropy(
    dense: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """CE per drone over visible slots only (stable; no -inf rows)."""
    n = targets.size(0)
    losses = []
    for i in range(n):
        vis = candidate_mask[i].nonzero(as_tuple=True)[0]
        if vis.numel() == 0:
            continue
        row = dense[i, vis]
        tgt_slot = targets[i].item()
        local_idx = (vis == tgt_slot).nonzero(as_tuple=True)[0]
        if local_idx.numel() == 0:
            continue
        tgt_class = torch.tensor([local_idx[0].item()], device=row.device, dtype=torch.long)
        losses.append(F.cross_entropy(row.unsqueeze(0), tgt_class))
    if not losses:
        return dense.sum() * 0.0
    return torch.stack(losses).mean()


def sparse_negotiator_loss(
    edge_logits: torch.Tensor,
    ds_edge_index: torch.Tensor,
    targets: torch.Tensor,
    comm_edge_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    lambda_conflict: float = CONFLICT_LAMBDA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = targets.size(0)
    dense = edge_logits_to_dense(edge_logits, ds_edge_index, n)
    dense = dense.masked_fill(~candidate_mask, -1e4)
    ce_loss = sparse_cross_entropy(dense, targets, candidate_mask)
    conf_loss = conflict_penalty(dense, comm_edge_index, candidate_mask)
    if torch.isnan(ce_loss):
        ce_loss = dense.sum() * 0.0
    if torch.isnan(conf_loss):
        conf_loss = dense.sum() * 0.0
    total = ce_loss + lambda_conflict * conf_loss
    return total, ce_loss, conf_loss


def cost_regret_penalty(
    dense: torch.Tensor,
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Expected local travel cost relative to the Hungarian target cost."""
    masked = dense.masked_fill(~candidate_mask, -1e4)
    probs = F.softmax(masked, dim=1)
    probs = torch.nan_to_num(probs, nan=0.0)
    dist = torch.cdist(drone_pos.to(dense.device), slots.to(dense.device))
    expected = (probs * dist).sum(dim=1)
    target_cost = dist[torch.arange(targets.numel(), device=dense.device), targets.to(dense.device)]
    visible_rows = candidate_mask.to(dense.device).any(dim=1)
    if not bool(visible_rows.any()):
        return dense.sum() * 0.0
    regret = F.relu(expected - target_cost).masked_select(visible_rows)
    return regret.mean()


def strict_assignment_loss(
    edge_logits: torch.Tensor,
    ds_edge_index: torch.Tensor,
    targets: torch.Tensor,
    comm_edge_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    lambda_conflict: float = CONFLICT_LAMBDA,
    lambda_regret: float = REGRET_LAMBDA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assignment-aware sparse loss for strict local training."""
    n = targets.size(0)
    dense = edge_logits_to_dense(edge_logits, ds_edge_index, n)
    dense = dense.masked_fill(~candidate_mask, -1e4)
    ce_loss = sparse_cross_entropy(dense, targets, candidate_mask)
    conf_loss = conflict_penalty(dense, comm_edge_index, candidate_mask)
    regret_loss = cost_regret_penalty(dense, drone_pos, slots, targets, candidate_mask)
    for name, value in (("ce", ce_loss), ("conflict", conf_loss), ("regret", regret_loss)):
        if torch.isnan(value):
            if name == "ce":
                ce_loss = dense.sum() * 0.0
            elif name == "conflict":
                conf_loss = dense.sum() * 0.0
            else:
                regret_loss = dense.sum() * 0.0
    total = ce_loss + lambda_conflict * conf_loss + lambda_regret * regret_loss
    return total, ce_loss, conf_loss, regret_loss


def _neighbors_from_edge_index(edge_index: torch.Tensor, n: int) -> Dict[int, List[int]]:
    neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
    if edge_index.size(1) == 0:
        return neighbors
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if v not in neighbors[u]:
            neighbors[u].append(v)
    return neighbors


StrictClaim = Tuple[int, int, float, float]


def _claim_better(a: StrictClaim, b: StrictClaim) -> bool:
    """True when claim a beats claim b for the same slot."""
    _, drone_a, cost_a, bid_a = a
    _, drone_b, cost_b, bid_b = b
    if cost_a < cost_b - 1e-8:
        return True
    if cost_a > cost_b + 1e-8:
        return False
    if bid_a > bid_b + 1e-8:
        return True
    if bid_a < bid_b - 1e-8:
        return False
    return drone_a < drone_b


def _merge_claims(
    known: Dict[int, StrictClaim],
    incoming: Dict[int, StrictClaim],
) -> bool:
    changed = False
    for slot, claim in incoming.items():
        current = known.get(slot)
        if current is None or _claim_better(claim, current):
            known[slot] = claim
            changed = True
    return changed


def _best_unblocked_slot(
    dense: torch.Tensor,
    blocked: Set[int],
) -> Tuple[int, float]:
    row = dense.clone()
    for slot in blocked:
        row[slot] = -1e9
    if row.numel() == 0 or row.max().item() <= -1e8:
        return -1, -1e9
    slot = int(row.argmax().item())
    return slot, float(row[slot].item())


@torch.no_grad()
def assign_strict_decentralized_consensus(
    edge_logits: torch.Tensor,
    ds_edge_index: torch.Tensor,
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    comm_edge_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    max_rounds: int = MAX_CONSENSUS_ROUNDS,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Strict local consensus: drones only update from their own state and neighbor
    messages. No central solver, global used-slot set, or centralized final
    repair is used to create the returned assignment.
    """
    n = drone_pos.size(0)
    dense = edge_logits_to_dense(edge_logits, ds_edge_index, n, fill=-1e4)
    dense = dense.masked_fill(~candidate_mask, -1e4).cpu()
    dp = drone_pos.cpu()
    sl = slots.cpu()
    neighbors = _neighbors_from_edge_index(comm_edge_index.cpu(), n)

    blocked: List[Set[int]] = [set() for _ in range(n)]
    known: List[Dict[int, StrictClaim]] = [dict() for _ in range(n)]
    local_claims: List[int] = [-1] * n
    rounds_used = 0
    converged = False

    for round_idx in range(max_rounds):
        rounds_used = round_idx + 1
        changed = False

        for drone in range(n):
            slot, bid = _best_unblocked_slot(dense[drone], blocked[drone])
            local_claims[drone] = slot
            if slot < 0:
                continue
            cost = float(torch.norm(dp[drone] - sl[slot]).item())
            claim = (slot, drone, cost, bid)
            changed = _merge_claims(known[drone], {slot: claim}) or changed

        outgoing = [dict(k) for k in known]
        for drone in range(n):
            for neighbor in neighbors[drone]:
                changed = _merge_claims(known[neighbor], outgoing[drone]) or changed

        for drone in range(n):
            slot = local_claims[drone]
            if slot < 0:
                continue
            winner = known[drone].get(slot)
            if winner is not None and winner[1] != drone and slot not in blocked[drone]:
                blocked[drone].add(slot)
                changed = True

        if not changed:
            converged = True
            break

    assignment = torch.full((n,), -1, dtype=torch.long)
    for drone in range(n):
        slot, bid = _best_unblocked_slot(dense[drone], blocked[drone])
        if slot < 0:
            continue
        cost = float(torch.norm(dp[drone] - sl[slot]).item())
        own_claim = (slot, drone, cost, bid)
        _merge_claims(known[drone], {slot: own_claim})
        winner = known[drone].get(slot)
        if winner is None or winner[1] == drone:
            assignment[drone] = slot

    assigned = assignment[assignment >= 0]
    unique_assigned = assigned.unique().numel() if assigned.numel() > 0 else 0
    duplicate_count = int(assigned.numel() - unique_assigned)
    unassigned_count = int((assignment < 0).sum().item())
    info = {
        "rounds": float(rounds_used),
        "converged": float(converged),
        "assigned_rate": float(assigned.numel() / max(n, 1)),
        "unassigned_rate": float(unassigned_count / max(n, 1)),
        "conflict_rate": float(duplicate_count / max(n, 1)),
        "bijection": float(unassigned_count == 0 and duplicate_count == 0),
    }
    return assignment, info


def _flood_slot_winners(
    local_claim: List[int],
    local_bid: List[float],
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    neighbors: Dict[int, List[int]],
    n: int,
    flood_steps: int,
) -> List[int]:
    """Return known_winner[s] = drone with best (min cost, max bid) claim for slot s."""
    slot_cost = [float("inf")] * n
    slot_bid = [-1e9] * n
    slot_winner = [-1] * n

    for i in range(n):
        s = local_claim[i]
        if s < 0:
            continue
        cost = torch.norm(drone_pos[i] - slots[s]).item()
        bid = local_bid[i]
        if cost < slot_cost[s] or (math.isclose(cost, slot_cost[s]) and bid > slot_bid[s]):
            slot_cost[s] = cost
            slot_bid[s] = bid
            slot_winner[s] = i

    known_winner = slot_winner.copy()
    known_cost = slot_cost.copy()
    known_bid = slot_bid.copy()

    for _ in range(flood_steps):
        new_winner = known_winner.copy()
        new_cost = known_cost.copy()
        new_bid = known_bid.copy()
        for i in range(n):
            for j in neighbors[i]:
                s = local_claim[j]
                if s < 0 or known_winner[j] < 0:
                    continue
                if known_cost[j] < new_cost[s] or (
                    math.isclose(known_cost[j], new_cost[s]) and known_bid[j] > new_bid[s]
                ):
                    new_cost[s] = known_cost[j]
                    new_winner[s] = known_winner[j]
                    new_bid[s] = known_bid[j]
        known_winner, known_cost, known_bid = new_winner, new_cost, new_bid

    return known_winner


@torch.no_grad()
def assign_gossip_consensus(
    edge_logits: torch.Tensor,
    ds_edge_index: torch.Tensor,
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    comm_edge_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    max_rounds: int = MAX_GOSSIP_ROUNDS,
    flood_steps: int = GOSSIP_FLOOD_STEPS,
) -> Tuple[torch.Tensor, int]:
    """
    Multi-hop gossip assignment over sparse local bids.

    Each round: flood best claimants per slot across comm graph, then losers
    block contested slots and re-pick locally.
    """
    n = drone_pos.size(0)
    dense = edge_logits_to_dense(edge_logits, ds_edge_index, n, fill=-1e4)
    dense = dense.masked_fill(~candidate_mask, -1e4)

    neighbors = _neighbors_from_edge_index(comm_edge_index.cpu(), n)
    blocked = [set() for _ in range(n)]
    rounds_used = 0

    for round_idx in range(max_rounds):
        rounds_used = round_idx + 1
        local_claim = [-1] * n
        local_bid = [-1e9] * n
        for i in range(n):
            row = dense[i].clone()
            for b in blocked[i]:
                row[b] = -1e9
            if row.max() <= -1e8:
                continue
            j = int(row.argmax().item())
            local_claim[i] = j
            local_bid[i] = row[j].item()

        known_winner = _flood_slot_winners(
            local_claim, local_bid, drone_pos, slots, neighbors, n, flood_steps
        )

        new_blocked_any = False
        for i in range(n):
            s = local_claim[i]
            if s < 0:
                continue
            if known_winner[s] >= 0 and known_winner[s] != i:
                if s not in blocked[i]:
                    blocked[i].add(s)
                    new_blocked_any = True

        if not new_blocked_any:
            break

    # Final picks after blocking
    local_claim = [-1] * n
    local_bid = [-1e9] * n
    for i in range(n):
        row = dense[i].clone()
        for b in blocked[i]:
            row[b] = -1e9
        if row.max() > -1e8:
            j = int(row.argmax().item())
            local_claim[i] = j
            local_bid[i] = row[j].item()

    known_winner = _flood_slot_winners(
        local_claim, local_bid, drone_pos, slots, neighbors, n, flood_steps
    )

    assignment = torch.full((n,), -1, dtype=torch.long)
    for s in range(n):
        d = known_winner[s]
        if d >= 0 and assignment[d] < 0:
            assignment[d] = s

    used = set(assignment[assignment >= 0].tolist())
    for i in range(n):
        if assignment[i] >= 0:
            continue
        row = dense[i].clone()
        for s in used:
            row[s] = -1e9
        if row.max() > -1e8:
            j = int(row.argmax().item())
            assignment[i] = j
            used.add(j)

    for i in range(n):
        if assignment[i] < 0:
            for s in range(n):
                if s not in used:
                    assignment[i] = s
                    used.add(s)
                    break

    return assignment, rounds_used


@torch.no_grad()
def assign_with_decentralized_fallback(
    logits: torch.Tensor,
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    edge_index: torch.Tensor,
    max_iters: int = 30,
) -> torch.Tensor:
    """1-hop baseline (from v2)."""
    n = logits.size(0)
    adj_logits = logits.clone().float()
    if edge_index.size(1) == 0:
        return adj_logits.argmax(dim=1)
    neighbors = _neighbors_from_edge_index(edge_index, n)
    for _ in range(max_iters):
        assignment = adj_logits.argmax(dim=1)
        conflicts_detected = False
        new_adj_logits = adj_logits.clone()
        for i in range(n):
            slot_i = assignment[i].item()
            dist_i = torch.norm(drone_pos[i] - slots[slot_i]).item()
            for j in neighbors[i]:
                if assignment[j].item() == slot_i:
                    dist_j = torch.norm(drone_pos[j] - slots[slot_i]).item()
                    if dist_i > dist_j or (math.isclose(dist_i, dist_j) and i > j):
                        new_adj_logits[i, slot_i] = float("-inf")
                        conflicts_detected = True
                        break
        if not conflicts_detected:
            break
        adj_logits = new_adj_logits
    return adj_logits.argmax(dim=1)


def assignment_cost(
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    assignment: torch.Tensor,
) -> float:
    return sum(
        torch.norm(drone_pos[i] - slots[assignment[i]]).item() for i in range(assignment.size(0))
    )


def assignment_cost_with_unassigned_penalty(
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    assignment: torch.Tensor,
) -> float:
    valid = assignment >= 0
    if valid.any():
        assigned_cost = sum(
            torch.norm(drone_pos[i] - slots[assignment[i]]).item()
            for i in range(assignment.size(0))
            if assignment[i] >= 0
        )
    else:
        assigned_cost = 0.0
    arena_span = torch.cdist(drone_pos, slots).max().item() if drone_pos.numel() else 1.0
    penalty = max(arena_span, 1.0)
    return assigned_cost + float((~valid).sum().item()) * penalty


def assignment_quality_metrics(
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    assignment: torch.Tensor,
    target: torch.Tensor,
    rounds: float = 0.0,
) -> Dict[str, float]:
    n = assignment.numel()
    valid = assignment >= 0
    assigned = assignment[valid]
    duplicate_count = int(assigned.numel() - assigned.unique().numel()) if assigned.numel() else 0
    unassigned_count = int((~valid).sum().item())
    opt_cost = assignment_cost(drone_pos, slots, target)
    pred_cost = assignment_cost_with_unassigned_penalty(drone_pos, slots, assignment)
    return {
        "bijection_rate": float(unassigned_count == 0 and duplicate_count == 0),
        "conflict_rate": float(duplicate_count / max(n, 1)),
        "unassigned_rate": float(unassigned_count / max(n, 1)),
        "cost_ratio_vs_hungarian": float(pred_cost / (opt_cost + 1e-8)),
        "slot_match_rate": float((assignment == target).float().mean().item()),
        "consensus_rounds": float(rounds),
    }


def evaluate_local_model(
    model: LocalNegotiatorGNN,
    dataset: List[Data],
    device: torch.device,
    force_gt_visibility: bool = False,
) -> Dict[str, float]:
    model.eval()
    bijection_rates, cost_ratios, match_rates, gossip_rounds = [], [], [], []

    for data in dataset:
        fid = data.formation_id.item()
        f_emb = model.formation_embedding(torch.tensor(fid, device=device))
        dp, sl, y = data.drone_pos, data.slots, data.y
        mask = build_candidate_mask(
            dp,
            sl,
            SLOT_VISIBILITY_RADIUS,
            y=y,
            force_gt=force_gt_visibility,
        )
        x = build_node_features(dp, sl, f_emb.cpu(), candidate_mask=mask).to(device)
        ei, ea = build_comm_graph(dp)
        ds_ei = build_candidate_edges(mask)[0]
        ds_ea = build_candidate_edge_attr(dp, sl, ds_ei)

        edge_logits = model(
            x,
            ei.to(device),
            ea.to(device),
            ds_ei.to(device),
            ds_ea.to(device),
        )
        asgn, rounds = assign_gossip_consensus(
            edge_logits.cpu(),
            ds_ei,
            dp,
            sl,
            ei,
            mask,
        )
        n = dp.size(0)
        opt_cost = assignment_cost(dp, sl, y)
        pred_cost = assignment_cost(dp, sl, asgn)
        bijection_rates.append(float(len(asgn.unique()) == n))
        cost_ratios.append(pred_cost / (opt_cost + 1e-8))
        match_rates.append((asgn == y).float().mean().item())
        gossip_rounds.append(float(rounds))

    return {
        "bijection_rate": float(np.mean(bijection_rates)),
        "cost_ratio": float(np.mean(cost_ratios)),
        "match_rate": float(np.mean(match_rates)),
        "gossip_rounds": float(np.mean(gossip_rounds)),
    }


def evaluate_strict_decentralized(
    model: LocalNegotiatorGNN,
    dataset: List[Data],
    device: torch.device,
    slot_radius: float = SLOT_VISIBILITY_RADIUS,
    max_rounds: int = MAX_CONSENSUS_ROUNDS,
) -> Dict[str, float]:
    """Evaluate strict-local inference only; Hungarian is used as baseline labels."""
    model.eval()
    metrics: Dict[str, List[float]] = {
        "bijection_rate": [],
        "conflict_rate": [],
        "unassigned_rate": [],
        "cost_ratio_vs_hungarian": [],
        "slot_match_rate": [],
        "consensus_rounds": [],
        "converged_rate": [],
    }

    for data in dataset:
        fid = data.formation_id.item()
        f_emb = model.formation_embedding(torch.tensor(fid, device=device))
        dp, sl, y = data.drone_pos, data.slots, data.y
        mask = build_candidate_mask(dp, sl, slot_radius, y=y, force_gt=False)
        x = build_node_features(dp, sl, f_emb.cpu(), candidate_mask=mask).to(device)
        ei, ea = build_comm_graph(dp)
        ds_ei = build_candidate_edges(mask)[0]
        ds_ea = build_candidate_edge_attr(dp, sl, ds_ei)

        edge_logits = model(
            x,
            ei.to(device),
            ea.to(device),
            ds_ei.to(device),
            ds_ea.to(device),
        )
        asgn, info = assign_strict_decentralized_consensus(
            edge_logits.cpu(),
            ds_ei,
            dp,
            sl,
            ei,
            mask,
            max_rounds=max_rounds,
        )
        sample_metrics = assignment_quality_metrics(
            dp, sl, asgn, y, rounds=info["rounds"]
        )
        for key, value in sample_metrics.items():
            metrics[key].append(value)
        metrics["converged_rate"].append(info["converged"])

    return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}


def train_local_model(
    model: LocalNegotiatorGNN,
    train_data: List[Data],
    val_data: List[Data],
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_match": [],
    }
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for data in train_data:
            data = data  # already pyg
            fid = data.formation_id.item()
            f_emb = model.formation_embedding(torch.tensor(fid, device=device))
            x = build_node_features(
                data.drone_pos,
                data.slots,
                f_emb.cpu(),
                candidate_mask=data.candidate_mask,
            ).to(device)
            ei, ea = data.edge_index.to(device), data.edge_attr.to(device)
            ds_ei, ds_ea = data.ds_edge_index.to(device), data.ds_edge_attr.to(device)
            mask = data.candidate_mask.to(device)

            edge_logits = model(x, ei, ea, ds_ei, ds_ea)
            loss, _, _ = sparse_negotiator_loss(
                edge_logits, ds_ei, data.y.to(device), ei, mask
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for data in val_data:
                fid = data.formation_id.item()
                f_emb = model.formation_embedding(torch.tensor(fid, device=device))
                x = build_node_features(
                    data.drone_pos,
                    data.slots,
                    f_emb.cpu(),
                    candidate_mask=data.candidate_mask,
                ).to(device)
                ei, ea = data.edge_index.to(device), data.edge_attr.to(device)
                ds_ei, ds_ea = data.ds_edge_index.to(device), data.ds_edge_attr.to(device)
                mask = data.candidate_mask.to(device)
                edge_logits = model(x, ei, ea, ds_ei, ds_ea)
                loss, _, _ = sparse_negotiator_loss(
                    edge_logits, ds_ei, data.y.to(device), ei, mask
                )
                v_loss += loss.item()

        t_loss /= max(len(train_data), 1)
        v_loss /= max(len(val_data), 1)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            metrics = evaluate_local_model(model, val_data[:50], device)
            history.setdefault("val_match", []).append(metrics["match_rate"])
            print(
                f"Epoch {epoch:3d} | train_loss={t_loss:.4f} | val_loss={v_loss:.4f} | "
                f"match={metrics['match_rate']:.3f} | bij={metrics['bijection_rate']:.3f} | "
                f"cost_ratio={metrics['cost_ratio']:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_strict_decentralized_model(
    model: LocalNegotiatorGNN,
    train_data: List[Data],
    val_data: List[Data],
    device: torch.device,
    epochs: int = 150,
    lr: float = 1e-3,
    min_bijection_for_best: float = 0.40,
    force_gt_train: bool = True,
    ckpt_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Train LocalNegotiatorGNN with:
      - Cosine LR decay from `lr` to 1e-5 over `epochs` epochs.
      - Per-epoch training data shuffle.
      - Mid-training checkpoint: saved to `ckpt_path` every time cost ratio
        improves (regardless of bijection, subject to the min_bijection penalty).
      - force_gt_train=True: GT slot always in candidate mask during training
        so every drone receives a gradient signal. Val/eval always use
        force_gt=False (strict visibility).
      - Logs every 5 epochs: train loss, val loss, bijection, conflict,
        cost ratio, match rate, consensus rounds.

    Checkpoint policy
    -----------------
    score = cost_ratio_vs_hungarian
    If bijection_rate < min_bijection_for_best: score += 10 * gap
    Saved in-memory (best_state) and to disk (ckpt_path) whenever score improves.

    Parameters
    ----------
    ckpt_path : str or None
        If given, the best checkpoint is written here mid-training in the same
        format as strict_local_negotiator_best.pt so all downstream loaders
        work without changes. If None, only the in-memory best_state is kept.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine decay from lr → 1e-5 over all epochs.
    # eta_min is set to 1e-5 — low enough to not kill fine-tuning at the end,
    # high enough to keep gradients flowing.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    history: Dict[str, List[float]] = {
        "train_loss":                  [],
        "val_loss":                    [],
        "val_bijection_rate":          [],
        "val_cost_ratio_vs_hungarian": [],
        "val_conflict_rate":           [],
        "val_unassigned_rate":         [],
        "val_slot_match_rate":         [],
        "val_consensus_rounds":        [],
        "val_converged_rate":          [],
        "lr":                          [],
    }
    best_score = float("inf")
    best_state: Optional[Dict] = None

    for epoch in range(1, epochs + 1):

        # ── Shuffle training data each epoch ─────────────────────────────────
        shuffled = list(train_data)
        random.shuffle(shuffled)

        # ── Training pass ─────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for data in shuffled:
            fid = data.formation_id.item()
            f_emb = model.formation_embedding(torch.tensor(fid, device=device))
            strict_mask = build_candidate_mask(
                data.drone_pos,
                data.slots,
                SLOT_VISIBILITY_RADIUS,
                y=data.y,
                force_gt=force_gt_train,
            )
            x = build_node_features(
                data.drone_pos,
                data.slots,
                f_emb.cpu(),
                candidate_mask=strict_mask,
            ).to(device)
            ei, ea = data.edge_index.to(device), data.edge_attr.to(device)
            ds_ei = build_candidate_edges(strict_mask)[0].to(device)
            ds_ea = build_candidate_edge_attr(
                data.drone_pos, data.slots, ds_ei.cpu()
            ).to(device)
            mask = strict_mask.to(device)

            edge_logits = model(x, ei, ea, ds_ei, ds_ea)
            loss, _, _, _ = strict_assignment_loss(
                edge_logits,
                ds_ei,
                data.y.to(device),
                ei,
                mask,
                data.drone_pos.to(device),
                data.slots.to(device),
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        # ── Cosine LR step ────────────────────────────────────────────────────
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        history["lr"].append(current_lr)

        # ── Validation loss ───────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for data in val_data:
                fid = data.formation_id.item()
                f_emb = model.formation_embedding(torch.tensor(fid, device=device))
                strict_mask = build_candidate_mask(
                    data.drone_pos,
                    data.slots,
                    SLOT_VISIBILITY_RADIUS,
                    y=data.y,
                    force_gt=False,  # always strict at val — mirrors real inference
                )
                x = build_node_features(
                    data.drone_pos,
                    data.slots,
                    f_emb.cpu(),
                    candidate_mask=strict_mask,
                ).to(device)
                ei, ea = data.edge_index.to(device), data.edge_attr.to(device)
                ds_ei = build_candidate_edges(strict_mask)[0].to(device)
                ds_ea = build_candidate_edge_attr(
                    data.drone_pos, data.slots, ds_ei.cpu()
                ).to(device)
                mask = strict_mask.to(device)
                edge_logits = model(x, ei, ea, ds_ei, ds_ea)
                loss, _, _, _ = strict_assignment_loss(
                    edge_logits,
                    ds_ei,
                    data.y.to(device),
                    ei,
                    mask,
                    data.drone_pos.to(device),
                    data.slots.to(device),
                )
                v_loss += loss.item()

        t_loss /= max(len(train_data), 1)
        v_loss /= max(len(val_data), 1)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        # ── Val inference metrics ─────────────────────────────────────────────
        metrics = evaluate_strict_decentralized(model, val_data[:100], device)
        for key in (
            "bijection_rate",
            "cost_ratio_vs_hungarian",
            "conflict_rate",
            "unassigned_rate",
            "slot_match_rate",
            "consensus_rounds",
            "converged_rate",
        ):
            history[f"val_{key}"].append(metrics[key])

        # ── Checkpoint: save whenever score improves ──────────────────────────
        score = metrics["cost_ratio_vs_hungarian"]
        if metrics["bijection_rate"] < min_bijection_for_best:
            score += 10.0 * (min_bijection_for_best - metrics["bijection_rate"])
        if score < best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch":            epoch,
                        "best_score":       best_score,
                        "config": {
                            "num_formations":         NUM_FORMATIONS,
                            "comm_radius":            COMM_RADIUS,
                            "slot_visibility_radius": SLOT_VISIBILITY_RADIUS,
                            "force_gt_visibility_train": FORCE_GT_VISIBILITY_TRAIN,
                        },
                        "history": history,
                        # 'metrics' key filled with partial data here;
                        # overwritten with full train/val/test after the loop.
                        "metrics": {"val": metrics},
                    },
                    ckpt_path,
                )

        # ── Logging ───────────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | lr={current_lr:.2e} | "
                f"train={t_loss:.4f} | val={v_loss:.4f} | "
                f"bij={metrics['bijection_rate']:.3f} | "
                f"conflict={metrics['conflict_rate']:.3f} | "
                f"cost_ratio={metrics['cost_ratio_vs_hungarian']:.4f} | "
                f"match={metrics['slot_match_rate']:.3f} | "
                f"rounds={metrics['consensus_rounds']:.1f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
