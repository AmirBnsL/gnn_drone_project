"""
decent_auction.py
=================
Fully decentralized learned auction for drone-to-slot assignment.

This module is intentionally separate from bertsekas_auction_v2.py.  The main
difference is the inference protocol:

  * there is no single global price vector;
  * each drone maintains its own local view of prices and slot owners;
  * drones communicate only with graph neighbours;
  * each drone decides its final assignment from its own local state.

Hungarian is used only as an offline teacher/baseline through the dataset
labels.  It is never called in the inference path.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from local_negotiator_v2 import (
    COMM_RADIUS,
    HIDDEN_DIM,
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


DECENT_AUCTION_ROUNDS = 20
DECENT_RE_ENCODE_EVERY = 5
DECENT_TRAIN_K = 4
PRICE_FEAT_DIM = 3
AUGMENTED_IN_DIM = NODE_FEAT_DIM + PRICE_FEAT_DIM
EPS_INIT = 0.05
EPS_MIN = 1e-3
EPS_MAX = 1.0
CE_WEIGHT = 1.0
REGRET_WEIGHT = 0.10
COVERAGE_WEIGHT = 0.20
MIN_BIJ_FOR_BEST = 0.70
FREEZE_EPOCHS = 10
UNFREEZE_BIJECTION_GUARD = 0.45


def _current_train_k(
    joint_epoch: int,
    train_k: int = DECENT_TRAIN_K,
    anneal_epochs: int = 20,
) -> int:
    """Ramp recurrent auction-state exposure from 0 to train_k."""
    if train_k <= 0:
        return 0
    if joint_epoch <= 0:
        return 0
    t = min(1.0, joint_epoch / max(anneal_epochs, 1))
    return int(round(t * train_k))


def _neighbors_from_edge_index(edge_index: torch.Tensor, n: int) -> Dict[int, List[int]]:
    neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
    if edge_index.numel() == 0:
        return neighbors
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if u != v and v not in neighbors[u]:
            neighbors[u].append(v)
    return neighbors


class DecentAuctionGNN(nn.Module):
    """
    Price-aware wrapper around LocalNegotiatorGNN.

    The encoder receives three local price features per drone:
      1. price of the current claimed slot;
      2. mean price of visible slots in that drone's local view;
      3. fraction of visible slots that already have a known owner.
    """

    def __init__(self, base_model: LocalNegotiatorGNN):
        super().__init__()
        self.formation_embedding = base_model.formation_embedding
        self.gat_layers = base_model.gat_layers
        self.layer_norms = base_model.layer_norms
        self.edge_scorer = base_model.edge_scorer
        self.log_temp = base_model.log_temp

        self.price_proj = nn.Linear(AUGMENTED_IN_DIM, HIDDEN_DIM)
        with torch.no_grad():
            self.price_proj.weight[:, :NODE_FEAT_DIM].copy_(base_model.input_proj.weight)
            self.price_proj.weight[:, NODE_FEAT_DIM:].zero_()
            self.price_proj.bias.copy_(base_model.input_proj.bias)

    def _encode(
        self,
        x_aug: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = F.relu(self.price_proj(x_aug))
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            if edge_index.size(1) > 0:
                h_new = gat(h, edge_index, edge_attr=edge_attr)
            else:
                h_new = torch.zeros_like(h)
            h = ln(h + F.elu(h_new))
        return h

    def _score_edges(
        self,
        h: torch.Tensor,
        ds_edge_index: torch.Tensor,
        ds_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if ds_edge_index.size(1) == 0:
            return torch.zeros(0, device=h.device)
        i = ds_edge_index[0]
        temp = torch.exp(self.log_temp).clamp(min=1e-3)
        edge_in = torch.cat([h[i], ds_edge_attr], dim=-1)
        return self.edge_scorer(edge_in).squeeze(-1) / temp

    @staticmethod
    def _price_features(
        local_prices: torch.Tensor,
        local_owner: torch.Tensor,
        current_claims: torch.Tensor,
        candidate_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        n = local_prices.size(0)
        prices = local_prices.to(device)
        owner = local_owner.to(device)
        claims = current_claims.to(device)
        mask = candidate_mask.to(device)
        mask_f = mask.float()

        claimed_price = torch.zeros(n, device=device)
        valid_claim = claims >= 0
        if valid_claim.any():
            rows = torch.arange(n, device=device)[valid_claim]
            claimed_price[valid_claim] = prices[rows, claims[valid_claim]]

        counts = mask_f.sum(dim=1).clamp(min=1.0)
        mean_visible_price = (prices * mask_f).sum(dim=1) / counts
        owned_visible = ((owner >= 0) & mask).float().sum(dim=1) / counts
        return torch.stack([claimed_price, mean_visible_price, owned_visible], dim=1)

    def forward_with_local_state(
        self,
        x_base: torch.Tensor,
        local_prices: torch.Tensor,
        local_owner: torch.Tensor,
        current_claims: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        ds_edge_index: torch.Tensor,
        ds_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        dev = x_base.device
        n = x_base.size(0)
        pf = self._price_features(
            local_prices, local_owner, current_claims, candidate_mask, dev
        )
        x_aug = torch.cat([x_base, pf], dim=1)
        h = self._encode(x_aug, edge_index, edge_attr)
        logits = self._score_edges(h, ds_edge_index, ds_edge_attr)
        dense = edge_logits_to_dense(logits, ds_edge_index, n, fill=-1e9)
        return dense.masked_fill(~candidate_mask.to(dev), -1e9)


def _local_eps_from_values(
    values: torch.Tensor,
    prices: torch.Tensor,
    visible: torch.Tensor,
) -> Tuple[int, float, float]:
    net = (values - prices).masked_fill(~visible, -1e9)
    if not bool(visible.any()):
        return -1, 0.0, EPS_INIT

    k = min(2, int(visible.sum().item()))
    top = torch.topk(net, k=k).values
    best_slot = int(net.argmax().item())
    best = float(top[0].item())
    second = float(top[1].item()) if k > 1 else best - EPS_INIT
    eps = float(max(EPS_MIN, min(EPS_MAX, 0.5 * max(best - second, EPS_INIT))))
    bid_price = float(prices[best_slot].item() + max(best - second, 0.0) + eps)
    return best_slot, bid_price, eps


@torch.no_grad()
def decent_auction(
    values: torch.Tensor,
    candidate_mask: torch.Tensor,
    comm_edge_index: torch.Tensor,
    max_rounds: int = DECENT_AUCTION_ROUNDS,
    initial_prices: Optional[torch.Tensor] = None,
    initial_owner: Optional[torch.Tensor] = None,
    initial_claims: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fully decentralized auction over local drone states.

    Each drone i stores local_prices[i], local_owner[i], and current_claims[i].
    A round contains local bidding followed by one-hop neighbour gossip.  Merge
    rules are commutative: higher known price wins; equal prices use lower
    owner id as a deterministic tie break.
    """
    v = values.cpu().clone()
    mask = candidate_mask.cpu()
    n = v.size(0)
    neighbors = _neighbors_from_edge_index(comm_edge_index.cpu(), n)

    local_prices = (
        initial_prices.cpu().clone()
        if initial_prices is not None
        else torch.zeros(n, n)
    )
    local_owner = (
        initial_owner.cpu().clone()
        if initial_owner is not None
        else torch.full((n, n), -1, dtype=torch.long)
    )
    current_claims = (
        initial_claims.cpu().clone()
        if initial_claims is not None
        else torch.full((n,), -1, dtype=torch.long)
    )

    messages_sent = 0
    rounds_used = 0
    converged = False

    for round_idx in range(max_rounds):
        rounds_used = round_idx + 1
        prev_prices = local_prices.clone()
        prev_owner = local_owner.clone()
        prev_claims = current_claims.clone()

        for drone in range(n):
            claim = int(current_claims[drone].item())
            still_owner = claim >= 0 and int(local_owner[drone, claim].item()) == drone
            if still_owner:
                continue

            slot, bid_price, _ = _local_eps_from_values(
                v[drone], local_prices[drone], mask[drone]
            )
            if slot < 0:
                current_claims[drone] = -1
                continue

            current_claims[drone] = slot
            if (
                bid_price > float(local_prices[drone, slot].item()) + 1e-9
                or int(local_owner[drone, slot].item()) < 0
                or drone < int(local_owner[drone, slot].item())
            ):
                local_prices[drone, slot] = bid_price
                local_owner[drone, slot] = drone

        price_snapshot = local_prices.clone()
        owner_snapshot = local_owner.clone()

        for sender in range(n):
            for receiver in neighbors[sender]:
                messages_sent += 1
                for slot in range(n):
                    sender_price = price_snapshot[sender, slot]
                    receiver_price = local_prices[receiver, slot]
                    sender_owner = int(owner_snapshot[sender, slot].item())
                    receiver_owner = int(local_owner[receiver, slot].item())

                    better_price = sender_price > receiver_price + 1e-9
                    same_price = abs(float(sender_price - receiver_price)) <= 1e-9
                    better_tie = (
                        same_price
                        and sender_owner >= 0
                        and (receiver_owner < 0 or sender_owner < receiver_owner)
                    )
                    if bool(better_price) or better_tie:
                        local_prices[receiver, slot] = sender_price
                        local_owner[receiver, slot] = sender_owner

        for drone in range(n):
            claim = int(current_claims[drone].item())
            if claim >= 0 and int(local_owner[drone, claim].item()) != drone:
                current_claims[drone] = -1

        state_changed = (
            (local_prices - prev_prices).abs().max().item() > 1e-6
            or not torch.equal(local_owner, prev_owner)
            or not torch.equal(current_claims, prev_claims)
        )
        if not state_changed:
            converged = True
            break

    final = torch.full((n,), -1, dtype=torch.long)
    for drone in range(n):
        claim = int(current_claims[drone].item())
        if claim >= 0 and int(local_owner[drone, claim].item()) == drone:
            final[drone] = claim

    assigned = final[final >= 0]
    dup = int(assigned.numel() - assigned.unique().numel()) if assigned.numel() else 0
    unass = int((final < 0).sum().item())
    info = {
        "rounds": float(rounds_used),
        "converged": float(converged),
        "messages_sent": float(messages_sent),
        "conflict_rate": float(dup / max(n, 1)),
        "unassigned_rate": float(unass / max(n, 1)),
        "bijection": float(dup == 0 and unass == 0),
    }
    return final, info, local_prices, local_owner, current_claims


class DecentAuctionModel(nn.Module):
    def __init__(self, base_model: Optional[LocalNegotiatorGNN] = None):
        super().__init__()
        if base_model is None:
            base_model = LocalNegotiatorGNN()
        self.gnn = DecentAuctionGNN(base_model)

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
        train_k: int = 0,
    ) -> torch.Tensor:
        n = x_base.size(0)
        prices = torch.zeros(n, n)
        owner = torch.full((n, n), -1, dtype=torch.long)
        claims = torch.full((n,), -1, dtype=torch.long)

        with torch.no_grad():
            for _ in range(max(int(train_k), 0)):
                warm_values = self.gnn.forward_with_local_state(
                    x_base,
                    prices.to(x_base.device),
                    owner.to(x_base.device),
                    claims.to(x_base.device),
                    candidate_mask,
                    edge_index,
                    edge_attr,
                    ds_edge_index,
                    ds_edge_attr,
                )
                _, _, prices, owner, claims = decent_auction(
                    warm_values.detach().cpu(),
                    candidate_mask.detach().cpu(),
                    edge_index.detach().cpu(),
                    max_rounds=1,
                    initial_prices=prices,
                    initial_owner=owner,
                    initial_claims=claims,
                )

        return self.gnn.forward_with_local_state(
            x_base,
            prices.to(x_base.device),
            owner.to(x_base.device),
            claims.to(x_base.device),
            candidate_mask,
            edge_index,
            edge_attr,
            ds_edge_index,
            ds_edge_attr,
        )

    @torch.no_grad()
    def assign(
        self,
        data: Data,
        device: torch.device,
        slot_radius: float = SLOT_VISIBILITY_RADIUS,
        max_rounds: int = DECENT_AUCTION_ROUNDS,
        re_encode_every: int = DECENT_RE_ENCODE_EVERY,
    ) -> Tuple[torch.Tensor, Dict]:
        self.eval()
        fid = data.formation_id.item()
        f_emb = self.gnn.formation_embedding(torch.tensor(fid, device=device))

        mask = build_candidate_mask(
            data.drone_pos, data.slots, slot_radius, y=data.y, force_gt=False
        )
        x = build_node_features(
            data.drone_pos, data.slots, f_emb.detach().cpu(), candidate_mask=mask
        ).to(device)
        edge_index, edge_attr = build_comm_graph(data.drone_pos, COMM_RADIUS)
        ds_edge_index = build_candidate_edges(mask)[0]
        ds_edge_attr = build_candidate_edge_attr(data.drone_pos, data.slots, ds_edge_index)

        n = x.size(0)
        local_prices = torch.zeros(n, n)
        local_owner = torch.full((n, n), -1, dtype=torch.long)
        claims = torch.full((n,), -1, dtype=torch.long)

        total_messages = 0.0
        total_rounds = 0.0
        converged = 0.0
        final = torch.full((n,), -1, dtype=torch.long)

        remaining = max_rounds
        while remaining > 0:
            values = self.gnn.forward_with_local_state(
                x,
                local_prices.to(device),
                local_owner.to(device),
                claims.to(device),
                mask.to(device),
                edge_index.to(device),
                edge_attr.to(device),
                ds_edge_index.to(device),
                ds_edge_attr.to(device),
            ).cpu()

            chunk = min(re_encode_every, remaining)
            final, info, local_prices, local_owner, claims = decent_auction(
                values,
                mask,
                edge_index,
                max_rounds=chunk,
                initial_prices=local_prices,
                initial_owner=local_owner,
                initial_claims=claims,
            )
            total_messages += info["messages_sent"]
            total_rounds += info["rounds"]
            remaining -= chunk
            if info["converged"]:
                converged = 1.0
                break

        assigned = final[final >= 0]
        dup = int(assigned.numel() - assigned.unique().numel()) if assigned.numel() else 0
        unass = int((final < 0).sum().item())
        return final, {
            "rounds": float(total_rounds),
            "converged": float(converged),
            "messages_sent": float(total_messages),
            "conflict_rate": float(dup / max(n, 1)),
            "unassigned_rate": float(unass / max(n, 1)),
            "bijection": float(dup == 0 and unass == 0),
        }


def decent_auction_loss(
    values: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    drone_pos: torch.Tensor,
    slots: torch.Tensor,
    ce_weight: float = CE_WEIGHT,
    regret_weight: float = REGRET_WEIGHT,
    coverage_weight: float = COVERAGE_WEIGHT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = sparse_cross_entropy(values, targets.to(values.device), candidate_mask.to(values.device))
    regret = cost_regret_penalty(
        values,
        drone_pos.to(values.device),
        slots.to(values.device),
        targets.to(values.device),
        candidate_mask.to(values.device),
    )
    mask = candidate_mask.to(values.device).bool()
    visible_rows = mask.any(dim=1)
    if bool(visible_rows.any()):
        v_max = values.masked_fill(~mask, -1e9).max(dim=1).values
        coverage = F.relu(-v_max).masked_select(visible_rows).mean()
    else:
        coverage = values.sum() * 0.0
    total = ce_weight * ce + regret_weight * regret + coverage_weight * coverage
    return total, ce, regret, coverage


@torch.no_grad()
def evaluate_decent_auction(
    model: DecentAuctionModel,
    dataset: List[Data],
    device: torch.device,
    slot_radius: float = SLOT_VISIBILITY_RADIUS,
    max_rounds: int = DECENT_AUCTION_ROUNDS,
    re_encode_every: int = DECENT_RE_ENCODE_EVERY,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, List[float]] = {
        "bijection_rate": [],
        "conflict_rate": [],
        "unassigned_rate": [],
        "cost_ratio_vs_hungarian": [],
        "slot_match_rate": [],
        "consensus_rounds": [],
        "converged_rate": [],
        "messages_sent": [],
    }
    for data in dataset:
        assignment, info = model.assign(
            data,
            device,
            slot_radius=slot_radius,
            max_rounds=max_rounds,
            re_encode_every=re_encode_every,
        )
        sample = assignment_quality_metrics(
            data.drone_pos, data.slots, assignment, data.y, rounds=info["rounds"]
        )
        for key, value in sample.items():
            metrics[key].append(value)
        metrics["converged_rate"].append(info["converged"])
        metrics["messages_sent"].append(info["messages_sent"])
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def _new_params(model: DecentAuctionModel) -> List[nn.Parameter]:
    return list(model.gnn.price_proj.parameters())


def _base_params(model: DecentAuctionModel) -> List[nn.Parameter]:
    return (
        list(model.gnn.formation_embedding.parameters())
        + list(model.gnn.gat_layers.parameters())
        + list(model.gnn.layer_norms.parameters())
        + list(model.gnn.edge_scorer.parameters())
        + [model.gnn.log_temp]
    )


def train_decent_auction_model(
    model: DecentAuctionModel,
    train_data: List[Data],
    val_data: List[Data],
    device: torch.device,
    epochs: int = 80,
    lr: float = 3e-4,
    freeze_epochs: int = FREEZE_EPOCHS,
    min_bijection_for_best: float = MIN_BIJ_FOR_BEST,
    force_gt_train: bool = True,
    eval_subset: int = 100,
    train_k: int = DECENT_TRAIN_K,
    unfreeze_bijection_guard: float = UNFREEZE_BIJECTION_GUARD,
    ckpt_path: Optional[str] = "decent_auction_best.pt",
) -> Dict[str, List[float]]:
    new_params = _new_params(model)
    base_params = _base_params(model)
    for p in base_params:
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(new_params, lr=lr, weight_decay=1e-4)
    # Frozen-phase scheduler: spans the whole run so LR decays even if the
    # bijection guard is never met and unfreezing never occurs.
    frozen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-5
    )
    joint_scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None
    frozen = True
    joint_epoch = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_bijection_rate": [],
        "val_cost_ratio_vs_hungarian": [],
        "val_slot_match_rate": [],
        "val_conflict_rate": [],
        "val_unassigned_rate": [],
        "val_converged_rate": [],
        "val_messages_sent": [],
        "train_k": [],
        "lr": [],
    }
    best_score = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        current_k = 0 if frozen else _current_train_k(joint_epoch, train_k=train_k)
        history["train_k"].append(float(current_k))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        model.train()
        random.shuffle(train_data)
        losses: List[float] = []

        for data in train_data:
            fid = data.formation_id.item()
            f_emb = model.gnn.formation_embedding(torch.tensor(fid, device=device))
            mask = build_candidate_mask(
                data.drone_pos,
                data.slots,
                SLOT_VISIBILITY_RADIUS,
                y=data.y,
                force_gt=force_gt_train,
            )
            x = build_node_features(
                data.drone_pos, data.slots, f_emb.detach().cpu(), candidate_mask=mask
            ).to(device)
            edge_index, edge_attr = build_comm_graph(data.drone_pos, COMM_RADIUS)
            ds_edge_index = build_candidate_edges(mask)[0]
            ds_edge_attr = build_candidate_edge_attr(
                data.drone_pos, data.slots, ds_edge_index
            )

            values = model.forward_values(
                x,
                mask.to(device),
                edge_index.to(device),
                edge_attr.to(device),
                ds_edge_index.to(device),
                ds_edge_attr.to(device),
                train_k=current_k,
            )
            loss, _, _, _ = decent_auction_loss(
                values, data.y, mask, data.drone_pos, data.slots
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(float(loss.item()))

        history["train_loss"].append(float(np.mean(losses)) if losses else 0.0)

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for data in val_data[:eval_subset]:
                fid = data.formation_id.item()
                f_emb = model.gnn.formation_embedding(torch.tensor(fid, device=device))
                mask = build_candidate_mask(
                    data.drone_pos,
                    data.slots,
                    SLOT_VISIBILITY_RADIUS,
                    y=data.y,
                    force_gt=False,
                )
                x = build_node_features(
                    data.drone_pos, data.slots, f_emb.detach().cpu(), candidate_mask=mask
                ).to(device)
                edge_index, edge_attr = build_comm_graph(data.drone_pos, COMM_RADIUS)
                ds_edge_index = build_candidate_edges(mask)[0]
                ds_edge_attr = build_candidate_edge_attr(
                    data.drone_pos, data.slots, ds_edge_index
                )
                values = model.forward_values(
                    x,
                    mask.to(device),
                    edge_index.to(device),
                    edge_attr.to(device),
                    ds_edge_index.to(device),
                    ds_edge_attr.to(device),
                    train_k=current_k,
                )
                val_loss, _, _, _ = decent_auction_loss(
                    values, data.y, mask, data.drone_pos, data.slots
                )
                val_losses.append(float(val_loss.item()))

        val_metrics = evaluate_decent_auction(
            model, val_data[:eval_subset], device, max_rounds=DECENT_AUCTION_ROUNDS
        )
        history["val_loss"].append(float(np.mean(val_losses)) if val_losses else 0.0)
        for key in (
            "bijection_rate",
            "cost_ratio_vs_hungarian",
            "slot_match_rate",
            "conflict_rate",
            "unassigned_rate",
            "converged_rate",
            "messages_sent",
        ):
            history[f"val_{key}"].append(val_metrics[key])

        score = val_metrics["cost_ratio_vs_hungarian"]
        if val_metrics["bijection_rate"] < min_bijection_for_best:
            score += 10.0 * (min_bijection_for_best - val_metrics["bijection_rate"])
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path:
                torch.save(
                    {"model_state_dict": best_state, "metrics": val_metrics, "epoch": epoch},
                    ckpt_path,
                )

        if (
            frozen
            and epoch >= freeze_epochs
            and val_metrics["bijection_rate"] >= unfreeze_bijection_guard
        ):
            for p in base_params:
                p.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": new_params, "lr": lr},
                    {"params": base_params, "lr": lr * 0.05},
                ],
                weight_decay=1e-4,
            )
            joint_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(epochs - epoch, 1),
                eta_min=1e-5,
            )
            frozen = False
            joint_epoch = 1
            print(
                f"  [Epoch {epoch}] Encoder unfrozen "
                f"(bijection={val_metrics['bijection_rate']:.3f}, guard={unfreeze_bijection_guard:.3f})."
            )
        elif not frozen:
            joint_epoch += 1

        if frozen:
            frozen_scheduler.step()
        elif joint_scheduler is not None:
            joint_scheduler.step()

        # Early stopping: stop when val_loss has not improved for 15 consecutive epochs.
        patience = 15
        if len(history["val_loss"]) > patience:
            recent_best = min(history["val_loss"][:-patience])
            if all(v >= recent_best for v in history["val_loss"][-patience:]):
                print(f"  Early stopping at epoch {epoch} — val loss has not improved for {patience} epochs.")
                break

        phase = "frozen" if frozen else "joint"
        print(
            f"epoch {epoch:03d} "
            f"[{phase}|K={current_k}] "
            f"loss={history['train_loss'][-1]:.4f} "
            f"val_cost={val_metrics['cost_ratio_vs_hungarian']:.4f} "
            f"val_bij={val_metrics['bijection_rate']:.3f} "
            f"val_match={val_metrics['slot_match_rate']:.3f} "
            f"val_conv={val_metrics['converged_rate']:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history