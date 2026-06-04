"""
super_decent_auction.py
=======================
SuperGlueSwarmMatcher + fully local-state DecentAuction.

Use this when you want the stronger cross-attentional value initializer from
superglue_negotiator.py, but still want decentralized conflict resolution from
decent_auction.py.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from decent_auction import DECENT_AUCTION_ROUNDS, decent_auction
from local_negotiator_v2 import (
    COMM_RADIUS,
    SLOT_VISIBILITY_RADIUS,
    assignment_quality_metrics,
    build_candidate_mask,
    build_comm_graph,
    load_negotiator_dataset,
    prepare_dataset,
)
from superglue_negotiator_new import (
    SuperGlueSwarmMatcher,
    evaluate_superglue_matcher,
    superglue_loss,
)


MIN_BIJ_FOR_BEST = 0.75
FREEZE_EPOCHS = 10          # was 5 — give the head more time before touching backbone
UNFREEZE_BIJECTION_GUARD = 0.75   # was 0.5 — only unfreeze when auction is actually working


class SuperDecentAuctionModel(nn.Module):
    """
    Thin wrapper:
      SuperGlueSwarmMatcher -> V, eps
      decent_auction        -> hard decentralized assignment

    The current decent_auction protocol learns from V.  The eps output is kept
    and trained by superglue_loss so the same checkpoint can also initialize
    Bertsekas-style variants that consume eps explicitly.
    """

    def __init__(self, matcher: Optional[SuperGlueSwarmMatcher] = None):
        super().__init__()
        self.matcher = matcher if matcher is not None else SuperGlueSwarmMatcher()

    @property
    def formation_embedding(self):
        return self.matcher.formation_embedding

    def forward_values(
        self,
        data: Data,
        device: torch.device,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_mask is None:
            candidate_mask = build_candidate_mask(
                data.drone_pos,
                data.slots,
                SLOT_VISIBILITY_RADIUS,
                y=data.y,
                force_gt=False,
            )
        edge_index, _ = build_comm_graph(data.drone_pos, COMM_RADIUS)
        values, eps = self.matcher(
            data.drone_pos.to(device).float(),
            data.slots.to(device).float(),
            data.formation_id.to(device),
            candidate_mask.to(device),
            edge_index.to(device),
        )
        return values, eps, candidate_mask

    @torch.no_grad()
    def assign(
        self,
        data: Data,
        device: torch.device,
        slot_radius: float = SLOT_VISIBILITY_RADIUS,
        max_rounds: int = DECENT_AUCTION_ROUNDS,
    ) -> Tuple[torch.Tensor, Dict]:
        self.eval()
        mask = build_candidate_mask(
            data.drone_pos, data.slots, slot_radius, y=data.y, force_gt=False
        )
        edge_index, _ = build_comm_graph(data.drone_pos, COMM_RADIUS)

        n = data.drone_pos.size(0)
        local_prices = torch.zeros(n, n)
        local_owner = torch.full((n, n), -1, dtype=torch.long)
        claims = torch.full((n,), -1, dtype=torch.long)

        values, _ = self.matcher(
            data.drone_pos.to(device).float(),
            data.slots.to(device).float(),
            data.formation_id.to(device),
            mask.to(device),
            edge_index.to(device),
            local_prices=local_prices.to(device),
            local_owner=local_owner.to(device),
            claims=claims.to(device),
        )
        final, info, _, _, _ = decent_auction(
            values.cpu(),
            mask,
            edge_index,
            max_rounds=max_rounds,
        )

        assigned = final[final >= 0]
        dup = int(assigned.numel() - assigned.unique().numel()) if assigned.numel() else 0
        unass = int((final < 0).sum().item())
        return final, {
            "rounds": float(info["rounds"]),
            "converged": float(info["converged"]),
            "messages_sent": float(info["messages_sent"]),
            "conflict_rate": float(dup / max(n, 1)),
            "unassigned_rate": float(unass / max(n, 1)),
            "bijection": float(dup == 0 and unass == 0),
        }


@torch.no_grad()
def evaluate_super_decent_auction(
    model: SuperDecentAuctionModel,
    dataset: List[Data],
    device: torch.device,
    slot_radius: float = SLOT_VISIBILITY_RADIUS,
    max_rounds: int = DECENT_AUCTION_ROUNDS,
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
        )
        sample = assignment_quality_metrics(
            data.drone_pos, data.slots, assignment, data.y, rounds=info["rounds"]
        )
        for key, value in sample.items():
            metrics[key].append(value)
        metrics["converged_rate"].append(info["converged"])
        metrics["messages_sent"].append(info["messages_sent"])
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def train_super_decent_auction(
    model: SuperDecentAuctionModel,
    train_data: List[Data],
    val_data: List[Data],
    device: torch.device,
    epochs: int = 80,
    lr: float = 3e-4,
    freeze_epochs: int = FREEZE_EPOCHS,
    min_bijection_for_best: float = MIN_BIJ_FOR_BEST,
    unfreeze_bijection_guard: float = UNFREEZE_BIJECTION_GUARD,
    force_gt_train: bool = True,
    eval_subset: int = 100,
    ckpt_path: Optional[str] = "super_decent_auction_best.pt",
    patience: int = 15,
) -> Dict[str, List[float]]:
    """
    Fine-tunes the SuperGlue matcher with the same supervised value/eps loss,
    while validating through the actual decent_auction inference path.
    """
    model.to(device)

    backbone_params = [
        p for name, p in model.named_parameters()
        if name.startswith("matcher.layers.")
    ]
    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("matcher.layers.")
    ]
    for p in backbone_params:
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
    [
        {"params": head_params,     "lr": lr},
        {"params": backbone_params, "lr": lr * 0.05},  # ← 5× smaller
    ],
    weight_decay=1e-4,)
    # Frozen-phase scheduler: spans the whole run so LR always decays,
    # even if the bijection guard is never met.
    frozen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(freeze_epochs, 1), eta_min=1e-5
    )
    joint_scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None
    frozen = True
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
        "lr": [],
        "phase": [],
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
            _n = data.drone_pos.size(0)
            _lp = torch.zeros(_n, _n, device=device)
            _lo = torch.full((_n, _n), -1, dtype=torch.long, device=device)
            _cl = torch.full((_n,), -1, dtype=torch.long, device=device)
            values, eps = model.matcher(
                data.drone_pos.to(device).float(),
                data.slots.to(device).float(),
                data.formation_id.to(device),
                mask.to(device),
                build_comm_graph(data.drone_pos, COMM_RADIUS)[0].to(device),
                local_prices=_lp,
                local_owner=_lo,
                claims=_cl,
            )
            loss, _ = superglue_loss(values, eps, data, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(float(loss.item()))

        if frozen:
            frozen_scheduler.step()
        elif joint_scheduler is not None:
            joint_scheduler.step()
        history["train_loss"].append(float(np.mean(losses)) if losses else 0.0)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["phase"].append(0.0 if frozen else 1.0)

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
                _n = data.drone_pos.size(0)
                _lp = torch.zeros(_n, _n, device=device)
                _lo = torch.full((_n, _n), -1, dtype=torch.long, device=device)
                _cl = torch.full((_n,), -1, dtype=torch.long, device=device)
                values, eps = model.matcher(
                    data.drone_pos.to(device).float(),
                    data.slots.to(device).float(),
                    data.formation_id.to(device),
                    mask.to(device),
                    build_comm_graph(data.drone_pos, COMM_RADIUS)[0].to(device),
                    local_prices=_lp,
                    local_owner=_lo,
                    claims=_cl,
                )
                val_loss, _ = superglue_loss(values, eps, data, mask)
                val_losses.append(float(val_loss.item()))

        val_metrics = evaluate_super_decent_auction(
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
            score += 15.0 * (min_bijection_for_best - val_metrics["bijection_rate"])
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

        if (
            frozen
            and epoch > freeze_epochs
            and val_metrics["bijection_rate"] >= unfreeze_bijection_guard
        ):
            for p in backbone_params:
                p.requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": head_params, "lr": lr},
                    {"params": backbone_params, "lr": lr * 0.25},
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
                f"  [Epoch {epoch}] SuperGlue backbone unfrozen "
                f"(bijection={val_metrics['bijection_rate']:.3f}, guard={unfreeze_bijection_guard:.3f})."
            )

        # Early stopping: stop when val_loss has not improved for `patience` epochs.
        # All history arrays are appended before this check so they stay in sync on break.
        if len(history["val_loss"]) > patience:
            recent_best = min(history["val_loss"][:-patience])
            if all(v >= recent_best for v in history["val_loss"][-patience:]):
                print(f"  Early stopping at epoch {epoch} — val loss has not improved for {patience} epochs.")
                break

        phase = "frozen" if frozen else "joint"
        print(
            f"epoch {epoch:03d} "
            f"[{phase}] "
            f"train={history['train_loss'][-1]:.4f} "
            f"val={history['val_loss'][-1]:.4f} "
            f"bij={val_metrics['bijection_rate']:.3f} "
            f"cost={val_metrics['cost_ratio_vs_hungarian']:.4f} "
            f"match={val_metrics['slot_match_rate']:.3f} "
            f"unass={val_metrics['unassigned_rate']:.3f} "
            f"rounds={val_metrics['consensus_rounds']:.1f}"
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
    raw = load_negotiator_dataset("./dataset/negotiator_dataset_v1.pt")
    matcher = SuperGlueSwarmMatcher()
    train_data, val_data, test_data = prepare_dataset(
        raw,
        matcher.formation_embedding.weight.detach().cpu(),
        seed=seed,
        force_gt_visibility=False,
    )
    model = SuperDecentAuctionModel(matcher).to(device)
    history = train_super_decent_auction(
        model,
        train_data[:7000],
        val_data[:500],
        device,
        epochs=80,
        lr=3e-4,
        ckpt_path="super_decent_auction_best.pt",
    )
    print("matcher validation:", evaluate_superglue_matcher(model.matcher, val_data[:500], device))
    print("auction test:", evaluate_super_decent_auction(model, test_data[:500], device))
