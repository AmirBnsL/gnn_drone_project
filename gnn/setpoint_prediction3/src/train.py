"""
V3 Training Script — GATv2 with custom loss (stillness + direction).

Run: python train.py --train_path ... --val_path ... --test_path ...
Or use from the notebook 12_training_and_evaluation.
"""
import argparse, os, json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model import SetpointGATv2
from dataloader import load_splits, DatasetNormalizer, normalize_batch, engineer_x

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
TRAIN_CONFIG = {
    "in_channels": 64,
    "hidden_channels": 64,
    "out_channels": 4,
    "edge_dim": 7,
    "heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 100,
    "patience": 15,
    "raw_frame_dim": 35,       # V3: 35 (was 32)
    "yaw_idx_in_frame": 25,
    # After engineering: frame is 32 dims. cos(yaw)=idx25, sin(yaw)=idx26 in frame1
    # Frame2 starts at 32, so cos=57, sin=58
    "cos_sin_indices": [25, 26, 57, 58],
    "yaw_quantile": 0.99,
    # Custom loss weights
    "stillness_weight": 0.2,
    "direction_weight": 0.05,
    "stillness_decay": 2.0,     # exp(-decay * d) — 50cm activation radius
    "direction_min_dist": 0.5,  # only apply direction loss when d > 0.5m
}


# ═══════════════════════════════════════════════════════════════════════════
# Custom Loss Function
# ═══════════════════════════════════════════════════════════════════════════
def compute_loss(pred, target, x_raw, cfg):
    """
    V3 Custom Loss = MSE + stillness + direction.

    Args:
        pred: (N, 4) predicted displacement [dx, dy, dz, dyaw]
        target: (N, 4) ground-truth displacement (already scaled to [-1,1])
        x_raw: (N, 64) engineered features (first frame starts at 0)
        cfg: config dict with loss weights
    """
    # 1. Standard MSE
    mse = F.mse_loss(pred, target)

    # Extract local_pos_err from engineered features (indices 22-24 of first frame)
    local_pos_err = x_raw[:, 22:25]

    # 2. Stillness: penalize large XYZ outputs when close to goal
    dist_to_goal = torch.norm(local_pos_err, dim=1)
    at_goal_weight = torch.exp(-dist_to_goal * cfg["stillness_decay"])
    stillness = (pred[:, :3].norm(dim=1) * at_goal_weight).mean()

    # 3. Direction: align with goal direction (only when far enough)
    far_mask = dist_to_goal > cfg["direction_min_dist"]
    if far_mask.any():
        pred_xy = pred[far_mask, :2]
        goal_xy = local_pos_err[far_mask, :2]
        # Normalize to avoid instability
        pn = pred_xy.norm(dim=1, keepdim=True).clamp(min=1e-4)
        gn = goal_xy.norm(dim=1, keepdim=True).clamp(min=1e-4)
        direction = 1.0 - F.cosine_similarity(pred_xy / pn, goal_xy / gn, dim=1).mean()
    else:
        direction = torch.tensor(0.0, device=pred.device)

    total = mse + cfg["stillness_weight"] * stillness + cfg["direction_weight"] * direction
    return total, mse.item(), stillness.item(), direction.item()


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, normalizer, cfg, device):
    model.train()
    total_loss, total_mse, total_still, total_dir, n = 0, 0, 0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        # Store raw x before normalization for loss computation
        x_eng = engineer_x(batch.x, cfg["raw_frame_dim"], cfg["yaw_idx_in_frame"])
        batch = normalize_batch(batch, normalizer)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss, mse_v, still_v, dir_v = compute_loss(pred, batch.target, x_eng, cfg)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = batch.num_graphs
        total_loss += loss.item()*bs; total_mse += mse_v*bs
        total_still += still_v*bs; total_dir += dir_v*bs; n += bs
    return total_loss/n, total_mse/n, total_still/n, total_dir/n


@torch.no_grad()
def evaluate(model, loader, normalizer, cfg, device):
    model.eval()
    total_loss, total_mse, n = 0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        x_eng = engineer_x(batch.x, cfg["raw_frame_dim"], cfg["yaw_idx_in_frame"])
        batch = normalize_batch(batch, normalizer)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss, mse_v, _, _ = compute_loss(pred, batch.target, x_eng, cfg)
        bs = batch.num_graphs
        total_loss += loss.item()*bs; total_mse += mse_v*bs; n += bs
    return total_loss/n, total_mse/n


def train(train_path, val_path, test_path, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, test_ds = load_splits(train_path, val_path, test_path)
    normalizer = DatasetNormalizer.fit(train_ds, cfg).to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    model = SetpointGATv2(
        in_ch=cfg["in_channels"], hid_ch=cfg["hidden_channels"],
        out_ch=cfg["out_channels"], edge_dim=cfg["edge_dim"],
        heads=cfg["heads"], num_layers=cfg["num_layers"], dropout=cfg["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val, patience_counter = float("inf"), 0

    for epoch in range(1, cfg["epochs"]+1):
        t_loss, t_mse, t_still, t_dir = train_one_epoch(model, train_loader, optimizer, normalizer, cfg, device)
        v_loss, v_mse = evaluate(model, val_loader, normalizer, cfg, device)
        scheduler.step(v_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d} | Train Loss: {t_loss:.5f} (MSE:{t_mse:.5f} Still:{t_still:.5f} Dir:{t_dir:.5f}) | Val: {v_loss:.5f} | LR: {lr:.6f}")

        if v_loss < best_val:
            best_val = v_loss; patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_gatv2.pth"))
            normalizer.save(os.path.join(ckpt_dir, "normalization_stats.pt"))
            print(f"  → Saved best model (val_loss={best_val:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}"); break

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_gatv2.pth"), weights_only=True))
    test_loss, test_mse = evaluate(model, test_loader, normalizer, cfg, device)
    print(f"\nTest Loss: {test_loss:.5f} | Test MSE: {test_mse:.5f}")

    results = {"best_val_loss": best_val, "test_loss": test_loss, "test_mse": test_mse, "config": cfg}
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "eval_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--test_path", required=True)
    args = parser.parse_args()
    train(args.train_path, args.val_path, args.test_path, TRAIN_CONFIG)
