from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ZeroResidualRegressor(nn.Module):
    def forward(self, data):
        return torch.zeros(
            (data.x.size(0), 3),
            dtype=data.x.dtype,
            device=data.x.device,
        )

class MLPResidualRegressor(nn.Module):
    def __init__(self, in_dim: int = 13, hidden_dim: int = 64, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        return self.net(data.x)


class GATResidualRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int = 13,
        hidden_dim: int = 32,
        out_dim: int = 3,
        heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=3,
            dropout=dropout,
        )

        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            edge_dim=3,
            dropout=dropout,
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp_head(x)

class NNConvResidualRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int = 13,
        hidden_dim: int = 32,
        out_dim: int = 3,
    ):
        super().__init__()

        from torch_geometric.nn import NNConv, GraphNorm

        # edge network for conv1: maps edge_attr(3) -> in_dim * hidden_dim
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, in_dim * hidden_dim),
        )

        # edge network for conv2: maps edge_attr(3) -> hidden_dim * hidden_dim
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim * hidden_dim),
        )

        self.conv1 = NNConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            nn=self.edge_mlp1,
            aggr="mean",
        )
        self.norm1 = GraphNorm(hidden_dim)

        self.conv2 = NNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            nn=self.edge_mlp2,
            aggr="mean",
        )
        self.norm2 = GraphNorm(hidden_dim)

        self.skip_proj = nn.Linear(in_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x0 = self.skip_proj(x)

        h = self.conv1(x, edge_index, edge_attr)
        h = self.norm1(h, batch)
        h = F.relu(h)

        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h, batch)

        h = F.relu(h + x0)   # residual skip

        return self.head(h)

def build_model(model_name: str, in_dim: int = 13):
    model_name = model_name.lower()
    if model_name == "zero":
        return ZeroResidualRegressor()
    elif model_name == "mlp":
        return MLPResidualRegressor(in_dim=in_dim)
    elif model_name == "gat":
        return GATResidualRegressor(in_dim=in_dim)
    elif model_name == "nnconv":
        return NNConvResidualRegressor(in_dim=in_dim)
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Use 'mlp' or 'gat' or 'zero' or 'nnconv'.")


if __name__ == "__main__":
    from dataset_loader import get_datasets

    dataset_dir = "residual_correction/datasets"
    train_dataset, _, _ = get_datasets(dataset_dir)
    sample = train_dataset[0]

    for name in ["zero","mlp", "gat","nnconv"]:
        model = build_model(name, in_dim=sample.x.shape[1])
        out = model(sample)
        print(f"\nModel: {name}")
        print("Output shape:", tuple(out.shape))
        print("Target shape:", tuple(sample.target.shape))