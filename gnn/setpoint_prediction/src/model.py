"""
Setpoint Prediction — GATv2 Model

Decentralized Navigator GNN for drone swarm setpoint prediction.
Predicts body-frame setpoint errors [Δx_local, Δy_local, Δz, Δyaw] from local
state + neighbor communication via 3-hop GATv2 message passing.

Usage:
    from model import SetpointGATv2, load_gnn_model

    model = SetpointGATv2(in_channels=10, edge_dim=4)
    pred = model(batch.x, batch.edge_index, batch.edge_attr)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class SetpointGATv2(nn.Module):
    """
    GATv2-based Navigator GNN for decentralized setpoint prediction.

    Architecture:
      K × GATv2Conv (multi-head attention with edge features)
      + residual skip connections
      + LayerNorm + ELU activation
      + 2-layer MLP head → unbounded 4-dim output

    Args:
        in_channels:  Input feature dim per node (default: 10)
        edge_dim:     Edge feature dim (default: 4)
        hidden_dim:   Hidden dimension after each GATv2 layer (default: 64)
        out_channels: Output dim — setpoint components (default: 4)
        heads:        Number of attention heads per layer (default: 4)
        num_layers:   Number of GATv2 layers / K-hops (default: 3)
        dropout:      Dropout rate (default: 0.05)
    """

    def __init__(
        self,
        in_channels:  int = 10,
        edge_dim:     int = 4,
        hidden_dim:   int = 64,
        out_channels: int = 4,
        heads:        int = 4,
        num_layers:   int = 3,
        dropout:      float = 0.05,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout    = dropout

        # Store hyperparams for checkpointing
        self.hparams = {
            "in_channels":  in_channels,
            "edge_dim":     edge_dim,
            "hidden_dim":   hidden_dim,
            "out_channels": out_channels,
            "heads":        heads,
            "num_layers":   num_layers,
            "dropout":      dropout,
        }

        # ── GATv2 Layers ──────────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_dim
            out_per_head = hidden_dim // heads

            self.convs.append(
                GATv2Conv(
                    in_channels=c_in,
                    out_channels=out_per_head,
                    heads=heads,
                    edge_dim=edge_dim,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

            if c_in != hidden_dim:
                self.skips.append(nn.Linear(c_in, hidden_dim, bias=False))
            else:
                self.skips.append(nn.Identity())

        # ── MLP Head ──────────────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_channels),
            # NO activation — unbounded setpoint output
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:          [N, in_channels]  Node features
            edge_index: [2, E]            Communication graph
            edge_attr:  [E, edge_dim]     Edge features

        Returns:
            pred:       [N, out_channels] Predicted setpoints (normalized)
        """
        h = x
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index, edge_attr)
            h_skip = self.skips[i](h)
            h_new = h_new + h_skip
            h_new = self.norms[i](h_new)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new

        return self.mlp(h)


def load_gnn_model(path, device="cpu"):
    """
    Load a complete checkpoint.

    Returns:
        (model, normalizer, checkpoint_dict)
    """
    from dataset import SetpointNormalizer

    ckpt = torch.load(path, map_location=device, weights_only=False)

    model = SetpointGATv2(**ckpt["model_hparams"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    normalizer = SetpointNormalizer.from_dict(ckpt["normalizer"])

    return model, normalizer, ckpt
