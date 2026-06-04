"""SetpointGATv2 model definition — V3 with updated input dimensions.

Changes from V2:
- in_channels: 58 → 64 (added integral_pos_err × 2 frames = +6 after engineering)
- Graph-level shift trigger head (mean + max pool → logit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class SetpointGATv2(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, edge_dim, heads, num_layers, dropout):
        super().__init__()
        head_dim = hid_ch // heads

        self.conv_first = GATv2Conv(in_ch, head_dim, heads=heads, edge_dim=edge_dim, concat=True)
        self.proj_first = nn.Linear(in_ch, hid_ch)
        self.norm_first = nn.LayerNorm(hid_ch)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hid_ch, head_dim, heads=heads, edge_dim=edge_dim, concat=True))
            self.norms.append(nn.LayerNorm(hid_ch))

        self.head = nn.Linear(hid_ch, out_ch)
        self.shift_head = nn.Linear(2 * hid_ch, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch=None):
        res = self.proj_first(x)
        x = self.norm_first(self.conv_first(x, edge_index, edge_attr) + res)
        x = F.dropout(F.elu(x), p=self.dropout, training=self.training)

        for conv, norm in zip(self.convs, self.norms):
            res = x
            x = norm(conv(x, edge_index, edge_attr) + res)
            x = F.dropout(F.elu(x), p=self.dropout, training=self.training)

        node_out = self.head(x)
        if batch is None:
            mean_r = x.mean(0, keepdim=True)
            max_r = x.max(0, keepdim=True).values
        else:
            mean_r = global_mean_pool(x, batch)
            max_r = global_max_pool(x, batch)
        graph_repr = torch.cat([mean_r, max_r], dim=-1)
        shift_logit = self.shift_head(graph_repr).squeeze(-1)
        return node_out, shift_logit
