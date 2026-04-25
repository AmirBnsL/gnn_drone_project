"""SetpointGATv2 model definition — shared between training and inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


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
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        res = self.proj_first(x)
        x = self.norm_first(self.conv_first(x, edge_index, edge_attr) + res)
        x = F.dropout(F.elu(x), p=self.dropout, training=self.training)

        for conv, norm in zip(self.convs, self.norms):
            res = x
            x = norm(conv(x, edge_index, edge_attr) + res)
            x = F.dropout(F.elu(x), p=self.dropout, training=self.training)

        return self.head(x)
