import torch
import numpy as np
from torch_geometric.data import Data
from dataset_generator.features import build_edges

def build_graph(node_features, positions, communication_radius=4.0):
    """
    node_features: (N, F)
    positions: (N, 3)
    """

    edge_index, edge_attr = build_edges(
        positions,
        communication_radius,
        global_velocities=None
    )

    x = torch.tensor(np.array(node_features), dtype=torch.float32)
    pos = torch.tensor(np.array(positions), dtype=torch.float32)

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float32)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos
    )

    return graph