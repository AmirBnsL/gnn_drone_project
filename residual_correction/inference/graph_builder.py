import torch
from torch_geometric.data import Data

def build_graph(node_features, positions, communication_radius=4.0):
    x = torch.tensor(node_features, dtype=torch.float32)

    pos = torch.tensor(positions, dtype=torch.float32)

    num_nodes = x.shape[0]

    # fully connected fallback (safe version)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, pos=pos, edge_index=edge_index)