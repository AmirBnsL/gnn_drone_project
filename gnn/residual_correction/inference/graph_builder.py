import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_COLLECTION_PATH = PROJECT_ROOT / "data-collection"

sys.path.append(str(DATA_COLLECTION_PATH))

import torch
import numpy as np
from torch_geometric.data import Data

from dataset_generator.features import build_edges

def build_graph(
    node_features,
    positions,
    communication_radius=10.0
):

    x = torch.tensor(node_features, dtype=torch.float32)
    print("graph x shape AFTER tensor:", x.shape)
    positions = np.asarray(positions, dtype=np.float32)

    zero_velocities = np.zeros_like(positions)

    edges, edge_attrs = build_edges(
        positions,
        communication_radius,
        zero_velocities
    )

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 7), dtype=torch.float32)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(positions, dtype=torch.float32)
    )

    return graph