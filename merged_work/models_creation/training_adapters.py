from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass
class AssignmentSample:
    """Lightweight container mirroring negotiator dataset fields."""

    graph: Data


@dataclass
class SetpointSample:
    """Container for a single setpoint‑prediction graph."""

    graph: Data


def to_negotiator_sample(graph: Data) -> AssignmentSample:
    """
    Adapt a generic assignment graph into the shape expected by
    `local_negotiator.prepare_dataset` and the Bertsekas auction model.

    Required fields on `graph`:
      - drone_pos: (N, 2)
      - slots: (N, 2)
      - y: (N,) long
      - formation_id: scalar long
    """
    assert hasattr(graph, "drone_pos") and hasattr(graph, "slots"), "graph is missing drone_pos/slots"
    assert hasattr(graph, "y"), "graph is missing y labels"
    assert hasattr(graph, "formation_id"), "graph is missing formation_id"

    # We keep x as‑is; downstream feature builders recompute node features.
    if graph.x is None:
        graph.x = torch.zeros((graph.drone_pos.size(0), 1), dtype=torch.float32)
    return AssignmentSample(graph=graph)


def to_setpoint_v3_sample(graph: Data) -> SetpointSample:
    """
    Adapt a generic physics rollout graph into the contract used by
    `setpoint_prediction3`:

      - x: engineered/stacked features (later processed by engineer_x/normalize_batch)
      - edge_index: (2, E)
      - edge_attr: (E, 7)
      - target: (N, 4) setpoint labels
    """
    assert hasattr(graph, "x"), "graph is missing x"
    assert hasattr(graph, "edge_index"), "graph is missing edge_index"
    assert hasattr(graph, "edge_attr"), "graph is missing edge_attr"
    assert hasattr(graph, "target"), "graph is missing target"

    return SetpointSample(graph=graph)

