"""Message types for decentralized swarm simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StateMsg:
    sender_id: int
    pos: List[float]
    yaw: float
    local_vel: List[float]
    alert: bool
    stuck: bool


@dataclass
class ShiftProposal:
    sender_id: int
    step: int


@dataclass
class ShiftEvent:
    step: int
    fired: bool


@dataclass
class AssignmentResult:
    assignment: List[int]
    messages_sent: int = 0
