"""Per-drone state for decentralized simulation (no local GNN copy)."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np

from agents.messages import ShiftProposal, StateMsg

RAW_FRAME_DIM = 41


class DroneAgent:
    def __init__(self, drone_id: int, comm_radius: float, vel_hist_len: int, int_buf_len: int):
        self.id = drone_id
        self.comm_radius = comm_radius
        self.my_slot_idx: int = -1
        self.setpoint = np.zeros(4, dtype=np.float32)
        self.prev_frame = np.zeros(RAW_FRAME_DIM, dtype=np.float32)
        self.vel_hist: Deque[np.ndarray] = deque(maxlen=vel_hist_len)
        self.int_buf: Deque[np.ndarray] = deque(maxlen=int_buf_len)
        self.inbox_states: List[StateMsg] = []
        self.inbox_shift_proposals: List[ShiftProposal] = []
        self.outbox_states: List[StateMsg] = []
        self.outbox_shift_proposals: List[ShiftProposal] = []
        self.local_stuck: bool = False

    def clear_mailboxes(self) -> None:
        self.inbox_states.clear()
        self.inbox_shift_proposals.clear()
        self.outbox_states.clear()
        self.outbox_shift_proposals.clear()

    def prepare_outbox(
        self,
        pos: np.ndarray,
        yaw: float,
        local_vel: np.ndarray,
        alert: bool,
        stuck: bool,
        step: int,
    ) -> None:
        self.local_stuck = stuck
        self.outbox_states = [
            StateMsg(
                sender_id=self.id,
                pos=[float(pos[0]), float(pos[1]), float(pos[2])],
                yaw=float(yaw),
                local_vel=[float(local_vel[0]), float(local_vel[1]), float(local_vel[2])],
                alert=bool(alert),
                stuck=bool(stuck),
            )
        ]
        self.outbox_shift_proposals = []
        if stuck:
            self.outbox_shift_proposals.append(ShiftProposal(sender_id=self.id, step=step))

    def receive(self, other: "DroneAgent") -> None:
        for msg in other.outbox_states:
            self.inbox_states.append(msg)
        for prop in other.outbox_shift_proposals:
            self.inbox_shift_proposals.append(prop)

    def has_shift_proposal(self) -> bool:
        if self.local_stuck or self.outbox_shift_proposals:
            return True
        return len(self.inbox_shift_proposals) > 0
