"""Comm-radius messaging and sequential inference turn logging."""

from __future__ import annotations

from typing import List

import numpy as np

from agents.drone_agent import DroneAgent


def exchange_messages(
    agents: List[DroneAgent],
    positions: List[np.ndarray],
    yaws: List[float],
    local_vels: List[np.ndarray],
    alerts: List[bool],
    stuck_flags: List[bool],
    step: int,
) -> int:
    """Symmetric delivery within comm_radius; returns ShiftProposal delivery count."""
    n = len(agents)
    for agent in agents:
        agent.clear_mailboxes()
    for i, agent in enumerate(agents):
        agent.local_stuck = stuck_flags[i]
        agent.prepare_outbox(positions[i], yaws[i], local_vels[i], alerts[i], stuck_flags[i], step)

    proposal_count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pi, pj = positions[i], positions[j]
            if float(np.linalg.norm(pi - pj)) <= agents[i].comm_radius:
                agents[i].receive(agents[j])
                proposal_count += len(agents[j].outbox_shift_proposals)

    return proposal_count


def swarm_has_shift_proposal(agents: List[DroneAgent]) -> bool:
    return any(a.has_shift_proposal() for a in agents)


def sequential_turn_log(n: int) -> List[int]:
    return list(range(n))
