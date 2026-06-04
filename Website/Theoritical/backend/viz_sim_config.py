"""
Viz-only simulation knobs (Website/Theoritical).

These do NOT affect merged_work / Kaggle training until you copy values there.
"""

from __future__ import annotations

# Run until convergence (early exit); MAX_STEPS is a safety cap only.
RUN_UNTIL_CONVERGED = True
MAX_STEPS = 2000  # default cap when RUN_UNTIL_CONVERGED is False
MAX_STEPS_CONVERGED = 6000  # safety cap when RUN_UNTIL_CONVERGED is True
RECORD_EVERY = 2


def simulation_step_cap(max_steps: int | None = None) -> int:
    """Physics-step limit for one recorder episode (override or default cap)."""
    if max_steps is not None:
        return int(max_steps)
    if RUN_UNTIL_CONVERGED:
        return int(MAX_STEPS_CONVERGED)
    return int(MAX_STEPS)

# Early stop when all drones within threshold for this many consecutive steps
CONV_THRESHOLD = 0.35
CONV_STEPS = 25

# Slot shift: full formation +Z when stuck near blocked slot
SLOT_SHIFT_ONCE = True
SLOT_SHIFT_DELTA_Z = 2.0
SHIFT_MIN_STEPS = 20
SHIFT_REQUIRE_FULL_VEL_WINDOW = True
SHIFT_MIN_SLOT_ERROR = 0.4
SHIFT_MAX_SLOT_ERROR = 5.0
SHIFT_REQUIRE_LOCAL_STUCK_COUNT = 1
SHIFT_MAX_ALONG_DIST = 3.0

# Stuck detection (viz experiment)
VEL_STUCK_EPS = 0.12  # matches setpoint_rollout.VEL_EPS
VEL_STUCK_HISTORY_LEN = 5
STUCK_MAX_ALONG_DIST = 5.0  # legacy central ray gate if strict path unused

# Runtime obstacle safety (Website only)
SAFETY_CLEARANCE = 0.35
SAFETY_MIN_SURFACE_DIST = 0.25
SAFETY_CLAMP_PUSH_SCALE = 1.0  # multiplier on repulsive nudge in safety_clamp_setpoint
APF_PROXIMITY_MARGIN = 0.5  # treat obstacle as visible if surface dist below clearance+margin

# Viz-only APF strength (central / decentral recorders)
CENTRAL_APF_OBS_INFLUENCE = 4.0
CENTRAL_APF_FORCE_CAP = 2.5
DECENTRAL_APF_OBS_INFLUENCE = 4.0
DECENTRAL_APF_FORCE_CAP = 2.5
# 3D obstacle altitude filtering margin (m) for Website runtime safety/APF.
OBSTACLE_ALTITUDE_MARGIN = 0.3

# Decentralized GNN simulation
COMM_RADIUS = 10.0
SLOT_VISIBILITY_RADIUS = 10.0
SHIFT_TRIGGER_THRESHOLD = 0.5
SHIFT_COOLDOWN_STEPS = 30
SETPOINT_CTRL_EVERY = 1
LEADER_DRONE_ID = 0

# Setpoint GNN: hybrid pred + pull toward assigned slot (PyFlyt needs strong goal term).
SETPOINT_PRED_GAIN = 0.65
SETPOINT_GOAL_GAIN = 0.35
SETPOINT_GOAL_BOOST_DIST = 2.0  # extra pull toward slot when within this distance (m)
SETPOINT_GOAL_BOOST = 0.25  # added to goal_weight when close to slot
SETPOINT_GAIN_MAX = 25.0
SETPOINT_D_HALF = 2.0
SETPOINT_YAW_GAIN = 3.0
SETPOINT_RAMP_STEPS = 1
SETPOINT_MAX_STEP_XY = 2.0
SETPOINT_MAX_STEP_Z = 1.0
SETPOINT_MAX_STEP_YAW = 0.4

# Learned shift head gate (requires shift_weight > 0 in training); off by default.
ENABLE_HYBRID_SLOT_SHIFT = False
# Decentral goal repair: Z slot shift from strict local stuck (no runtime APF).
ENABLE_DECENTRAL_GOAL_REPAIR = True
# Decentral APF safety layer (matches training runtime prior); flag-gated.
ENABLE_DECENTRAL_APF = True
