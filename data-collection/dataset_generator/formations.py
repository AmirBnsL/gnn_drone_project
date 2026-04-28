import numpy as np
from scipy.optimize import linear_sum_assignment

def resolve_formation_name(dataset_type: str):
    if dataset_type in {"a", "formation_a"}: return "a"
    if dataset_type in {"rectangle", "rectangular", "formation_rectangle"}: return "rectangle"
    if dataset_type in {"triangle", "formation_triangle"}: return "triangle"
    return None

def _formation_a_offsets(num_drones: int, spacing: float = 2.0):
    offsets = np.zeros((num_drones, 3))
    if num_drones == 1: return offsets
    num_crossbar = (num_drones // 5) if num_drones > 5 else 0
    num_v_legs = num_drones - num_crossbar
    if num_v_legs % 2 == 0:
        num_crossbar += 1
        num_v_legs -= 1
    offsets[0] = np.array([0.0, 0.0, 0.0])
    for idx in range(1, num_v_legs):
        level = (idx + 1) // 2
        side = -1.0 if idx % 2 == 1 else 1.0
        offsets[idx, 0] = side * level * spacing
        offsets[idx, 1] = level * spacing
    if num_crossbar > 0:
        max_level = (num_v_legs + 1) // 2
        mid_level = max(1, max_level // 2)
        left_x = -mid_level * spacing
        right_x = mid_level * spacing
        width = right_x - left_x
        for crossbar_idx in range(num_crossbar):
            fraction = (crossbar_idx + 1) / (num_crossbar + 1)
            target_idx = num_v_legs + crossbar_idx
            offsets[target_idx, 0] = left_x + (fraction * width)
            offsets[target_idx, 1] = mid_level * spacing
    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets

def _formation_rectangle_offsets(num_drones: int, spacing: float = 2.0):
    offsets = np.zeros((num_drones, 3))
    if num_drones == 0: return offsets
    perimeter = num_drones * spacing
    side = perimeter / 4.0
    corners = [
        np.array([-side / 2, -side / 2]),
        np.array([side / 2, -side / 2]),
        np.array([side / 2, side / 2]),
        np.array([-side / 2, side / 2]),
    ]
    for i in range(min(num_drones, 4)):
        offsets[i, :2] = corners[i]
    if num_drones > 4:
        remaining = num_drones - 4
        drones_per_edge = [remaining // 4 + (1 if e < remaining % 4 else 0) for e in range(4)]
        current_idx = 4
        for e in range(4):
            start_c = corners[e]
            end_c = corners[(e + 1) % 4]
            num_edge = drones_per_edge[e]
            for step in range(1, num_edge + 1):
                fraction = step / (num_edge + 1)
                offsets[current_idx, :2] = start_c + fraction * (end_c - start_c)
                current_idx += 1
    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets

def _formation_triangle_offsets(num_drones: int, spacing: float = 2.0):
    offsets = np.zeros((num_drones, 3))
    if num_drones == 0: return offsets
    perimeter = num_drones * spacing
    side = perimeter / 3.0
    h = side * np.sqrt(3) / 2.0
    corners = [
        np.array([-side / 2, -h / 3]),
        np.array([side / 2, -h / 3]),
        np.array([0, 2 * h / 3]),
    ]
    for i in range(min(num_drones, 3)):
        offsets[i, :2] = corners[i]
    if num_drones > 3:
        remaining = num_drones - 3
        drones_per_edge = [remaining // 3 + (1 if e < remaining % 3 else 0) for e in range(3)]
        current_idx = 3
        for e in range(3):
            start_c = corners[e]
            end_c = corners[(e + 1) % 3]
            num_edge = drones_per_edge[e]
            for step in range(1, num_edge + 1):
                fraction = step / (num_edge + 1)
                offsets[current_idx, :2] = start_c + fraction * (end_c - start_c)
                current_idx += 1
    offsets[:, :2] -= np.mean(offsets[:, :2], axis=0)
    return offsets

def apply_obstacle_avoidance(slots, obstacles, obstacle_radii, padding=1.0):
    if len(obstacles) == 0: return slots
    safe_slots = np.copy(slots)
    obs_xy = np.asarray(obstacles, dtype=np.float32)[:, :2]
    if obstacle_radii is None or len(obstacle_radii) == 0:
        obstacle_radii = np.ones((len(obstacles),), dtype=np.float32)
    for i in range(len(safe_slots)):
        for obs_idx, obs in enumerate(obs_xy):
            min_dist = float(obstacle_radii[obs_idx]) + padding
            diff = safe_slots[i, :2] - obs
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                direction = diff / (dist + 1e-6)
                safe_slots[i, :2] = obs + direction * min_dist
    return safe_slots

def _build_formation_setpoints(
    formation_name: str,
    start_pos: np.ndarray,
    obstacles=np.empty((0, 2)),
    obstacle_radii=np.empty((0,)),
):
    num_drones = len(start_pos)
    formation_center = np.mean(start_pos[:, :2], axis=0)
    target_altitude = np.mean(start_pos[:, 2])

    if formation_name == "a": offsets = _formation_a_offsets(num_drones)
    elif formation_name == "rectangle": offsets = _formation_rectangle_offsets(num_drones)
    elif formation_name == "triangle": offsets = _formation_triangle_offsets(num_drones)
    else: return None, None, None

    global_slots = np.zeros((num_drones, 3))
    global_slots[:, :2] = formation_center + offsets[:, :2]
    safe_global_slots = apply_obstacle_avoidance(global_slots, obstacles, obstacle_radii, padding=1.0)
    dist_matrix = np.linalg.norm(start_pos[:, None, :2] - safe_global_slots[None, :, :2], axis=2)
    _, assigned_indices = linear_sum_assignment(dist_matrix)

    setpoints = np.zeros((num_drones, 4))
    for i in range(num_drones):
        slot_idx = assigned_indices[i]
        setpoints[i, :2] = safe_global_slots[slot_idx, :2]
        setpoints[i, 2] = 0.0
        setpoints[i, 3] = target_altitude

    return setpoints, assigned_indices, offsets
