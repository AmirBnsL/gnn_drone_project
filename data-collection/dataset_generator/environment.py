import numpy as np
import pybullet as p
from PyFlyt.core import Aviary

from dataset_generator.utils import wind_generator
from dataset_generator.formations import resolve_formation_name, _build_formation_setpoints

def create_aviary(
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    environmental_wind: bool,
    obstacles: np.ndarray = np.empty((0, 3)), 
    obstacle_radii: np.ndarray | None = None,
    obstacle_radius: float = 1.0,
    graphical: bool = False,
):
    drone_options = dict()
    drone_options["control_hz"] = 60
    env = Aviary(start_pos=start_pos, start_orn=start_orn, drone_type="quadx", render=graphical, physics_hz=240, drone_options=drone_options)
    
    if environmental_wind: 
        env.register_wind_field_function(wind_generator)
        
    env.set_mode(7)
    
    if len(obstacles) > 0:
        physics_client = env._client
        if obstacle_radii is None or len(obstacle_radii) == 0:
            obstacle_radii_arr = np.full((len(obstacles),), obstacle_radius, dtype=np.float32)
        else:
            obstacle_radii_arr = np.asarray(obstacle_radii, dtype=np.float32)
            
        for obs, obs_radius in zip(obstacles, obstacle_radii_arr):
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=float(obs_radius), physicsClientId=physics_client)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=float(obs_radius), rgbaColor=[1, 0, 0, 0.5], physicsClientId=physics_client)
            
            p.createMultiBody(
                baseMass=0, 
                baseCollisionShapeIndex=col_id, 
                baseVisualShapeIndex=vis_id, 
                basePosition=[obs[0], obs[1], obs[2]], 
                physicsClientId=physics_client
            )
            
        env.register_all_new_bodies()
        
    return env


def build_setpoints(
    dataset_type: str,
    start_pos: np.ndarray,
    start_orn: np.ndarray,
    rng: np.random.Generator,
    obstacles: np.ndarray = np.empty((0, 2)),
    obstacle_radii: np.ndarray | None = None,
    obstacle_radius: float = 1.0,
):
    num_drones = len(start_pos)
    if obstacle_radii is None:
        obstacle_radii = np.full((len(obstacles),), obstacle_radius, dtype=np.float32)
    formation_name = resolve_formation_name(dataset_type)
    if formation_name is not None:
        setpoints, col_ind, offsets = _build_formation_setpoints(formation_name, start_pos, obstacles, obstacle_radii)
        if setpoints is not None: return setpoints, col_ind, offsets

    setpoints = np.zeros((num_drones, 4))
    if dataset_type == "hovering":
        setpoints[:, :2] = start_pos[:, :2]
        setpoints[:, 2] = start_orn[:, 2]
        setpoints[:, 3] = start_pos[:, 2]
    else:
        radius = 10.0 if dataset_type == "aggressive" else 5.0
        setpoints[:, :2] = start_pos[:, :2] + rng.uniform(-radius, radius, size=(num_drones, 2))
        setpoints[:, 2] = rng.uniform(-np.pi, np.pi, size=(num_drones,))
        setpoints[:, 3] = rng.uniform(1.0, radius, size=(num_drones,))
    col_ind = np.arange(num_drones)
    return setpoints, col_ind, np.zeros((num_drones, 3))


def compute_apf_setpoints(
    current_positions: np.ndarray,
    final_setpoints: np.ndarray,
    obstacles: np.ndarray,
    obstacle_radii: np.ndarray,
    attractive_gain: float,
    repulsive_gain: float,
    repulsion_padding: float,
    max_step_size: float,
    vertical_gain: float,
):
    if len(current_positions) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    final_target_positions = np.column_stack(
        [final_setpoints[:, 0], final_setpoints[:, 1], final_setpoints[:, 3]]
    )

    attractive = final_target_positions - current_positions
    attractive[:, :2] *= attractive_gain
    attractive[:, 2] *= vertical_gain

    repulsive_xyz = np.zeros((len(current_positions), 3), dtype=np.float32)

    if len(obstacles) > 0:
        if obstacle_radii is None or len(obstacle_radii) == 0:
            obstacle_radii = np.ones((len(obstacles),), dtype=np.float32)

        diff = current_positions[:, None, :3] - obstacles[None, :, :3]
        dist = np.linalg.norm(diff, axis=2) + 1e-6

        influence_radius = obstacle_radii[None, :] + repulsion_padding
        inside_influence = dist < influence_radius

        safe_dist = np.maximum(dist, 1e-3)
        repulse_strength = (
            repulsive_gain * (1.0 / safe_dist - 1.0 / influence_radius) / (safe_dist**2)
        )
        repulse_strength = np.where(inside_influence, repulse_strength, 0.0)

        direction = diff / safe_dist[..., None]
        repulsive_xyz = np.sum(repulse_strength[..., None] * direction, axis=1)

    total_delta = np.copy(attractive)
    total_delta[:, :3] += repulsive_xyz

    norm = np.linalg.norm(total_delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_step_size / (norm + 1e-6))
    bounded_delta = total_delta * scale
    intermediate_positions = current_positions + bounded_delta

    intermediate_setpoints = np.zeros_like(final_setpoints)
    intermediate_setpoints[:, 0] = intermediate_positions[:, 0]
    intermediate_setpoints[:, 1] = intermediate_positions[:, 1]
    intermediate_setpoints[:, 2] = final_setpoints[:, 2]
    intermediate_setpoints[:, 3] = intermediate_positions[:, 2]
    return intermediate_setpoints
