import numpy as np
import pybullet as p

def extract_node_features(
    drone,
    obstacles,
    obstacle_radii,
    noisy_sensors=False,
    include_formation=True,
    formation_one_hot=None,
    physics_client=None
):
    state = drone.state

    global_pos = np.array(state[3], copy=True)
    global_euler = np.array(state[1], copy=True)
    local_lin_vel = np.array(state[2], copy=True)
    local_ang_vel = np.array(state[0], copy=True)

    # IMPORTANT: must match training noise logic if used
    if noisy_sensors:
        noise = 0.03
        global_pos += np.random.normal(0, noise, 3)
        global_euler += np.random.normal(0, noise, 3)
        local_lin_vel += np.random.normal(0, noise, 3)
        local_ang_vel += np.random.normal(0, noise, 3)

    # lidar fallback (simple version for inference safety)
    obs_features = np.zeros(8, dtype=np.float32)

    feat = np.concatenate([
        local_lin_vel,
        local_ang_vel,
        obs_features
    ])

    if include_formation and formation_one_hot is not None:
        feat = np.concatenate([feat, formation_one_hot])

    return feat, global_pos