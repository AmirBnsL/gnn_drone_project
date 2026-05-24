import numpy as np
from dataset_generator.features import compute_lidar_features

def extract_node_features(
    drone,
    obstacles,
    obstacle_radii,
    noisy_sensors=False,
    noise_variance=0.01,
    include_formation=False,
    formation_one_hot=None,
    physics_client=None
):
    state = drone.state

    global_pos = np.array(state[3], dtype=np.float32)
    global_euler = np.array(state[1], dtype=np.float32)
    local_lin_vel = np.array(state[2], dtype=np.float32)
    local_ang_vel = np.array(state[0], dtype=np.float32)

    # lidar (EXACT same as training)
    obs_features = compute_lidar_features(
        global_pos[None, :],
        global_euler[None, :],
        obstacles,
        obstacle_radii,
        physics_client=physics_client,
    )[0]

    # optional noise (same idea as training)
    if noisy_sensors:
        local_lin_vel += np.random.normal(0, noise_variance, 3)
        local_ang_vel += np.random.normal(0, noise_variance, 3)

    node_feature = np.concatenate([
        local_lin_vel,
        local_ang_vel,
        obs_features
    ])

    if include_formation and formation_one_hot is not None:
        node_feature = np.concatenate([node_feature, formation_one_hot])

    return node_feature.astype(np.float32), global_pos