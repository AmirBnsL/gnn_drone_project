import os
import numpy as np
import pybullet as p
from PyFlyt.core import Aviary


# Define a simple wind function matching what you see in example 09
def wind_generator(time: float, position: np.ndarray):
    """Generates an upward draft with random turbulence."""
    wind = np.zeros_like(position)
    
    # Give a base updraft force plus randomness based on position
    wind[:, 2] = np.sin(time) * 0.5 + np.random.normal(0, 0.2, size=(len(position),))
    
    # Push slightly horizontally 
    wind[:, 0] = np.cos(time / 2.0) * 0.3
    wind[:, 1] = np.sin(time / 2.0) * 0.3
    
    return wind


def generate_dataset(
    num_episodes=50,
    max_steps=500,
    dataset_name="formation_dataset",
    dataset_type="random",  # 'random', 'hovering', 'aggressive'
    noisy_sensors=False,
    noise_variance=0.01,
    environmental_wind=False,
    communication_radius=np.inf, # For edge formation
):
    """
    Generates a dataset to train a GNN to imitate the cascaded PID controller for multi-drone formations.
    Time step represents 120 Hz control looprate inherently.
    Data is stored purely in the Local Body Frame (ENU conventions).
    """
    os.makedirs("datasets", exist_ok=True)
    dataset_path = f"datasets/{dataset_name}_{dataset_type}.npz"

    all_states = []
    all_targets = []
    all_labels = []
    all_edges = []

    for ep in range(num_episodes):
        print(f"[{dataset_name}] Starting Episode {ep+1}/{num_episodes} ...")
        
        # Randomize drone config based on dataset_type
        num_drones = np.random.randint(10, 21)
        
        # Randomize initial positions globally
        start_pos = np.random.uniform(-10.0, 10.0, size=(num_drones, 3))
        start_pos[:, 2] = np.random.uniform(0.5, 5.0, size=(num_drones,))  # Z is Up, keep > 0
        
        # Randomize initial yaw, keep roll/pitch flat
        start_orn = np.zeros((num_drones, 3))
        start_orn[:, 2] = np.random.uniform(-np.pi, np.pi, size=(num_drones,))
        
        # Setup Environment for PyFlyt
        env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            drone_type="quadx",
            render=False, # Set to True to visualize
        )
        
        # Inject Environmental Wind Noise
        if environmental_wind:
            env.register_wind_field_function(wind_generator)
        
        # We rely on Mode 7: (x, y, yaw, z) cascaded position control
        env.set_mode(7)

        # Generate Formation Goal Sets
        if dataset_type == "hovering":
            setpoints = np.zeros((num_drones, 4))
            setpoints[:, :2] = start_pos[:, :2]
            setpoints[:, 2]  = start_orn[:, 2]
            setpoints[:, 3]  = start_pos[:, 2]
        else: # aggressive/random
            radius = 10.0 if dataset_type == "aggressive" else 5.0
            setpoints = np.zeros((num_drones, 4))
            setpoints[:, :2] = start_pos[:, :2] + np.random.uniform(-radius, radius, size=(num_drones, 2))
            setpoints[:, 2]  = np.random.uniform(-np.pi, np.pi, size=(num_drones,)) # target yaw
            setpoints[:, 3]  = np.random.uniform(1.0, radius, size=(num_drones,)) # target Z

        env.set_all_setpoints(setpoints)
        
        for step in range(max_steps):
            
            # Record current step data
            episode_states = []
            episode_targets = []
            episode_labels = []
            
            # Track positions for distance graph
            global_positions = []
            
            for i in range(num_drones):
                drone = env.drones[i]
                
                # Fetch PyFlyt states 
                state = drone.state
                global_pos = state[3]
                global_euler = state[1]
                local_lin_vel = state[2]
                local_ang_vel = state[0]
                
                global_positions.append(global_pos)
                
                # Inject Sensor Noise if requested
                if noisy_sensors:
                    global_pos += np.random.normal(0, noise_variance, size=3)
                    global_euler += np.random.normal(0, noise_variance, size=3)
                    local_lin_vel += np.random.normal(0, noise_variance, size=3)
                    local_ang_vel += np.random.normal(0, noise_variance, size=3)

                target_global_pos = np.array([setpoints[i, 0], setpoints[i, 1], setpoints[i, 3]])
                target_global_yaw = setpoints[i, 2]
                
                # Calculate Error
                global_pos_error = target_global_pos - global_pos
                yaw_error = target_global_yaw - global_euler[2]
                yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
                
                # Rotate into Body Frame
                rotation_quaternion = p.getQuaternionFromEuler(global_euler)
                rot_matrix = np.array(p.getMatrixFromQuaternion(rotation_quaternion)).reshape(3, 3)
                local_pos_error = rot_matrix.T @ global_pos_error
                
                gnn_input_state = np.concatenate([local_lin_vel, local_ang_vel])
                gnn_input_target = np.concatenate([local_pos_error, np.array([yaw_error])])
                motor_pwm_labels = drone.pwm 
                
                episode_states.append(gnn_input_state)
                episode_targets.append(gnn_input_target)
                episode_labels.append(motor_pwm_labels)
                
            # Build Graph Edges (Simulating Ad-Hoc Communication Networks)
            global_positions = np.array(global_positions)
            edges = []
            for i in range(num_drones):
                for j in range(num_drones):
                    if i != j:
                        dist = np.linalg.norm(global_positions[i] - global_positions[j])
                        if dist <= communication_radius:
                            edges.append([i, j])
                            
            all_edges.append(edges)
            all_states.append(episode_states)
            all_targets.append(episode_targets)
            all_labels.append(episode_labels)
                
            # Advance simulation (1 control step)
            env.step()

        # Cleanly disconnect for next episode
        env.disconnect()

    # Save to disk
    # Due to variable drone counts, we'll save as object arrays
    all_states = np.array(all_states, dtype=object)
    all_targets = np.array(all_targets, dtype=object)
    all_labels = np.array(all_labels, dtype=object)
    all_edges = np.array(all_edges, dtype=object)
    
    np.savez(dataset_path, states=all_states, targets=all_targets, labels=all_labels, edges=all_edges)
    print(f"✅ Generated Dataset -> {dataset_path}")


if __name__ == "__main__":
    # Generate Clean Dataset
    generate_dataset(num_episodes=20, max_steps=400, dataset_name="clean_comm_10m", dataset_type="random", communication_radius=10.0)
    
    # Generate Sensor Noise Dataset
    generate_dataset(num_episodes=20, max_steps=400, dataset_name="noisy_sensor_comm_5m", dataset_type="random", noisy_sensors=True, noise_variance=0.05, communication_radius=5.0)
    
    # Generate Environmental Wind Dataset
    generate_dataset(num_episodes=20, max_steps=400, dataset_name="turbulent_wind_comm_inf", dataset_type="aggressive", environmental_wind=True, communication_radius=np.inf)
