from dataset_generator import generate_dataset_parallel

if __name__ == "__main__":
    # Default dataset generator config (similar to the original script)
    config = {
        "num_episodes": 600,
        "max_steps": 1500,
        "worker_batch_size": 60,
        "num_workers": 6,
        "dataset_name": "setpoint_prediction_dataset_parallel_apf_40000",
        "dataset_type": "mixed_formations",
        "task_type": "setpoint_prediction",
        "noisy_sensors": False,
        "environmental_wind": False,
        "dynamic_formation": False,
        "inject_failures": False,
        "communication_radius": 10.0,
        "include_formation_in_state": True,
        "tapered_sampling": True,
        "conv_stopping": True,
        "conv_threshold": 0.2,
        "num_obstacles": (0, 10),
        "seed": 12345,
        "obstacle_radius_range": (0.4, 1.8),
        "apf_enabled": True,
        "apf_attractive_gain": 1.0,
        "apf_repulsive_gain": 1.2,
        "apf_repulsion_padding": 0.8,
        "apf_max_step_size": 1.5,
        "apf_vertical_gain": 1.0,
        "residual_balance_ratio": 0.5,
        "residual_dropout_threshold": 0.1,
    }

    generate_dataset_parallel(**config)


