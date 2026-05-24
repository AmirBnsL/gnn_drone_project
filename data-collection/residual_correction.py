from dataset_generator import generate_dataset_parallel

if __name__ == "__main__":
    # Default dataset generator config (similar to the original script)
    config = {
        "num_episodes": 5000,
        "worker_batch_size": 60,
        "num_workers": 6,
        "dataset_name": "residual_correction_datasetE_parallel_apf_50000",
        "dataset_type": "mixed_formations",
        "task_type": "residual_correction",
        "noisy_sensors": True,
        "noise_variance": 0.03,
        "environmental_wind": True,
        "dynamic_formation": True,
        "inject_failures": False,
        "communication_radius": 4.0,
        "include_formation_in_state": True,
        "tapered_sampling": True,
        "conv_stopping": False,
        "conv_threshold": 0.2,
        "num_obstacles": (30,45),
        "seed": 12345,
        "obstacle_radius_range": (0.4, 1.8),
        "apf_enabled": False,
        "residual_balance_ratio": 0.5,
        "residual_dropout_threshold": 0.1,
    }

    generate_dataset_parallel(**config)
