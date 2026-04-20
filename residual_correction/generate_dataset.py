from datacollection_wrapper import load_generate_dataset
import shutil
from pathlib import Path

generate_dataset = load_generate_dataset()

if __name__ == "__main__":
    generated_files, metadata_path = generate_dataset(
        dataset_name="residual_failures_dense",
        dataset_type="mixed_formations",
        task_type="residual_correction",
        num_episodes=80,
        num_obstacles=6,
        obstacle_radius=2.0,
        include_formation_in_state=True,
        communication_radius=10.0,
        noisy_sensors=True,
        noise_variance=0.01,
        dynamic_formation=False,
        inject_failures=True,
    )

    print("Generated files:", generated_files)
    print("Metadata:", metadata_path)

    # Move dataset to your folder
    source_dir = Path(metadata_path).parent
    target_dir = Path("residual_correction/datasets")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\nMoving dataset files to:", target_dir)

    for file_name in generated_files.values():
        src = source_dir / file_name
        dst = target_dir / file_name
        shutil.move(str(src), str(dst))
        print(f"Moved {file_name}")

    # Move metadata file
    shutil.move(metadata_path, target_dir / Path(metadata_path).name)

    print("\nAll files moved successfully.")

    print("\nAll files moved successfully.")
    """
     dataset_name="residual_test",
        dataset_type="mixed_formations",
        task_type="residual_correction",
        num_episodes=10,
        num_obstacles=3,
        obstacle_radius=1.5,
        include_formation_in_state=True,
        communication_radius=10.0,
        """