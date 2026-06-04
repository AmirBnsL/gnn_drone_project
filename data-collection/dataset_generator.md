# `dataset_generator` Package Documentation

## Folder structure
```
 dataset_generator/
 ├─ __init__.py                # Package initialisation
 ├─ constants.py               # Global constants (formation names, split names, task types, etc.)
 ├─ environment.py             # Environment utilities (wind, obstacles, etc.)
 ├─ features.py                # Feature extraction helpers for graph construction
 ├─ formations.py              # Helpers to resolve formation names and generate mixed formations
 ├─ parallel.py                # **Core parallel dataset generation logic**
 ├─ simulation.py              # Low‑level physics episode simulation
 ├─ utils.py                   # Miscellaneous utilities (sharding, metadata handling, split calculations)
```

* **`constants.py`** – defines `FORMATION_NAMES`, `SPLIT_NAMES`, `TASK_TYPES`, and seed offsets used throughout the generator.
* **`environment.py`** – functions that create wind fields, obstacle layouts and other environment parameters.
* **`features.py`** – converts raw simulation state into graph‑structured features for GNN consumption.
* **`formations.py`** – maps a formation identifier to a concrete drone layout and provides utilities for mixed‑formation sampling.
* **`simulation.py`** – runs a single physics episode (`run_physics_episode`) and generates residual‑correction samples.
* **`utils.py`** – helper functions such as `compute_split_episode_counts`, `resolve_split_spread_scale`, `save_dataset_shard`, and `write_dataset_metadata`.
* **`parallel.py`** – orchestrates the creation of large‑scale datasets using multiple processes. The most important public entry point is `generate_dataset_parallel`.

---

## `generate_dataset_parallel`

```python
generate_dataset_parallel(
    worker_batch_size: int = 32,
    num_workers: int = 4,
    num_episodes: int = 50,
    max_steps: int = 500,
    dataset_name: str = "formation_dataset",
    dataset_type: str = "mixed_formations",
    task_type: Literal[
        "setpoint_prediction",
        "residual_correction",
        "formation_assignment_homo",
        "formation_assignment_hetero",
    ] = "setpoint_prediction",
    num_obstacles: int | tuple[int, int] = 0,
    obstacle_radius: float = 1.0,
    obstacle_radius_range: tuple[float, float] | None = None,
    residual_balance_ratio: float = 0.5,
    residual_dropout_threshold: float = 0.1,
    residual_samples_per_seed: int = 10,
    inject_failures: bool = False,
    dynamic_formation: bool = False,
    noisy_sensors: bool = False,
    noise_variance: float = 0.01,
    environmental_wind: bool = False,
    communication_radius: float = np.inf,
    include_formation_in_state: bool = True,
    mixed_formation_types: tuple = FORMATION_NAMES,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 12345,
    base_xy_limit: float = 10.0,
    altitude_range: tuple[float, float] = (0.5, 5.0),
    validation_spread_scale: float = 1.25,
    test_spread_scale: float = 1.5,
    tapered_sampling: bool = True,
    dense_sampling_steps: int = 120,
    mid_sampling_steps: int = 240,
    mid_step_stride: int = 2,
    late_step_stride: int = 5,
    conv_stopping: bool = True,
    conv_threshold: float = 0.2,
    apf_enabled: bool = True,
    apf_attractive_gain: float = 0.8,
    apf_repulsive_gain: float = 1.2,
    apf_repulsion_padding: float = 2.5,
    apf_max_step_size: float = 0.35,
    apf_vertical_gain: float = 0.5,
) -> tuple[dict[str, list[str]], str]
```

### Description
Creates a large synthetic dataset for GNN‑based drone formation control. The function distributes episode generation across multiple processes, writes intermediate shards to disk, and finally assembles a metadata file describing the dataset.

### Parameters
| Name | Type | Description |
|------|------|-------------|
| **worker_batch_size** | `int` | Number of episodes each worker processes per batch. Controls chunk size for the process pool. |
| **num_workers** | `int` | How many parallel processes to spawn. |
| **num_episodes** | `int` | Total number of episodes to generate (will be split into train/val/test according to `split_ratios`). |
| **max_steps** | `int` | Maximum simulation steps per episode (used by the physics engine). |
| **dataset_name** | `str` | Base name for the generated dataset files. |
| **dataset_type** | `str` | Identifier of the formation type (e.g., `mixed_formations`, `line`, `v_shape`). Determines how formations are sampled. |
| **task_type** | `Literal["setpoint_prediction", "residual_correction", "formation_assignment_homo", "formation_assignment_hetero"]` | The learning task the dataset supports. |
| **num_obstacles** | `int` or `tuple[int, int]` | Fixed number of obstacles or a range `(min, max)` sampled per episode. |
| **obstacle_radius** | `float` | Radius of each spherical obstacle when `obstacle_radius_range` is not provided. |
| **obstacle_radius_range** | `tuple[float, float] | None` | Optional range to sample obstacle radii from. |
| **residual_balance_ratio** | `float` | Desired proportion of “near‑zero” residual samples when `task_type="residual_correction"`. |
| **residual_dropout_threshold** | `float` | Threshold below which a residual sample is considered “near‑zero”. |
| **residual_samples_per_seed** | `int` | Number of residual‑correction samples generated for each random seed. |
| **inject_failures** | `bool` | If `True`, randomly injects failure conditions (e.g., sensor drop‑out). |
| **dynamic_formation** | `bool` | Enables time‑varying formation changes during an episode. |
| **noisy_sensors** | `bool` | Adds Gaussian noise to sensor readings. |
| **noise_variance** | `float` | Variance of the Gaussian noise when `noisy_sensors=True`. |
| **environmental_wind** | `bool` | Simulates wind disturbances if enabled. |
| **communication_radius** | `float` | Maximum distance for inter‑drone communication; `np.inf` disables the limit. |
| **include_formation_in_state** | `bool` | Whether the formation identifier is part of the observation vector. |
| **mixed_formation_types** | `tuple` | Collection of formation names used when `dataset_type` is a mixed‑formation setting. |
| **split_ratios** | `tuple[float, float, float]` | Proportions for train / validation / test splits (must sum to 1). |
| **seed** | `int` | Global random seed for reproducibility. |
| **base_xy_limit** | `float` | Base XY area limit for initial drone placement; scaled per split. |
| **altitude_range** | `tuple[float, float]` | Minimum and maximum altitude for drones. |
| **validation_spread_scale** | `float` | Scale factor applied to `base_xy_limit` for the validation split. |
| **test_spread_scale** | `float` | Scale factor applied to `base_xy_limit` for the test split. |
| **tapered_sampling** | `bool` | If `True`, uses a tapered step‑sampling scheme (dense early, sparse later). |
| **dense_sampling_steps** | `int` | Number of steps sampled densely at the start of an episode. |
| **mid_sampling_steps** | `int` | Number of steps sampled with a medium stride. |
| **mid_step_stride** | `int` | Stride for the mid‑sampling region. |
| **late_step_stride** | `int` | Stride for the late‑sampling region. |
| **conv_stopping** | `bool` | Enables early stopping when the convergence metric falls below `conv_threshold`. |
| **conv_threshold** | `float` | Threshold for the convergence metric used with `conv_stopping`. |
| **apf_enabled** | `bool` | Turns on the Artificial Potential Field controller for collision avoidance. |
| **apf_attractive_gain** | `float` | Gain for the attractive component of the APF. |
| **apf_repulsive_gain** | `float` | Gain for the repulsive component of the APF. |
| **apf_repulsion_padding** | `float` | Extra padding added around obstacles for repulsion. |
| **apf_max_step_size** | `float` | Maximum step size the APF can command. |
| **apf_vertical_gain** | `float` | Scaling factor for vertical motion in the APF. |

### Returns
* **generated_files** – `dict[str, list[str]]` mapping each split name (`"train"`, `"val"`, `"test"`) to a list of shard filenames that were written to the `datasets/` directory.
* **metadata_path** – `str` path to the JSON metadata file (`*_metadata.json`) that records the full configuration and per‑episode statistics.

---

## Usage example
```python
from dataset_generator.parallel import generate_dataset_parallel

files, meta_path = generate_dataset_parallel(
    num_workers=8,
    num_episodes=2000,
    max_steps=600,
    dataset_type="v_shape",
    task_type="setpoint_prediction",
    split_ratios=(0.7, 0.15, 0.15),
    seed=42,
)
print("Shards written:", files)
print("Metadata saved at:", meta_path)
```

The function will create a temporary directory under `datasets/`, write shard files (`*.safetensors`) for each split, and finally move them into the permanent `datasets/` folder alongside the generated metadata JSON.

---

## Extending the generator
* To add a new **task type**, extend `TASK_TYPES` in `constants.py` and implement the corresponding logic inside `simulate_worker_chunk`.
* To support a new **formation**, add its name to `FORMATION_NAMES` and implement the geometry in `formations.py`.
* Custom post‑processing of generated graphs can be added in `utils.save_dataset_shard`.

---

*This documentation lives in `data-collection/dataset_generator.md` and is intended for developers who want to understand or extend the synthetic dataset pipeline.*