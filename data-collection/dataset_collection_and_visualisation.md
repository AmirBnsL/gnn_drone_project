# Dataset Collection and Visualisation

This document provides a comprehensive reference for the primary functions used to generate and visualize PyTorch Geometric (PyG) datasets for the GNN Drone Swarm project.

## 1. Dataset Generation: `generate_dataset()`

The `generate_dataset` function acts as the core PyFlyt / PyBullet simulator loop. It orchestrates episodes, computes drone setpoints, applies sensor physics, builds the graph representations, applies early stopping/tapered sampling, and saves the partitioned PyG `.pt` splits alongside metadata.

### Core Configuration
* **`dataset_name`** (`str`, default: `"formation_dataset"`): Prefix for the generated `.pt` and `.json` artifacts.
* **`dataset_type`** (`str`, default: `"mixed_formations"`): Dictates the movement task. Options include specific formation strings (`"a"`, `"rectangle"`, `"triangle"`), `"mixed_formations"` for randomly alternating targets per episode, `"hovering"`, or `"aggressive"` for random high-speed individual setpoints. This allows testing whether a single generalized model outperforms specialized models trained on isolated tasks.
* **`task_type`** (`Literal`, default: `"setpoint_prediction"`): Defines what the GNN is expected to output.
  * `"setpoint_prediction"`: Predicts ego-centric target coordinates mapping towards the slot.
  * `"residual_correction"`: Predicts the displacement distance from a naive initial slot assignment.
  * `"formation_assignment_homo"`: Returns a discrete optimal assignment class ID (homogeneous setup). The GNN learns a single global assignment pattern (e.g., drone 1 goes to slot A, etc.). This setup is simpler but lacks flexibility for varying swarm sizes.
  * `"formation_assignment_hetero"`: Outputs edge alignments bridging `drone` nodes to `slot` nodes dynamically. By constructing a bipartite graph, the GNN outputs a matching edge set rather than fixed node-level predictions, allowing for complex, size-invariant slot assignment.
* **`num_episodes`** (`int`, default: `200`): Total number of simulated episodes. More episodes yield richer data at the cost of simulation time.
* **`max_steps`** (`int`, default: `300`): Upper timeout limit for simulation ticks per episode. Allows learning complex maneuvers, though early stopping mitigates unnecessary simulation time.

### Environment & Simulation Complexity
* **`num_obstacles`** (`int`, default: `0`): Number of physical cylindrical obstacles added to the environment. Forces the swarm to learn collision avoidance and dynamic path planning through pseudo-LiDAR raycasts.
* **`obstacle_radius`** (`float`, default: `1.0`): Radius of the obstacles. Larger radii create larger no-fly zones, altering optimal paths and increasing the difficulty of the maneuver.
* **`inject_failures`** (`bool`, default: `False`): If `True`, mid-flight, a random drone is artificially disabled (motors set to 0), forcing the rest of the swarm to mathematically reassign slots and dynamically shrink the formation size $N-1$.
* **`dynamic_formation`** (`bool`, default: `False`): If `True`, midway through the episode, the target formation shape is instantly changed (e.g., from "rectangle" to "triangle"), forcing continuous in-flight re-matching.
* **`environmental_wind`** (`bool`, default: `False`): Introduces a sinusoidal updraft and turbulence field affecting physics.
* **`noisy_sensors`** (`bool`, default: `False`): Introduces Gaussian noise directly to kinematic state vectors before passing them to the GNN nodes.
* **`noise_variance`** (`float`, default: `0.01`): The variance ($\sigma^2$) scale for Gaussian state noise if enabled.
* **`graphical`** (`bool`, default: `False`): If `True`, forces PyBullet to render the 3D GUI window while simulating natively.

### Graph & Swarm parameters
* **`communication_radius`** (`float`, default: `np.inf`): The spherical cut-off distance (in meters) where interconnected drones form graph edges (`communicates` edge_index). A finite radius forces the learning of decentralized coordination and local interactions.
* **`include_formation_in_state`** (`bool`, default: `True`): When `True`, concatenates a one-hot vector representing the target formation (e.g. $[1, 0, 0]$ for "a") to each node's feature space, enabling the model to explicitly condition its behavior on the shape.
* **`mixed_formation_types`** (`tuple`): The set of allowable shapes to sample from when `dataset_type="mixed_formations"`.

### Space Initialization & Splits
* **`seed`** (`int`, default: `12345`): Global deterministic seed. Train/Val/Test use independent offsets internally to avoid overlap.
* **`split_ratios`** (`tuple[float, float, float]`, default: `(0.8, 0.1, 0.1)`): Proportional division of `num_episodes` into Train, Validation, and Test limits.
* **`base_xy_limit`** (`float`, default: `10.0`): The boundaries from which initial positions are uniformly sampled.
* **`altitude_range`** (`tuple[float, float]`, default: `(0.5, 5.0)`): The min and max randomized starting heights.
* **`validation_spread_scale`** (`float`, default: `1.25`) & **`test_spread_scale`** (`float`, default: `1.5`): Out-of-distribution (OOD) geometry scales. Increases the sampling limit multiplier for testing generalization limits.

### Temporal Tapering & Early Stopping
* **`tapered_sampling`** (`bool`, default: `True`): Dynamically downsamples the saved steps later into an episode when the drone is merely stabilizing, retaining dense frames earlier during aggressive maneuvers.
* **`dense_sampling_steps`**, **`mid_sampling_steps`**, **`mid_step_stride`**, **`late_step_stride`**: Integers setting the stride cadence cutoffs for temporal tapering. 
* **`conv_stopping`** (`bool`, default: `True`): Activates early stopping. Kills the episode loop immediately once the swarm perfectly reaches the targets.
* **`conv_threshold`** (`float`, default: `0.2`): The maximum absolute positional error (meters) allowed across ALL drones before the swarm is globally flagged as "converged".

---

## 2. Interactive Visualization: `visualize_episode_timelapse()`

Found in `visualization/dataset_visualizer.py`, this utility parses a generated PyG dataset `.pt` file and graphically replays an episode step-by-step using Plotly's interactive GUI widget slider.

### Parameters
* **`dataset`** (`InMemoryDataset` | `List[Data]`): The collated PyTorch Geometric dataset variable or list of parsed graphs loaded using PyG wrappers. Graphs must have spatial attributes (e.g., `graph.pos` or `graph['drone'].pos`) exported from the collection generator.
* **`episode_id`** (`int`): The integer ID belonging to the episode sequence you wish to view. Scans the subset to concatenate traces in temporal order `step_idx`.
* **`title`** (`str`, default: `"Drone Swarm Timelapse"`): Header title attached to the Plotly figure rendering.
* **`view_2d`** (`bool`, default: `True`): Toggles the underlying Plotly renderer layout.
  * `True`: Extracts and natively plots a top-down projection against the flat Cartesian plane. Axes limit out to $(X \leftrightarrow Y)$ mapping to visualize formation assignments across the ground.
  * `False`: Engages `go.Scatter3d` orbit visualization factoring in $(X, Y, Z)$ altitude positions inside a controllable volumetric container.
