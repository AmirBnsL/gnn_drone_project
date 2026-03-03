# `data_collection.py` — Detailed Walkthrough

This guide explains each part of your dataset generator and the NumPy functions used in each section.

---

## 1) Imports and Purpose

Your script imports:
- `os` for filesystem operations.
- `numpy as np` for numerical arrays and random sampling.
- `pybullet as p` for rotation conversions.
- `Aviary` from `PyFlyt.core` for drone simulation.

The goal is to generate supervised learning data for a GNN:
- **Inputs**: drone state + target error (in local body frame)
- **Labels**: PID motor outputs (`drone.pwm`)
- **Graph**: edges from communication-distance constraints

---

## 2) Wind Model: `wind_generator(...)`

This creates a time-varying wind field.

### NumPy used
- `np.zeros_like(position)`
  - Creates a zero array with the same shape as `position`.
- `np.sin(time)`, `np.cos(time / 2.0)`
  - Smooth periodic wind variation over time.
- `np.random.normal(0, 0.2, size=(len(position),))`
  - Adds Gaussian turbulence independently per drone.

Result: wind vector per drone in 3D (`x, y, z`).

---

## 3) Episode Initialization: `sample_episode_initial_conditions(...)`

Generates random initial positions and yaw for all drones in one episode.

### NumPy used
- `np.random.uniform(-10.0, 10.0, size=(num_drones, 3))`
  - Uniform random spawn positions in a 3D box.
- `start_pos[:, 2] = np.random.uniform(0.5, 5.0, size=(num_drones,))`
  - Overwrites z-values to keep drones above ground.
- `np.zeros((num_drones, 3))`
  - Initializes roll/pitch/yaw to zero.
- `start_orn[:, 2] = np.random.uniform(-np.pi, np.pi, size=(num_drones,))`
  - Random yaw in full circle range.

---

## 4) Environment Setup: `create_aviary(...)`

Creates the PyFlyt environment and configures control mode.

No direct NumPy computation here.

---

## 5) Goal Generation: `build_setpoints(...)`

Creates target setpoints `[x, y, yaw, z]` per drone.

### NumPy used
- `np.zeros((num_drones, 4))`
  - Preallocates setpoint matrix.
- Slicing, e.g. `setpoints[:, :2]`, `setpoints[:, 2]`, `setpoints[:, 3]`
  - Efficient column-wise assignment.
- `np.random.uniform(...)`
  - Random target offsets and target yaw/z values.

For `hovering`, targets are current pose; otherwise random/aggressive offsets.

---

## 6) Sensor Noise Injection: `maybe_add_sensor_noise(...)`

Optionally perturbs measurements to improve robustness.

### NumPy used
- `np.random.normal(0, noise_variance, size=3)`
  - 3D Gaussian noise for each vector.
- Vector addition, e.g. `global_pos + noise`
  - Elementwise perturbation.

Returns unchanged vectors when `noisy_sensors=False`.

---

## 7) Feature Construction per Drone: `build_drone_features(...)`

Converts raw drone state into model-ready input/target/label.

### NumPy used
- `np.array(state[k], copy=True)`
  - Copies data to avoid accidental in-place mutation of simulator buffers.
- `np.array([setpoint[0], setpoint[1], setpoint[3]])`
  - Builds 3D target position vector from setpoint format.
- `global_pos_error = target_global_pos - global_pos`
  - Vector subtraction.
- `yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi`
  - Wraps angle into `[-pi, pi]`.
- `np.array(p.getMatrixFromQuaternion(...)).reshape(3, 3)`
  - Converts flat 9-value rotation matrix into 3x3 matrix.
- `rot_matrix.T @ global_pos_error`
  - Matrix-vector multiplication to transform global error into body frame.
- `np.concatenate([local_lin_vel, local_ang_vel])`
  - Creates state feature vector (6 values).
- `np.concatenate([local_pos_error, np.array([yaw_error])])`
  - Creates target vector (4 values).

Outputs:
- `gnn_input_state` (6,)
- `gnn_input_target` (4,)
- `motor_pwm_labels` (4,)
- `global_pos` (for graph edge computation)

---

## 8) Graph Topology: `build_edges(...)`

Builds directed edges for drone communication graph.

### NumPy used
- `np.linalg.norm(global_positions[i] - global_positions[j])`
  - Euclidean distance between two drones.

If distance <= `communication_radius`, edge `[i, j]` is added.

---

## 9) One Simulation Step Collection: `collect_step_data(...)`

Collects all drones’ features and computes edges for a single timestep.

### NumPy used
- `np.array(global_positions)`
  - Converts list of 3D vectors to 2D numeric array before distance computation.

Returns per-step lists:
- `episode_states`, `episode_targets`, `episode_labels`, `edges`

---

## 10) Disk Save: `save_dataset(...)`

Stores all timesteps into `.npz` file.

### NumPy used
- `np.array(all_states, dtype=object)` (same for targets/labels/edges)
  - Uses `dtype=object` because number of drones can vary by episode.
- `np.savez(...)`
  - Saves multiple named arrays into a single compressed-like NumPy archive.

Saved keys:
- `states`
- `targets`
- `labels`
- `edges`

---

## 11) Main Orchestrator: `generate_dataset(...)`

High-level loop:
1. Create output folder with `os.makedirs("datasets", exist_ok=True)`.
2. For each episode:
   - Sample drone count and initial states.
   - Build environment and setpoints.
   - For each step up to `max_steps`:
     - Collect one graph sample.
     - Append to global lists.
     - Advance physics via `env.step()`.
3. Disconnect environment.
4. Save all data.

### NumPy behavior across loops
- List accumulation during simulation (faster append pattern).
- Final conversion to NumPy object arrays before `np.savez`.

---

## 12) Dataset Size Math

If one run uses:
- `num_episodes = E`
- `max_steps = S`

Then timestep samples approximately equal:

`total_samples = E * S`

Example from your script: `20 * 400 = 8000` samples per dataset file.

---

## 13) Quick NumPy Reference (from your script)

- `np.zeros_like(a)`: zeros with same shape as `a`
- `np.zeros(shape)`: zeros with explicit shape
- `np.random.uniform(low, high, size=...)`: uniform random numbers
- `np.random.normal(mean, std, size=...)`: Gaussian random numbers
- `np.sin`, `np.cos`: trigonometric functions
- `np.pi`: constant π
- `np.array(obj, copy=True)`: explicit array copy
- `arr.reshape(r, c)`: change shape without changing data order
- `arr.T`: transpose
- `A @ x`: matrix multiplication
- `np.concatenate([...])`: join vectors end-to-end
- `np.linalg.norm(v)`: vector magnitude (Euclidean norm)
- `np.savez(path, key=array, ...)`: save multiple arrays to `.npz`

---

## 14) Why This Structure Is Good for GNN Training

- Local-frame errors (`rot_matrix.T @ error`) improve invariance.
- Distance-based edges produce realistic sparse communication graphs.
- Optional noise/wind improves robustness and helps reduce covariate shift.
- Labels from `drone.pwm` provide direct imitation target for policy learning.

---

## 15) Physics Terminology Used in This Script

This section explains the flight/robotics terms that appear in your pipeline.

### Reference Frames

- **Global / World Frame**
  - Fixed frame of the simulator (shared by all drones).
  - Positions like `global_pos` and setpoints `[x, y, z]` are first defined here.

- **Body / Local Frame**
  - Frame attached to each drone (moves and rotates with the drone).
  - Your script converts position error from world frame into body frame using:
    - `rot_matrix.T @ global_pos_error`
  - This makes learning more invariant to where/how the drone is oriented globally.

### Attitude and Orientation

- **Roll, Pitch, Yaw (Euler angles)**
  - Orientation decomposition of the drone.
  - In this script, yaw is especially important for heading control.

- **Yaw Error**
  - Difference between target yaw and current yaw.
  - Wrapped to `[-pi, pi]` so equivalent angles are represented consistently:
    - `yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi`

- **Quaternion / Rotation Matrix**
  - Quaternion is a robust orientation representation.
  - PyBullet provides rotation matrix from quaternion.
  - Rotation matrix maps vectors between coordinate frames.

### Kinematics Terms

- **Position (`global_pos`)**
  - Where the drone is in 3D space.

- **Linear Velocity (`local_lin_vel`)**
  - Translational speed components in local/body frame.

- **Angular Velocity (`local_ang_vel`)**
  - Rotational rates around body axes.

- **Position Error (`global_pos_error`, `local_pos_error`)**
  - Vector from current position to target position.
  - Local-frame error is used as a GNN target input feature.

### Dynamics and Control

- **PID Controller (Cascaded)**
  - Inner loops and outer loops regulate position and attitude.
  - In your script, simulator mode 7 uses the built-in cascaded position control behavior.

- **Setpoint**
  - Desired command state `[x, y, yaw, z]` provided to the controller.
  - Controller generates motor outputs to track this target.

- **PWM Labels (`drone.pwm`)**
  - Motor command outputs (your supervised learning target).
  - GNN is trained to imitate this expert output.

### Simulation Timing

- **Control Step / Timestep**
  - One call to `env.step()` advances the simulation by one control tick.
  - Each tick stores one training graph sample.

- **Episode**
  - A single rollout with fixed initial conditions and setpoint assignment.
  - `num_episodes` controls how many rollouts you generate.

- **`max_steps`**
  - Number of timesteps recorded per episode.
  - Total samples roughly `num_episodes * max_steps`.

### Interaction Graph Terms

- **Communication Radius**
  - Maximum distance for two drones to be connected in the graph.
  - If `distance(i, j) <= communication_radius`, edge `[i, j]` is included.

- **Directed Edge**
  - Your graph stores ordered pairs (`i -> j`).
  - This can represent asymmetric message passing in a GNN implementation.

- **Sparse Graph**
  - Most drone pairs are not connected when radius is limited.
  - Better reflects realistic local communication constraints.

### Disturbance and Robustness Terms

- **Sensor Noise**
  - Artificial Gaussian perturbation added to measured state variables.
  - Simulates imperfect onboard sensing and improves generalization.

- **Wind Field / Disturbance**
  - External force disturbance applied by `wind_generator`.
  - Introduces non-ideal flight behavior for robustness.

- **Covariate Shift (practical ML term)**
  - Model sees slightly different states at inference than training.
  - Noise + wind + varied setpoints help reduce this mismatch.
