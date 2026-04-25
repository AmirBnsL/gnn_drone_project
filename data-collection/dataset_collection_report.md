# Dataset Collection Report

## Overview

This report summarizes the dataset collection improvements made for the drone swarm GNN project.

The dataset generation pipeline was enhanced to better support realistic swarm tasks, especially for the following dataset types:
- `setpoint_prediction`
- `residual_correction`
- `formation_assignment_homo`
- `formation_assignment_hetero`

The main improvements include parallelized generation, programmatic LiDAR sensing, improved handling of residual correction imbalance, randomized obstacle count, and convergence-based early stopping.

## Key Improvements

### 1. Parallelized Dataset Generation

The dataset generation script now supports parallel execution through a `num_workers` parameter.
- Multiple episodes are simulated concurrently using `ProcessPoolExecutor`.
- This greatly reduces generation time for large training datasets.
- The parallel approach saves intermediate episode graphs to temporary files and aggregates them after all workers complete.

### 2. Programmatic LiDAR Sensor Features

LiDAR-style sensor data was added programmatically to the drone feature vector.
- Each drone computes simulated 2D raycast readings around itself.
- The LiDAR observations are included in the node feature vector used for GNN training.
- This gives the model a local obstacle awareness signal without requiring real sensor hardware.

### 3. Residual Balance Ratio for `residual_correction`

To handle class imbalance in the `residual_correction` dataset type, a residual balance mechanism was introduced.
- The `residual_balance_ratio` parameter controls how often near-zero residual examples are kept.
- A threshold is used to identify naively correct steps that do not require residual correction.
- When the residual is below this threshold, the step may be discarded to avoid overwhelming the dataset with trivial examples.

### 4. Randomized Obstacle Count per Episode

The number of obstacles is now randomized each episode using a tuple `(min, max)`.
- This allows generation of richer and more varied scenes.
- It enables datasets to capture scenarios with zero obstacles and scenarios with multiple obstacles.
- The `num_obstacles` configuration accepts either a fixed integer or a range.

### 5. Convergence Stopping to Reduce Meaningless Steps

A convergence stopping mechanism was added to stop episodes early when drones are sufficiently close to their formation slots.
- This reduces the number of uninformative simulation steps.
- It makes dataset generation faster and focuses data on meaningful maneuvers.
- The stopping condition uses a distance threshold to detect when the formation has converged.

## From Amir Benslaimi

This report was written by Amir Benslaimi.

## Future Goals

- Find a better way to implement sensors for obstacle detection.
- Develop a better obstacle avoidance algorithm.
- Search for other simulators with better sensing features that remain computationally feasible on low-end PCs.
- Aim to emulate more realistic environments and sensor feedback.
