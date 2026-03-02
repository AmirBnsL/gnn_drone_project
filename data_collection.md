# Data Collection for GNN Imitation Learning

This repository contains a customized dataset generator utilizing the PyFlyt quadcopter simulation engine.

## `data-collection/data_collection.py`

This script is responsible for simulating quadcopter flights (via PyBullet physics) utilizing PyFlyt's cascaded PID Controller (Mode 7). The script orchestrates multiple drone formations and records their precise kinematics, creating a robust dataset suitable for training a Graph Neural Network (GNN) to clone the PID controller's behavior.

### Features
1. **Dynamic Scaling:** Supports generating datasets with a randomized swarm sizes (e.g., 10-20 drones).
2. **Local Frame Representation:** Features and targets are mapped to the drone's Local Body Frame (ENU conventions) making the learning task translationally invariant.
3. **Realistic Noise Modules:** Parameters exist to inject Gaussian sensor noise or environmental wind drafts directly into the simulation.
4. **Ad-Hoc Network Edges:** Computes inter-drone graph edges strictly using Euclidean distances, mimicking communication thresholds in real drone swarm topologies.

### Usage
```bash
python data-collection/data_collection.py
```
Outputs are serialized directly into the `datasets/` folder as `.npz` files encompassing all observed state nodes and topology edges.
