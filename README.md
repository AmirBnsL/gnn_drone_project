# Pattern Emergence in Drone Swarms Using Graph-Based AI Models

## Overview
This repository contains a complete pipeline for generating synthetic drone formation datasets, visualising them, training Graph Neural Network (GNN) models for several tasks, and deploying the trained models via a web interface.

- **Data collection** – uses the **PyFlyt** simulator to generate realistic multi‑drone episodes.
- **Visualization** – scripts for visualising the generated datasets.
- **GNN models** – implementations for set‑point prediction, formation assignment (homogeneous & heterogeneous) and residual‑correction tasks.
- **Website** – a lightweight web app for serving the trained models.

---

## 1. Installing `uv`
`uv` is a fast, modern Python package manager and virtual‑environment tool. It can be installed with `pip`:

```bash
# Install uv globally (or in a user‑wide environment)
python -m pip install --upgrade uv
```

> **Tip**: After installation, make sure the `uv` executable is on your `PATH`. You can verify the installation with:
>
```bash
uv --version
```

---

## 2. Using `uv` to Install Project Dependencies
Once `uv` is available, you can set up the project in a clean virtual environment:

```bash
# Create a new virtual environment in the project root (named .venv by default)
uv venv

# Activate the environment (Fish shell example)
source .venv/bin/activate.fish   # bash/zsh: source .venv/bin/activate

# Install all required packages from the lockfile (or directly from pyproject.toml)
uv sync   # reads uv.lock / pyproject.toml and installs exact versions
```

If you only have a `requirements.txt` file, you can still use `uv`:

```bash
uv pip install -r requirements.txt
```

---

## 3. Data Collection (PyFlyt)
The **data‑collection** module (`data-collection/`) contains utilities that drive the **PyFlyt** simulator to generate episodes.

- `dataset_generator/` – core logic for parallel dataset generation.

All generated files are stored under the `datasets/` directory.

---

## 4. Visualization
Visualization scripts live in `visualization/`. They can plot trajectories, formation layouts.


---

## 5. GNN Models
The **gnn/** package implements three families of models:

| Task | Module | Description |
|------|--------|-------------|
| Set‑point prediction | `gnn.setpoint` | Predict the next control set‑point for each drone.
| Formation assignment | `gnn.formation_assignment` | Assign drones to positions in a target formation (homogeneous & heterogeneous variants).
| Residual correction | `gnn.residual_correction` | Learn a residual that corrects a baseline controller.


---

## 6. Deploying Models (Web Interface)
A simple three.js based web app serves the trained models for inference.


---

## 7. Quick Start Checklist
1. Install `uv` with `pip`.
2. Create and activate a virtual environment: `uv venv && source .venv/bin/activate`.
3. Install dependencies: `uv sync`.
4. Generate a dataset using the `dataset_generator` utilities.
5. Visualise the data to verify correctness.
6. Train the desired GNN model(s).
7. Launch the web app to serve the trained model.

---

## 8. Contributing
Feel free to open issues or submit pull requests. Please keep the following in mind:
- Follow the existing code style (PEP 8, type hints).
- Add or update documentation when you introduce new functionality.
- Ensure that any new script works with the `uv` environment (`uv run <script>`).

---
