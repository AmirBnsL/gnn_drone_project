import torch
import numpy as np

from inference.feature_extractor import extract_node_features
from inference.graph_builder import build_graph
from inference.inference_engine import InferenceEngine
from models import build_model

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best_nnconv.pt"
MODEL_NAME = "nnconv"

# ----------------------------
# LOAD MODEL (MATCH TRAINING)
# ----------------------------
sample_in_dim = 12  # MUST match your dataset (adjust if needed)
model = build_model(MODEL_NAME, in_dim=sample_in_dim)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

engine = InferenceEngine(model, DEVICE)

print("Model loaded successfully")

# ----------------------------
# SIMULATION LOOP (EXAMPLE)
# ----------------------------
def run_step(env, formation_one_hot, noisy_sensors, obstacles, obstacle_radii):

    node_features = []
    positions = []

    for drone in env.drones:

        feat, pos = extract_node_features(
            drone,
            obstacles,
            obstacle_radii,
            noisy_sensors=noisy_sensors,
            include_formation=True,
            formation_one_hot=formation_one_hot,
            physics_client=env._client
        )

        node_features.append(feat)
        positions.append(pos)

    graph = build_graph(node_features, positions, communication_radius=4.0)

    # ----------------------------
    # INFERENCE (NO TRAINING)
    # ----------------------------
    corrections = engine.predict(graph).numpy()

    # ----------------------------
    # APPLY CONTROL
    # ----------------------------
    for i, drone in enumerate(env.drones):
        drone.apply_position_update(corrections[i])

    return corrections