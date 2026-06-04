import torch
from inference.feature_extractor import extract_node_features
from inference.graph_builder import build_graph
from inference.inference_engine import InferenceEngine
from models import build_model
import sys

sys.path.append("..")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "checkpoints/best_nnconv.pt"
MODEL_NAME = "nnconv"

# ----------------------------
# LOAD MODEL
# ----------------------------

model = build_model(MODEL_NAME, in_dim=73)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

engine = InferenceEngine(model, DEVICE)

print("Model loaded")

# ----------------------------
# INFERENCE STEP
# ----------------------------

def run_step(
    env,
    formation_one_hot,
    noisy_sensors,
    obstacles,
    obstacle_radii
):

    node_features = []
    positions = []

    # -----------------------------------
    # EXTRACT FEATURES FROM EACH DRONE
    # -----------------------------------

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
    print("num drones:", len(node_features))
    print("feature dim:", node_features[0].shape)
    # -----------------------------------
    # BUILD GRAPH
    # -----------------------------------

    graph = build_graph(
        node_features,
        positions
    )

    # -----------------------------------
    # NNCONV PREDICTION
    # -----------------------------------

    corrections = engine.predict(graph).numpy()

    # shape:
    # [num_drones, 3]
    # each row = [dx, dy, dz]

    return corrections