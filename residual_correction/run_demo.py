import sys
from pathlib import Path
import numpy as np
import torch

# =========================================================
# PATH SETUP
# =========================================================
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_COLLECTION = PROJECT_ROOT / "data-collection"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(DATA_COLLECTION))

# =========================================================
# IMPORTS
# =========================================================
from dataset_generator.environment import create_aviary, build_setpoints
from dataset_generator.utils import sample_episode_initial_conditions
from inference.run_simulation import run_step

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_DRONES = 5
STEPS = 150
FORMATION = "triangle"   # try: "a", "rectangle", "triangle"

# =========================================================
# INIT SCENE
# =========================================================
print("🚀 Initializing simulation...")

start_pos, start_orn = sample_episode_initial_conditions(
    N_DRONES,
    np.random.default_rng(0),
    xy_limit=3.0,
    altitude_range=(1.0, 3.0)
)

obstacles = np.array([
    [0.0, 0.0, 1.5],
    [1.5, 0.0, 1.5],
    [-1.5, 0.0, 1.5]
])

obstacle_radii = np.array([1.0, 1.0, 1.0])

env = create_aviary(
    start_pos=start_pos,
    start_orn=start_orn,
    environmental_wind=0.0,
    obstacles=obstacles,
    obstacle_radii=obstacle_radii,
    obstacle_radius=1.0,
    graphical=True
)

# =========================================================
# FORMATION SETPOINTS (GROUND TRUTH TARGET)
# =========================================================
setpoints, col_ind, naive_offsets = build_setpoints(
    FORMATION,
    start_pos,
    start_orn,
    np.random.default_rng(0),
    obstacles,
    obstacle_radii,
    1.0
)

active_drones = list(range(N_DRONES))

for i in active_drones:
    env.set_setpoint(i, setpoints[i])

print(f"✅ Formation loaded: {FORMATION}")
formation_one_hot = np.zeros(3, dtype=np.float32)

if FORMATION == "a":
    formation_one_hot[0] = 1

elif FORMATION == "rectangle":
    formation_one_hot[1] = 1

elif FORMATION == "triangle":
    formation_one_hot[2] = 1
# =========================================================
# METRICS
# =========================================================
baseline_errors = []
gnn_errors = []

# =========================================================
# MAIN LOOP
# =========================================================
print("🚀 Starting demo...")

for step in range(STEPS):

    try:
        # -------------------------------------------------
        # 1. RUN GNN RESIDUAL PREDICTION
        # -------------------------------------------------
        corrections = run_step(
            env=env,
            formation_one_hot=formation_one_hot,
            noisy_sensors=False,
            obstacles=obstacles,
            obstacle_radii=obstacle_radii
        )

        corrections = np.asarray(corrections)

        # -------------------------------------------------
        # 2. APPLY CORRECTIONS (ONLY IN GNN MODE)
        # -------------------------------------------------
        for i in active_drones:

            corrected = setpoints[i].copy()

            # ✔ correct axes:
            #corrected[0] += corrections[i][0]   # x
            #corrected[1] += corrections[i][1]   # y
            #corrected[3] += corrections[i][2]   # z

            env.set_setpoint(i, corrected)

        # -------------------------------------------------
        # 3. STEP SIMULATION
        # -------------------------------------------------
        env.step()
        if step % 20 == 0:

            print("\n========================")
            print(f"STEP {step}")

            for i in active_drones:

                drone_pos = env.drones[i].state[3]

                target = setpoints[i]
                target_pos = np.array([
                    target[0],
                    target[1],
                    target[3]
                ])

                err = np.linalg.norm(drone_pos - target_pos)

                print(f"Drone {i}")
                print("  current :", np.round(drone_pos, 2))
                print("  target  :", np.round(target_pos, 2))
                print("  error   :", round(err, 3))
                print("  residual:", np.round(corrections[i], 3))

        # -------------------------------------------------
        # 4. COMPUTE ERROR METRICS
        # -------------------------------------------------
        for i in active_drones:

            drone_pos = env.drones[i].state[3]

            target = setpoints[i]
            target_pos = np.array([target[0], target[1], target[3]])

            err = np.linalg.norm(drone_pos - target_pos)
            gnn_errors.append(err)

        # -------------------------------------------------
        # DEBUG OUTPUT
        # -------------------------------------------------
        if step % 20 == 0:
            print(f"Step {step}")
            print(f"   correction sample: {corrections[0]}")
            print(f"   mean error: {np.mean(gnn_errors):.4f}")

    except Exception as e:
        print("❌ Crash at step", step)
        print(e)
        break

# =========================================================
# END
# =========================================================
env.disconnect()

print("===================================")
print("📊 FINAL RESULTS")
print("GNN mean error:", np.mean(gnn_errors))
print("===================================")