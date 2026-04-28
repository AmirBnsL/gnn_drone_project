FORMATION_NAMES = ("a", "rectangle", "triangle")
FORMATION_TO_ID = {name: idx for idx, name in enumerate(FORMATION_NAMES)}
SPLIT_NAMES = ("train", "val", "test")
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}
TASK_TYPES = (
    "setpoint_prediction",
    "residual_correction",
    "formation_assignment_homo",
    "formation_assignment_hetero",
)
