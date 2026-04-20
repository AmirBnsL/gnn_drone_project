from pathlib import Path
import json
import torch


DATASET_DIR = Path("residual_correction/datasets")
#DATASET_NAME = "residual_test_mixed_formations"
# DATASET_NAME = "residual_main_mixed_formations"
DATASET_NAME = "residual_failures_dense_mixed_formations"
SPLIT = "train"


def describe_loaded_object(obj, name="loaded_object"):
    print(f"\n=== {name} type ===")
    print(type(obj))

    if isinstance(obj, dict):
        print(f"{name} keys:", list(obj.keys()))
        for k, v in obj.items():
            print(f"  - key={k!r}, type={type(v)}")
    elif isinstance(obj, (list, tuple)):
        print(f"{name} length:", len(obj))
        if len(obj) > 0:
            print(f"first element type: {type(obj[0])}")
    else:
        print(obj)


def extract_first_graph(obj):
    # Case 1: list of graphs
    if isinstance(obj, list):
        if len(obj) == 0:
            raise ValueError("Loaded list is empty.")
        return obj[0]

    # Case 2: tuple
    if isinstance(obj, tuple):
        if len(obj) == 0:
            raise ValueError("Loaded tuple is empty.")
        return obj[0]

    # Case 3: dict with common keys
    if isinstance(obj, dict):
        for key in ["graphs", "data", "items", "samples"]:
            if key in obj:
                value = obj[key]
                if isinstance(value, list) and len(value) > 0:
                    return value[0]

        # If values themselves are graph-like, take first value
        values = list(obj.values())
        if len(values) == 0:
            raise ValueError("Loaded dict is empty.")

        first_value = values[0]

        # Sometimes split dicts look like {episode_id: Data(...)}
        if hasattr(first_value, "x") or hasattr(first_value, "edge_index"):
            return first_value

        # Sometimes dict values are lists of Data objects
        if isinstance(first_value, list) and len(first_value) > 0:
            return first_value[0]

    raise TypeError("Could not automatically extract a graph from the loaded dataset.")


def get_num_graphs(obj):
    if isinstance(obj, (list, tuple)):
        return len(obj)
    if isinstance(obj, dict):
        return len(obj)
    return "unknown"


def print_graph_info(graph):
    print("\n=== First graph summary ===")
    print(graph)

    print("\n=== Tensor shapes ===")
    if hasattr(graph, "x") and graph.x is not None:
        print("x shape:", tuple(graph.x.shape))
    else:
        print("x not found")

    if hasattr(graph, "target") and graph.target is not None:
        print("target shape:", tuple(graph.target.shape))
    else:
        print("target not found")

    if hasattr(graph, "edge_index") and graph.edge_index is not None:
        print("edge_index shape:", tuple(graph.edge_index.shape))
    else:
        print("edge_index not found")

    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        print("edge_attr shape:", tuple(graph.edge_attr.shape))
    else:
        print("edge_attr not found")

    print("\n=== Residual statistics ===")
    if hasattr(graph, "target") and graph.target is not None:
        target = graph.target.float()

        print("target min:", target.min(dim=0).values)
        print("target max:", target.max(dim=0).values)
        print("target mean:", target.mean(dim=0))

        norms = torch.norm(target, dim=1)
        print("residual norm min:", norms.min().item())
        print("residual norm max:", norms.max().item())
        print("residual norm mean:", norms.mean().item())

        non_zero = (norms > 1e-6).sum().item()
        total = norms.numel()
        print(f"non-zero residuals: {non_zero}/{total}")
    else:
        print("No target field found.")

    print("\n=== Extra fields ===")
    for attr in ["pos", "obstacles", "formation_id", "episode_id", "step_idx"]:
        if hasattr(graph, attr):
            value = getattr(graph, attr)
            if torch.is_tensor(value):
                print(f"{attr}: tensor shape {tuple(value.shape)}")
            else:
                print(f"{attr}: {value}")


def main():
    dataset_path = DATASET_DIR / f"{DATASET_NAME}_{SPLIT}.pt"
    metadata_path = DATASET_DIR / f"{DATASET_NAME}_metadata.json"

    print(f"Loading dataset from: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    loaded = torch.load(dataset_path, weights_only=False)

    print(f"\nNumber of top-level items in split '{SPLIT}': {get_num_graphs(loaded)}")

    if metadata_path.exists():
        print(f"\nLoading metadata from: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("Task type:", metadata.get("task_type"))
        print("Dataset type:", metadata.get("dataset_type"))
        print("Metadata keys:", list(metadata.keys()))
    else:
        print("\nMetadata file not found.")

    describe_loaded_object(loaded, "loaded_dataset")

    graph = extract_first_graph(loaded)
    print_graph_info(graph)


if __name__ == "__main__":
    main()