import torch
from safetensors.torch import load_file
import io


def load_obj(path):

    data = load_file(path)
    payload = data["payload"]

    buffer = io.BytesIO(payload.cpu().numpy().tobytes())

    return torch.load(buffer, weights_only=False)


def check_dataset(path):

    obj = load_obj(path)

    print("\n=== TYPE ===")
    print(type(obj))

    data = obj["data"]

    print("\n=== AVAILABLE KEYS ===")
    print(data.keys())

    if not hasattr(data, "target"):
        print("No target found ❌")
        return

    target = data.target

    print("\n=== RESIDUAL STATS ===")

    norm = torch.norm(target, dim=-1)

    print("Mean:", norm.mean().item())
    print("Max:", norm.max().item())
    print("Min:", norm.min().item())
    print("Std:", norm.std().item())

    print("\n=== SPARSITY ===")
    print("<0.01 ratio:", (norm < 0.01).float().mean().item())
    print("<0.05 ratio:", (norm < 0.05).float().mean().item())


if __name__ == "__main__":

    check_dataset("datasets/combined_train.safetensors")