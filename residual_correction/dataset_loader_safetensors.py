import io
import torch
from safetensors.torch import load_file
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def load_obj(path):
    data = load_file(path)
    payload = data["payload"]

    byte_data = payload.cpu().numpy().tobytes()
    buffer = io.BytesIO(byte_data)

    return torch.load(buffer, weights_only=False)


class SimpleDataset:
    def __init__(self, obj):
        self.data = obj["data"]
        self.slices = obj["slices"]

    def __len__(self):
        return self.slices["x"].size(0) - 1

    
    def get(self, idx):

        data = Data()

        for key in self.slices.keys():

            value = getattr(self.data, key)
            ptr = self.slices[key]

            start = ptr[idx]
            end = ptr[idx + 1]

        # SPECIAL FIX FOR EDGE_INDEX
            if key == "edge_index":
                edge = value[:, start:end]   # IMPORTANT FIX
                setattr(data, key, edge)
            else:
                setattr(data, key, value[start:end])

        return data
    def __getitem__(self, idx):
        return self.get(idx)


def get_dataloaders(dataset_dir, batch_size=32):

    train = SimpleDataset(load_obj(f"{dataset_dir}/combined_train.safetensors"))
    val = SimpleDataset(load_obj(f"{dataset_dir}/combined_val.safetensors"))
    test = SimpleDataset(load_obj(f"{dataset_dir}/combined_test.safetensors"))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, _, _ = get_dataloaders("datasets")

    batch = next(iter(train_loader))

    print(batch)
    print("x:", batch.x.shape)
    print("edge_index:", batch.edge_index.shape)
    print("target:", batch.target.shape)