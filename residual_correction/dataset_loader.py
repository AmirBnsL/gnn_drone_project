from __future__ import annotations

from pathlib import Path
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader


class ResidualCorrectionDataset(InMemoryDataset):
    def __init__(self, root: str | Path, split: str):
        self.root_dir = Path(root)
        self.split = split

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}'. Must be one of: train, val, test")

        super().__init__(root=str(self.root_dir))
        self.data, self.slices = self._load_processed_file()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def _find_split_file(self) -> Path:
        matches = list(self.root_dir.glob(f"*_{self.split}.pt"))
        if not matches:
            raise FileNotFoundError(
                f"No '*_{self.split}.pt' file found in {self.root_dir}"
            )
        if len(matches) > 1:
            print(f"[Warning] Multiple {self.split} files found. Using: {matches[0].name}")
        return matches[0]

    def _load_processed_file(self):
        split_file = self._find_split_file()
        loaded = torch.load(split_file, weights_only=False)

        if not isinstance(loaded, dict):
            raise TypeError(
                f"Expected loaded dataset to be dict, got {type(loaded)}"
            )

        if "data" not in loaded or "slices" not in loaded:
            raise KeyError(
                f"Loaded file missing required keys. Found keys: {list(loaded.keys())}"
            )

        data = loaded["data"]
        slices = loaded["slices"]

        if not isinstance(data, Data):
            raise TypeError(f"Expected 'data' to be PyG Data, got {type(data)}")
        if not isinstance(slices, dict):
            raise TypeError(f"Expected 'slices' to be dict, got {type(slices)}")

        return data, slices


def get_datasets(dataset_dir: str | Path):
    dataset_dir = Path(dataset_dir)

    train_dataset = ResidualCorrectionDataset(dataset_dir, split="train")
    val_dataset = ResidualCorrectionDataset(dataset_dir, split="val")
    test_dataset = ResidualCorrectionDataset(dataset_dir, split="test")

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    dataset_dir: str | Path,
    batch_size: int = 16,
    shuffle_train: bool = True,
):
    train_dataset, val_dataset, test_dataset = get_datasets(dataset_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset_dir = Path("residual_correction/datasets")

    train_dataset, val_dataset, test_dataset = get_datasets(dataset_dir)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    sample = train_dataset[0]
    print("\nFirst train sample:")
    print(sample)
    print("x shape:", tuple(sample.x.shape))
    print("target shape:", tuple(sample.target.shape))
    print("edge_index shape:", tuple(sample.edge_index.shape))
    print("edge_attr shape:", tuple(sample.edge_attr.shape))