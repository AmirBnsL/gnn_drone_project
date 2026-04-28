import os
import argparse
import gc
from tqdm import tqdm
import torch
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from dataset_generator.utils import load_object_safetensors, save_dataset_shard

def extract_graphs_from_payload(payload):
    data = payload["data"]
    slices = payload["slices"]
    
    # Identify how many graphs are in this shard
    # Any key in slices will have num_graphs + 1 elements
    first_key = next(iter(slices))
    num_graphs = len(slices[first_key]) - 1
    
    graphs = []
    for i in range(num_graphs):
        if isinstance(data, HeteroData):
            graph = HeteroData()
            # Iterate over node types, edge types, etc.
            for store_name in data._container_names:
                # This is a bit complex for HeteroData, let's use the public API if possible
                pass
            # Actually, the simplest way is to use data.get_example(i) if it existed, 
            # but it doesn't in the same way.
            # Let's use a manual slice for HeteroData if needed, but for now let's try a simpler approach.
            # InMemoryDataset has a __getitem__ that does this slicing.
            
            class TempDataset(InMemoryDataset):
                def __init__(self, data, slices):
                    super().__init__(root=None)
                    self.data = data
                    self.slices = slices
                def len(self):
                    return num_graphs
            
            temp_ds = TempDataset(data, slices)
            graphs.append(temp_ds[i])
        else:
            # Standard Data object
            graph = Data()
            for key in slices.keys():
                item = data[key]
                if torch.is_tensor(item):
                    graph[key] = item[slices[key][i]:slices[key][i+1]]
                else:
                    graph[key] = item[slices[key][i]:slices[key][i+1]]
            graphs.append(graph)
            
    return graphs

def combine_shards(dataset_dir, split_name):
    # Find all shards for this split
    shard_files = [f for f in os.listdir(dataset_dir) if f.startswith("shard_") and f"split_{split_name}" in f]
    if not shard_files:
        print(f"No shards found for split {split_name}")
        return

    all_graphs = []
    formation_names = []
    
    print(f"Merging {len(shard_files)} shards for {split_name}...")
    for f in tqdm(shard_files, desc=f"Loading {split_name}"):
        shard_path = os.path.join(dataset_dir, f)
        try:
            payload = load_object_safetensors(shard_path)
            formation_names = payload["formation_names"]
            
            # Using a temporary dataset to leverage PyG's internal slicing
            data = payload["data"]
            slices = payload["slices"]
            
            first_key = next(iter(slices))
            num_graphs = len(slices[first_key]) - 1
            
            class TempDataset(InMemoryDataset):
                def __init__(self, data, slices):
                    super().__init__(root=None)
                    self.data = data
                    self.slices = slices
                def len(self):
                    return num_graphs
            
            temp_ds = TempDataset(data, slices)
            for i in range(num_graphs):
                all_graphs.append(temp_ds[i])
            
            del payload
            del data
            del slices
            gc.collect()
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not all_graphs:
        print("No graphs collected.")
        return

    print(f"Total graphs: {len(all_graphs)}. Collating and saving...")
    output_name = f"combined_{split_name}.safetensors"
    output_path = os.path.join(dataset_dir, output_name)
    
    save_dataset_shard(output_path, all_graphs, formation_names, split_name)
    print(f"Successfully saved {split_name} to {output_path}")

if __name__ == "__main__":
    # Resolve the default datasets directory relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_datasets_dir = os.path.join(project_root, "datasets")

    parser = argparse.ArgumentParser(description="Combine dataset shards into single files.")
    parser.add_argument("--dir", type=str, default=default_datasets_dir, help=f"Directory containing shards (default: {default_datasets_dir})")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"], help="Split to combine")
    parser.add_argument("--delete_shards", action="store_true", help="Delete original shards after combining")
    
    args = parser.parse_args()
    
    dataset_dir = args.dir
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory {dataset_dir} does not exist.")
        exit(1)

    splits_to_process = ["train", "val", "test"] if args.split == "all" else [args.split]
    
    for s in splits_to_process:
        combine_shards(dataset_dir, s)
        
    if args.delete_shards:
        print("Deleting shards...")
        for f in os.listdir(dataset_dir):
            if f.startswith("shard_") and (".safetensors" in f or ".pt" in f):
                os.remove(os.path.join(dataset_dir, f))
        print("Shards deleted.")
