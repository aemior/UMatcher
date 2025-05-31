import torch
import numpy as np
from torch.utils.data import Sampler, ConcatDataset, Subset
from lib.dataset.COCO import COCO
from lib.dataset.SA1B import SA1B
from typing import List

# Plan A: ProportionalDistributedSampler, sample from each dataset according to the specified proportions
class ProportionalDistributedSampler(Sampler):
    def __init__(self, datasets: List[torch.utils.data.Dataset], sample_size: int, proportions: List[float], shuffle: bool = True, num_replicas: int = 1, rank: int = 0):
        self.datasets = datasets
        self.sample_size = sample_size
        self.proportions = proportions
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        
        # Calculate number of samples needed from each dataset based on proportions
        self.num_samples_per_dataset = [int(sample_size * p) for p in proportions]

        # Ensure proportions sum to 1 and validate other arguments
        if not np.isclose(sum(proportions), 1.0):
            raise ValueError("Proportions must sum to 1.")
        if len(proportions) != len(datasets):
            raise ValueError("Length of proportions must match the number of datasets.")

        # Dataset lengths for reference
        self.dataset_lengths = [len(dataset) for dataset in datasets]

    def __iter__(self):
        all_indices = []
        
        for i, dataset in enumerate(self.datasets):
            # Sample indices for each dataset according to the specified proportion
            dataset_indices = np.random.choice(
                self.dataset_lengths[i],
                size=self.num_samples_per_dataset[i],
                replace=True  # Sampling with replacement to ensure we get required samples
            )
            # Adjust for the ConcatDataset indexing
            dataset_indices = dataset_indices + sum(self.dataset_lengths[:i])
            all_indices.extend(dataset_indices)
        
        # Shuffle all indices if required
        if self.shuffle:
            np.random.shuffle(all_indices)

        # Split indices across GPUs in a distributed setup
        indices = all_indices[self.rank:self.sample_size:self.num_replicas]
        
        return iter(indices)

    def __len__(self):
        # Length is the number of samples per epoch for each replica
        return self.sample_size // self.num_replicas

# Plan B: FusionDataset, reset the datasets every epoch
def FusionDataset(datasets: List[torch.utils.data.Dataset], proportions: List[float], shuffle: bool = True, random_state=None) -> torch.utils.data.Dataset:
    # Generate subsets for each dataset according to the specified proportions
    subsets = []
    for dataset, proportion in zip(datasets, proportions):
        num_samples = int(len(dataset) * proportion)
        if shuffle:
            if random_state is None:
                random_state = np.random
            indices = random_state.choice(len(dataset), num_samples, replace=False)  # Randomly select indices
        else:
            indices = np.arange(num_samples)
        subset = Subset(dataset, indices)  # Create subset based on indices
        subsets.append(subset)

    # Concatenate all subsets into a single dataset
    return ConcatDataset(subsets)

def GetMultiDataset(dataset_config, search_size, search_scale, template_size, template_scale, dual_template):
    # Load datasets
    print("Loading datasets...")
    datasets = []
    datasets_ratios = []
    for data_set in dataset_config:
        if data_set.RATIO == 0:
            continue
        if data_set.NAME == "COCO":
            datasets.append(
                COCO(
                    data_set.ROOT_DIR,
                    data_set.ANNO_FILE,
                    template_size,
                    template_scale,
                    search_size,
                    search_scale,
                    dual_template=dual_template
                )
            )
        elif data_set.NAME == "SA1B":
            datasets.append(
                SA1B(
                    data_set.ROOT_DIRS,
                    template_size,
                    template_scale,
                    search_size,
                    search_scale,
                    dual_template=dual_template
                )
            )
        datasets_ratios.append(data_set.RATIO)
        print(f"Dataset {data_set.NAME} loaded with ratio {data_set.RATIO}")
    return datasets, datasets_ratios

    