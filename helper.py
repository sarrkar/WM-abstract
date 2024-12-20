import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tasks.dms import DMSDataset
from tasks.ctx_dms import CtxDMSDataset
from tasks.nback import NBackDataset
from tasks.inter_dms import InterDMSDataset

from models.RNN_model import CustomRNN

def binary_encoding_conversion(binary_input, category_size = 4, identity_size = 2, position_size = 4):
    # Extract identity and position information
    identity_start = category_size
    identity_end = identity_start + category_size * identity_size
    position_start = identity_end
    position_end = position_start + position_size

    # Extract the binary slices for identity and position
    org_identity_bin = binary_input[identity_start:identity_end]
    org_position_bin = binary_input[position_start:position_end]
    # print(f"position bin: {org_position_bin}")
    # for each element in the bin, round it up to its closest integer
    identity_bin = torch.round(org_identity_bin)
    position_bin = torch.round(org_position_bin)


    position_index = torch.where(position_bin == 1)[0]
    
    identity_index = torch.where(identity_bin == 1)[0]
 

    # todo: temporary fix for the case where there are no 1s in the binary encoding
    # if len(position_index) == 0:
    #     position_bin[position_bin == 2] = 1
    #     position_index = torch.where(position_bin == 1)[0]
    # elif len(position_index) == 2:
    #     position_index = position_index[:1]
    # try:
    #     assert len(position_index) == 1, f"Position index should have length 1, but got {len(position_index)}"
    # except:
    #     position_index = torch.where(org_position_bin == torch.max(org_position_bin))[0]
    
    # if len(identity_index) == 0:
    #     identity_bin[identity_bin == 2] = 1
    #     identity_index = torch.where(identity_bin == 1)[0]
    # elif len(identity_index) == 2:
    #     identity_index = identity_index[:1]
    # try:
    #     assert len(identity_index) == 1, f"Identity index should have length 1, but got {len(position_index)}"
    # except:
    #     identity_index = torch.where(org_identity_bin == torch.max(org_identity_bin))[0]


    combined_int = position_index*category_size*identity_size + identity_index
    return combined_int[0]

# Define DataLoaders with a custom collate function if necessary
def custom_collate(batch):
    inputs, labels, task_indices = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    labels = torch.stack(labels, dim=0)
    task_indices = torch.tensor(np.array(task_indices))
    return inputs, labels, task_indices
class MultiTaskDataset(Dataset):
    def __init__(self, dataloaders_dict, mode='train', dataset_size = 128):
        """
        Initializes the MultiTaskDataset.

        Args:
            dataloaders_dict (dict): Dictionary containing task names as keys and tuples of (DatasetClass, kwargs) as values.
            mode (str): 'train' or 'val' to indicate the dataset split.
        """
        self.datasets = {}
        self.task_names = list(dataloaders_dict.keys())

        for task_name, (DatasetClass, kwargs) in dataloaders_dict.items():
            # Adjust kwargs based on mode if necessary
            dataset_kwargs = kwargs.copy()
            if mode == 'train':
                # Example: Modify parameters for training
                dataset_kwargs.update({'std': 0, 'pad_to': 6})
            elif mode == 'val':
                # Example: Modify parameters for validation
                dataset_kwargs.update({'std': 0, 'pad_to': 6, 'dataset_size': dataset_size})

            # Instantiate the dataset
            self.datasets[task_name] = DatasetClass(**dataset_kwargs)

        # Calculate the maximum length among all datasets
        self.max_length = max(len(dataset) for dataset in self.datasets.values())

        # Calculate weights based on dataset sizes for balanced sampling
        dataset_sizes = [len(ds) for ds in self.datasets.values()]
        total_size = sum(dataset_sizes)
        self.task_weights = [size / total_size for size in dataset_sizes]

    def __len__(self):
        return self.max_length

    def __getitem__(self, idx):
        """
        Fetches a sample from a randomly selected task.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (inputs, labels, task_index)
        """
        # Randomly select a task based on weights
        task_name = random.choices(self.task_names, weights=self.task_weights, k=1)[0]
        task_dataset = self.datasets[task_name]

        # Get the sample from the selected task's dataset
        sample = task_dataset[idx % len(task_dataset)]  

        inputs, labels, task_index = sample  
        return torch.Tensor(inputs), torch.Tensor(labels), task_index
