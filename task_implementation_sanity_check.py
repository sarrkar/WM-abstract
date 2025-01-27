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
from helper import MultiTaskDataset, custom_collate, binary_encoding_conversion

savefolder = "/home/xiaoxuan/projects/WM-abstract/results"

category_size: int = 4
identity_size: int = 2
position_size: int = 4
n_unique_stimuli = position_size * (category_size * identity_size)

# Define the task
dataloaders = {
    'dms_category': (DMSDataset, {"feature": "category"}),

    '1back_category': (NBackDataset, {"feature": "category", "nback_n": 1}),
    '2back_category': (NBackDataset, {"feature": "category", "nback_n": 2}),
    '3back_category': (NBackDataset, {"feature": "category", "nback_n": 3}),

    'interdms_AABB_category_category': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'category'}),
    'interdms_ABAB_category_category': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'category'}),
    
    'interdms_ABBA_category_category': (InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'category', 'feature_2': 'category'}),
    }



for i, (task_name, dataset) in enumerate(dataloaders.items()):
    # collect data for each individual task
    single_task_dataloader = {task_name: dataset}
    val_single_task_dataset = MultiTaskDataset(dataloaders_dict=single_task_dataloader, mode='val', dataset_size=32*32*20)

   # Visualize and save heatmap for task inputs
    inputs, actions, _ = next(iter(DataLoader(val_single_task_dataset, batch_size=1, collate_fn=custom_collate)))
    inputs_np = inputs.squeeze(0).numpy()
    actions_np = actions.squeeze(0).numpy()


    plt.figure(figsize=(10, 8))
    sns.heatmap(inputs_np, cmap="viridis", cbar=True)
    plt.title(f"Task: {task_name}, Action: {actions_np}")
    plt.xlabel("Features")
    plt.ylabel("Timesteps")

    heatmap_save_path = os.path.join(savefolder, f"heatmap_{task_name}.png")
    plt.savefig(heatmap_save_path)
    plt.close()

    print(f"Saved heatmap for {task_name} at {heatmap_save_path}")
