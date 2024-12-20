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

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rnn_type = 'GRU'
category_size: int = 4
identity_size: int = 2
position_size: int = 4
input_size = 16 + 43 # todo: modify based on the number of categories, identities, and positions + total number of tasks
output_size = 3  # 3 possible actions
hidden_size = 256
batch_size = 128
n_unique_stimuli = position_size * (category_size * identity_size)
model = CustomRNN(input_size, hidden_size, output_size, rnn_type).to(device)

# Load the saved model
results_dir = os.path.join('/home/xiaoxuan/projects/WM-abstract', 'results')
saved_model_path = os.path.join(results_dir, f'final_model_rep5.pth')
model.load_state_dict(torch.load(saved_model_path))
model.eval()  # Set the model to evaluation mode

print(f"Model loaded from {saved_model_path}")

# Define the task
dataloaders = {
    # 'dms_category': (DMSDataset, {"feature": "category"}),
    # 'dms_identity': (DMSDataset, {"feature": "identity"}),
    # 'dms_position': (DMSDataset, {"feature": "position"}),

    '1back_category': (NBackDataset, {"feature": "category", "nback_n": 1}),
    '2back_category': (NBackDataset, {"feature": "category", "nback_n": 2}),
    '3back_category': (NBackDataset, {"feature": "category", "nback_n": 3}),

    '1back_position': (NBackDataset, {"feature": "position", "nback_n": 1}),
    '2back_position': (NBackDataset, {"feature": "position", "nback_n": 2}),
    '3back_position': (NBackDataset, {"feature": "position", "nback_n": 3}),

    '1back_identity': (NBackDataset, {"feature": "identity", "nback_n": 1}),
    '2back_identity': (NBackDataset, {"feature": "identity", "nback_n": 2}),
    '3back_identity': (NBackDataset, {"feature": "identity", "nback_n": 3}),
    
    # 'interdms_AABB_category_category': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'category'}),
    # 'interdms_AABB_category_identity': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'identity'}),
    # 'interdms_AABB_category_position': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'position'}),
    # 'interdms_AABB_identity_category': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'category'}),
    # 'interdms_AABB_identity_identity': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'identity'}),
    # 'interdms_AABB_identity_position': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'position'}),
    # 'interdms_AABB_position_category': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'category'}),
    # 'interdms_AABB_position_identity': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'identity'}),
    # 'interdms_AABB_position_position': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'position'}),
    # 'interdms_ABAB_category_category': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'category'}),
    # 'interdms_ABAB_category_identity': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'identity'}),
    # 'interdms_ABAB_category_position': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'position'}),
    'interdms_ABAB_identity_category': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'identity', 'feature_2': 'category'}),
    'interdms_ABAB_identity_identity': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'identity', 'feature_2': 'identity'}),
    'interdms_ABAB_identity_position': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'identity', 'feature_2': 'position'}),
    'interdms_ABAB_position_category': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'position', 'feature_2': 'category'}),
    'interdms_ABAB_position_identity': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'position', 'feature_2': 'identity'}),
    'interdms_ABAB_position_position': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'position', 'feature_2': 'position'}),
    'interdms_ABBA_category_category': (InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'category', 'feature_2': 'category'}),
    'interdms_ABBA_category_identity': (InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'category', 'feature_2': 'identity'}),
    'interdms_ABBA_category_position': (InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'category', 'feature_2': 'position'}),
    'interdms_ABBA_identity_category': (InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'identity', 'feature_2': 'category'}),
    'interdms_ABBA_identity_identity': (InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'identity', 'feature_2': 'identity'}),
    'interdms_ABBA_identity_position': (InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'identity', 'feature_2': 'position'}),
    'interdms_ABBA_position_category': (InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'position', 'feature_2': 'category'}),
    'interdms_ABBA_position_identity': (InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'position', 'feature_2': 'identity'}),
    'interdms_ABBA_position_position': (InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'position', 'feature_2': 'position'}),

   
    }

mattered_stim_pairs = {
    'dms_category': [(0,1)],
    'dms_identity': [(0,1)],
    'dms_position': [(0,1)],

    '1back_category': [(0,1), (1,2), (2,3), (3,4), (4,5)],
    '2back_category': [(0,2), (1,3), (2,4), (3,5)],
    '3back_category': [(0,3), (1,4), (2,5),],

    '1back_position': [(0,1), (1,2), (2,3), (3,4), (4,5)],
    '2back_position': [(0,2), (1,3), (2,4), (3,5)],
    '3back_position': [(0,3), (1,4), (2,5),],

    '1back_identity': [(0,1), (1,2), (2,3), (3,4), (4,5)],
    '2back_identity': [(0,2), (1,3), (2,4), (3,5)],
    '3back_identity': [(0,3), (1,4), (2,5),],

    'interdms_AABB_category_category': [(0,1), (2,3)],
    'interdms_AABB_category_identity': [(0,1), (2,3)],
    'interdms_AABB_category_position': [(0,1), (2,3)],
    'interdms_AABB_identity_category': [(0,1), (2,3)],
    'interdms_AABB_identity_identity': [(0,1), (2,3)],
    'interdms_AABB_identity_position': [(0,1), (2,3)],
    'interdms_AABB_position_category': [(0,1), (2,3)],
    'interdms_AABB_position_identity': [(0,1), (2,3)],
    'interdms_AABB_position_position': [(0,1), (2,3)],

    'interdms_ABAB_category_category': [(0,2), (1,3)],
    'interdms_ABAB_category_identity': [(0,2), (1,3)],
    'interdms_ABAB_category_position': [(0,2), (1,3)],
    'interdms_ABAB_identity_category': [(0,2), (1,3)],
    'interdms_ABAB_identity_identity': [(0,2), (1,3)],
    'interdms_ABAB_identity_position': [(0,2), (1,3)],
    'interdms_ABAB_position_category': [(0,2), (1,3)],
    'interdms_ABAB_position_identity': [(0,2), (1,3)],
    'interdms_ABAB_position_position': [(0,2), (1,3)],

    'interdms_ABBA_category_category': [(0,3), (1,2)],
    'interdms_ABBA_category_identity': [(0,3), (1,2)],
    'interdms_ABBA_category_position': [(0,3), (1,2)],
    'interdms_ABBA_identity_category': [(0,3), (1,2)],
    'interdms_ABBA_identity_identity': [(0,3), (1,2)],
    'interdms_ABBA_identity_position': [(0,3), (1,2)],
    'interdms_ABBA_position_category': [(0,3), (1,2)],
    'interdms_ABBA_position_identity': [(0,3), (1,2)],
    'interdms_ABBA_position_position': [(0,3), (1,2)],
}

task_confusion_matrices = {task_name: np.zeros((n_unique_stimuli, n_unique_stimuli), dtype=int) for task_name in dataloaders.keys()}

for i, (task_name, dataset) in enumerate(dataloaders.items()):
    # collect data for each individual task
    single_task_dataloader = {task_name: dataset}
    val_single_task_dataset = MultiTaskDataset(dataloaders_dict=single_task_dataloader, mode='val', dataset_size=32*32*20)
    val_single_task_dataloader = DataLoader(
        val_single_task_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )

    confusion_template = {}
    accuracies = []
    # Evaluate the model to produce the confusion matrix
    with torch.no_grad():
        for i, (inputs, labels, task_index) in enumerate(val_single_task_dataloader):
            inputs, labels, task_index = inputs.to(device), labels.to(device), task_index.to(device)

            task_index_extended = task_index.unsqueeze(1)
            task_index_repeated = task_index_extended.repeat(1, inputs.shape[1], 1)
            concatenated = torch.cat((inputs, task_index_repeated), dim=-1)
            concatenated = concatenated.float()

            outputs, _ = model(concatenated)
            softmax_outputs = F.softmax(outputs, dim=-1)
            predicted_actions = torch.argmax(softmax_outputs, dim=-1)
            correct_predictions = (predicted_actions == labels).float()
            accuracy = correct_predictions.sum() / correct_predictions.numel()
            accuracies.append(accuracy.cpu().item())
            for stim_pair in mattered_stim_pairs[task_name]:
                stim_i_index, stim_j_index = stim_pair
                for ci_index in range(inputs.shape[0]):
                    curr_input = inputs[ci_index]
                    stim_i = binary_encoding_conversion(curr_input[stim_i_index])
                    stim_j = binary_encoding_conversion(curr_input[stim_j_index])
                    
                    if f"{stim_i},{stim_j}" not in confusion_template.keys():
                        confusion_template[f"{stim_i},{stim_j}"] = []
                    else:
                        confusion_template[f"{stim_i},{stim_j}"].append(predicted_actions.cpu().numpy()[ci_index, stim_j_index])
    print(len(confusion_template.keys()))
    assert len(confusion_template.keys()) == n_unique_stimuli**2
    for stim_i in range(n_unique_stimuli):
        for stim_j in range(n_unique_stimuli):
            task_confusion_matrices[task_name][stim_i,stim_j] = np.mean(confusion_template[f"{stim_i},{stim_j}"])
    
    # Plot the confusion matrix
    plt.figure(figsize=(24, 20))
    sns.heatmap(task_confusion_matrices[task_name], cmap='viridis', annot=True, fmt=".1f", cbar=True)
    plt.title(f'{task_name} task confusion matrix, average accuracy {np.mean(accuracies)}')
    plt.xlabel('Stimulus j')
    plt.ylabel('Stimulus i')
    plt.savefig(os.path.join(savefolder, f'{task_name}_confusion_matrix.png'))