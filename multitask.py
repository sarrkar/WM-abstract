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
from helper import MultiTaskDataset, custom_collate


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

task_name_task_index = {
    '1back_category': 3,
    '1back_identity': 2,
    '1back_position': 1,
    '2back_category': 6,
    '2back_identity': 5,
    '2back_position': 4,
    '3back_category': 9,
    '3back_identity': 8,
    '3back_position': 7,
    'dms_category': 43,
    'dms_identity': 42,
    'dms_position': 41,
    'interdms_AABB_category_category': 18,
    'interdms_AABB_category_identity': 17,
    'interdms_AABB_category_position': 16,
    'interdms_AABB_identity_category': 15,
    'interdms_AABB_identity_identity': 14,
    'interdms_AABB_identity_position': 13,
    'interdms_AABB_position_category': 12,
    'interdms_AABB_position_identity': 11,
    'interdms_AABB_position_position': 10, 
    'interdms_ABAB_category_category': 27,
    'interdms_ABAB_category_identity': 26,
    'interdms_ABAB_category_position': 25,
    'interdms_ABAB_identity_category': 24,
    'interdms_ABAB_identity_identity': 23,
    'interdms_ABAB_identity_position': 22,
    'interdms_ABAB_position_category': 21,
    'interdms_ABAB_position_identity': 20,
    'interdms_ABAB_position_position': 19,
    'interdms_ABBA_category_category': 36,
    'interdms_ABBA_category_identity': 35,
    'interdms_ABBA_category_position': 34,
    'interdms_ABBA_identity_category': 33,
    'interdms_ABBA_identity_identity': 32,
    'interdms_ABBA_identity_position': 31,
    'interdms_ABBA_position_category': 30,
    'interdms_ABBA_position_identity': 29,
    'interdms_ABBA_position_position': 28,

    'ctxdms_category_identity_position': 40,
    'ctxdms_position_category_identity': 37,
    'ctxdms_position_identity_category': 38,
    'ctxdms_identity_position_category': 39

}
dataloaders = {
    'dms_category': (DMSDataset, {"feature": "category"}),
    'dms_identity': (DMSDataset, {"feature": "identity"}),
    'dms_position': (DMSDataset, {"feature": "position"}),
    '1back_category': (NBackDataset, {"feature": "category", "nback_n": 1}),
    '2back_category': (NBackDataset, {"feature": "category", "nback_n": 2}),
    '3back_category': (NBackDataset, {"feature": "category", "nback_n": 3}),

    '1back_position': (NBackDataset, {"feature": "position", "nback_n": 1}),
    '2back_position': (NBackDataset, {"feature": "position", "nback_n": 2}),
    '3back_position': (NBackDataset, {"feature": "position", "nback_n": 3}),

    '1back_identity': (NBackDataset, {"feature": "identity", "nback_n": 1}),
    '2back_identity': (NBackDataset, {"feature": "identity", "nback_n": 2}),
    '3back_identity': (NBackDataset, {"feature": "identity", "nback_n": 3}),
    
    'interdms_AABB_category_category': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'category'}),
    'interdms_AABB_category_identity': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'identity'}),
    'interdms_AABB_category_position': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'position'}),
    'interdms_AABB_identity_category': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'category'}),
    'interdms_AABB_identity_identity': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'identity'}),
    'interdms_AABB_identity_position': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'position'}),
    'interdms_AABB_position_category': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'category'}),
    'interdms_AABB_position_identity': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'identity'}),
    'interdms_AABB_position_position': (InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'position'}),
    'interdms_ABAB_category_category': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'category'}),
    'interdms_ABAB_category_identity': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'identity'}),
    'interdms_ABAB_category_position': (InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'position'}),
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

   
    'ctxdms_category_identity_position': (CtxDMSDataset,
      {"features": ["category", "identity", "position"]}
    ),
    'ctxdms_position_category_identity': (CtxDMSDataset,
      {"features": ["position", "category", "identity"]}
    ),
    'ctxdms_position_identity_category': (CtxDMSDataset,
      {"features": ["position", "identity", "category"]}
    ),
    'ctxdms_identity_position_category': (CtxDMSDataset,
      {"features": ["identity", "position", "category"]}
    )
}
batch_size = 128

# Instantiate MultiTaskDataset for training and validation
train_multi_task_dataset = MultiTaskDataset(dataloaders_dict=dataloaders, mode='train')
val_multi_task_dataset = MultiTaskDataset(dataloaders_dict=dataloaders, mode='val')



train_dataloader = DataLoader(
    train_multi_task_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate
)

val_dataloader = DataLoader(
    val_multi_task_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=custom_collate
)


rnn_types = ['GRU']
input_size = 16 + 43 # todo: modify based on the number of categories, identities, and positions + total number of tasks
output_size = 3  # 3 possible actions


def train_model(model, train_dataloader, val_dataloader, num_epochs=2000, learning_rate=0.001, verbose=False, savefolder = None):
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([6.0, 6.0, 1.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []

    # Initialize dictionaries to track per-task accuracy
    task_train_accuracies = {}
    task_val_accuracies = {}

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        running_val_accuracy = 0.0
        total_train_batches = 0
        total_val_batches = 0

        # Training Phase
        model.train()
        for i, (inputs, labels, task_index) in enumerate(train_dataloader):

            inputs, labels, task_index = inputs.to(device), labels.to(device), task_index.to(device)

            task_index_extended = task_index.unsqueeze(1)
            task_index_repeated = task_index_extended.repeat(1, inputs.shape[1], 1)
            concatenated = torch.cat((inputs, task_index_repeated), dim=-1)
            concatenated = concatenated.float()

            optimizer.zero_grad()
            outputs, _ = model(concatenated)

            softmax_outputs = F.softmax(outputs, dim=-1)
            predicted_actions = torch.argmax(softmax_outputs, dim=-1)
            correct_predictions = (predicted_actions == labels).float()
            accuracy = correct_predictions.sum() / correct_predictions.numel()

            # Calculate loss and backpropagate
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1).long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += accuracy.item()
            total_train_batches += 1

            # Track accuracy per task
            unique_samples = torch.unique(task_index, dim = 0, return_inverse = False, return_counts = False)
            
            for task in unique_samples:
                if int(torch.where(task == 1)[0].cpu().numpy() + 1) not in task_train_accuracies.keys():
                    task_train_accuracies[int(torch.where(task == 1)[0].cpu().numpy() + 1)] = []
                matching_indices = torch.nonzero(torch.all(task_index == task, dim = 1)).squeeze()
                if matching_indices.is_cuda:
                    matching_indices = matching_indices.cpu()
                matching_indices = matching_indices.numpy()
       
                task_train_accuracies[int(torch.where(task == 1)[0].cpu().numpy() + 1)].append(float((correct_predictions[matching_indices].sum() / correct_predictions[matching_indices].numel()).cpu()))
           
        # Validation Phase
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels, task_index) in enumerate(val_dataloader):
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
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1).long())

                running_val_loss += loss.item()
                running_val_accuracy += accuracy.item()
                total_val_batches += 1

                # Track accuracy per task
                unique_samples = torch.unique(task_index, dim = 0, return_inverse = False, return_counts = False)
                
                for task in unique_samples:
                    if int(torch.where(task == 1)[0].cpu().numpy() + 1) not in task_val_accuracies.keys():
                        task_val_accuracies[int(torch.where(task == 1)[0].cpu().numpy() + 1)] = []
                    matching_indices = torch.nonzero(torch.all(task_index == task, dim = 1)).squeeze()
                    if matching_indices.is_cuda:
                        matching_indices = matching_indices.cpu()
                    matching_indices = matching_indices.numpy()
        
                    task_val_accuracies[int(torch.where(task == 1)[0].cpu().numpy() + 1)].append(float((correct_predictions[matching_indices].sum() / correct_predictions[matching_indices].numel()).cpu()))
            
        # Calculate Epoch Metrics
        epoch_loss = running_loss / total_train_batches
        epoch_accuracy = running_accuracy / total_train_batches
        epoch_val_loss = running_val_loss / total_val_batches
        epoch_val_accuracy = running_val_accuracy / total_val_batches

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
                    f'Accuracy: {epoch_accuracy * 100:.2f}%, '
                    f'Validation Loss: {epoch_val_loss:.4f}, '
                    f'Validation Accuracy: {epoch_val_accuracy * 100:.2f}%')
        if epoch % 500 == 0: # save every 500 epochs
            model_path = os.path.join(savefolder, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Intermediate Model is saved at {model_path}")


        if np.mean(val_accuracies[-10:]) > 0.99:
            break
    
    # save the model
    final_model_path = os.path.join(savefolder, f'final_model_rep5.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final models is saved at {final_model_path}")

    # Final Epoch Logging
    print(f'Final Epoch [{num_epochs}/{num_epochs}], Loss: {epoch_loss:.4f}, '
            f'Accuracy: {epoch_accuracy * 100:.2f}%, '
            f'Validation Loss: {epoch_val_loss:.4f}, '
            f'Validation Accuracy: {epoch_val_accuracy * 100:.2f}%')

    # Plotting Loss and Accuracy
    plt.figure(figsize=(15, 12))

    plt.subplot(2, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'training_trajectories_rep5.png'))

    # Plot per-task accuracy for training and validation
    task_names = list(dataloaders.keys())
    print(f"train_task_accuracy: {task_train_accuracies.keys()}")
    
    plt.figure(figsize=(24, 16))
    for task_id, task_name in enumerate(task_names):
        task_id = task_name_task_index[task_name]
        print(f"task_id: {task_id}")    
        # Plot training accuracy per task
        plt.plot(task_train_accuracies[int(task_id)], label=f'{task_name} - Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'per task training Accuracy')
    plt.legend(loc = "upper left", bbox_to_anchor=(1,1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'per_task_training_trajectories_rep5.png'))

    plt.figure(figsize=(24, 16))
    for task_id, task_name in enumerate(task_names):
        task_id = task_name_task_index[task_name]
        # Plot validation accuracy per task
        plt.plot(task_val_accuracies[int(task_id)], label=f'{task_name} - Validation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'per task validation Accuracy')
    plt.legend(loc = "upper left", bbox_to_anchor=(1,1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'per_task_validation_trajectories_rep5.png'))
    
    # save per task accuracy
    torch.save(task_val_accuracies, os.path.join(results_dir, f'per_task_validation_trajectories_rep5.pth'))
    torch.save(task_train_accuracies, os.path.join(results_dir, f'per_task_training_trajectories_rep5.pth'))

# Training the Model
hidden_states = [256]
results_dir = os.path.join('/home/xiaoxuan/projects/WM-abstract', 'results')
os.makedirs(results_dir, exist_ok=True)



for hidden_size in hidden_states:
    for rnn_type in rnn_types:
        print(f'Training with RNN type: {rnn_type}, Hidden size: {hidden_size}')
        model = CustomRNN(input_size, hidden_size, output_size, rnn_type).to(device)
        train_model(model, train_dataloader, val_dataloader, 
                    num_epochs=3000, learning_rate=0.001, verbose=True, savefolder=results_dir)
        
        print('-' * 80)