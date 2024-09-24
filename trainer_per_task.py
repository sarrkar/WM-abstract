import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tasks import NBackDataset, DMSDataset, CtxDMSDataset, InterDMSDataset
from models.RNN_model import CustomRNN
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, train_dataloader, val_dataloader, num_epochs=2000, learning_rate=0.001, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []

    best_accuracy = 0.0
    epochs_without_improvement = 0

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        running_val_accuracy = 0.0
        total_train_batches = 0
        total_val_batches = 0
        model.train()
        for i, (inputs, labels, task_index) in enumerate(train_dataloader):
            inputs, labels, task_index = inputs, labels, task_index

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

            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1).long())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += accuracy.item()
            total_train_batches += 1

        model.eval()
        for i, (inputs, labels, task_index) in enumerate(val_dataloader):
            inputs, labels, task_index = inputs, labels, task_index

            task_index_extended = task_index.unsqueeze(1)
            task_index_repeated = task_index_extended.repeat(1, inputs.shape[1], 1)
            concatenated = torch.cat((inputs, task_index_repeated), dim=-1)
            concatenated = concatenated.float()
            with torch.no_grad():
              outputs, _ = model(concatenated)
            softmax_outputs = F.softmax(outputs, dim=-1)
            predicted_actions = torch.argmax(softmax_outputs, dim=-1)
            correct_predictions = (predicted_actions == labels).float()
            accuracy = correct_predictions.sum() / correct_predictions.numel()
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1).long())
            running_val_loss += loss.item()
            running_val_accuracy += accuracy.item()
            total_val_batches += 1

        epoch_loss = running_loss / total_train_batches
        epoch_accuracy = running_accuracy / total_train_batches
        epoch_val_loss = running_val_loss / total_val_batches
        epoch_val_accuracy = running_val_accuracy / total_val_batches

        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        if verbose:
          print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy * 100:.2f}%')

        if len(accuracies) > 100 and np.mean(accuracies[-10:]) >= 0.99:
            # print(f"Accuracy has saturated. Stopping training at epoch {epoch+1}.")
            break
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy * 100:.2f}%')

    plt.figure(figsize=(5, 5))

    plt.subplot(2, 2, 1)
    plt.plot(losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(val_losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(val_accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()

batch_size = 2048

dataloaders = {
    'dms_category': (DMSDataset, {"feature": "category"}),
    'dms_identity': (DMSDataset, {"feature": "identity"}),
    'dms_position': (DMSDataset, {"feature": "position"}),
    'nback_category': (NBackDataset, {"feature": "category"}),
    'nback_identity': (NBackDataset, {"feature": "identity"}),
    'nback_position': (NBackDataset, {"feature": "position"}),
    'ctxdms_category_identity_position': (CtxDMSDataset,
      {"features": ["category", "identity", "position"]}
    ),
    'ctxdms_position_category_identity': (CtxDMSDataset,
      {"features": ["position", "category", "identity"]}
    ),
    'ctxdms_position_identity_category': (CtxDMSDataset,
      {"features": ["position", "identity", "category"]}
    ),
    'interdms_AABB_category': (InterDMSDataset,
      {"pattern": "AABB", "feature": "category"}
    ),
    'interdms_AABB_identity': (InterDMSDataset,
      {"pattern": "AABB", "feature": "identity"}
    ),
    'interdms_AABB_position': (InterDMSDataset,
      {"pattern": "AABB", "feature": "position"}
    ),
    'interdms_ABBA_category': (InterDMSDataset,
      {"pattern": "ABBA", "feature": "category"}
    ),
    'interdms_ABBA_identity': (InterDMSDataset,
      {"pattern": "ABBA", "feature": "identity"}
    ),
    'interdms_ABBA_position': (InterDMSDataset,
      {"pattern": "ABBA", "feature": "position"}
    ),
    'interdms_ABAB_category': (InterDMSDataset,
      {"pattern": "ABAB", "feature": "category"}
    ),
    'interdms_ABAB_identity': (InterDMSDataset,
      {"pattern": "ABAB", "feature": "identity"}
    ),
    'interdms_ABAB_position': (InterDMSDataset,
      {"pattern": "ABAB", "feature": "position"}
    ),
}

input_size = 24 + 18 
hidden_size = 4 
output_size = 3  
rnn_types = ['RNN']

for name, (dataset_class, kwargs) in dataloaders.items():
  dataset = dataset_class(dataset_size=batch_size, category_size=4, identity_size=4, std=0.15, **kwargs)
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  for rnn_type in rnn_types:
    print(rnn_type, name)
    model = CustomRNN(input_size, hidden_size, output_size, rnn_type)
    train_model(model, train_dataloader, val_dataloader, num_epochs=500, learning_rate=0.001)
    plt.savefig(f'figs/{rnn_type}_{name}.png')
  print('-' * 80)
