import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(model, train_dataloader, val_dataloader, num_epochs=2000, learning_rate=0.001, verbose=False):
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([6.0, 6.0, 1.0]).to('cuda'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []

    # Initialize confusion matrices for each task
    task_names = ['dms_category', 'dms_identity', 'dms_position', 'nback_category',
                  'nback_identity', 'nback_position', 'ctxdms_category_identity_position',
                  'ctxdms_position_category_identity', 'ctxdms_position_identity_category',
                  'interdms_AABB_category', 'interdms_AABB_identity', 'interdms_AABB_position',
                  'interdms_ABBA_category', 'interdms_ABBA_identity', 'interdms_ABBA_position',
                  'interdms_ABAB_category', 'interdms_ABAB_identity', 'interdms_ABAB_position']

    task_confusion_matrices = {task_name: np.zeros((3, 3), dtype=int) for task_name in task_names}
    dms_confusion = {key: [[[] for _ in range(16 if key == 'dms_identity' else 4)] for _ in range(16 if key == 'dms_identity' else 4)] for key in ['dms_category', 'dms_identity', 'dms_position']}
    dms_embs = {key: [[] for _ in range(16 if key == 'dms_identity' else 4)] for key in ['dms_category', 'dms_identity', 'dms_position']}

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
            inputs, labels, task_index = inputs.to('cuda'), labels.to('cuda'), task_index.to('cuda')

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
            inputs, labels, task_index = inputs.to('cuda'), labels.to('cuda'), task_index.to('cuda')

            task_index_extended = task_index.unsqueeze(1)
            task_index_repeated = task_index_extended.repeat(1, inputs.shape[1], 1)
            concatenated = torch.cat((inputs, task_index_repeated), dim=-1)
            concatenated = concatenated.float()
            with torch.no_grad():
                outputs,_ = model(concatenated)
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

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy * 100:.2f}%')

    # Confusion matrices update for each task
    for i, (inputs, labels, task_index) in enumerate(val_dataloader):
        inputs, labels, task_index = inputs.to('cuda'), labels.to('cuda'), task_index.to('cuda')

        task_index_extended = task_index.unsqueeze(1)
        task_index_repeated = task_index_extended.repeat(1, inputs.shape[1], 1)
        concatenated = torch.cat((inputs, task_index_repeated), dim=-1)
        concatenated = concatenated.float()

        with torch.no_grad():
            outputs, h = model(concatenated)

        softmax_outputs = F.softmax(outputs, dim=-1)
        predicted_actions = torch.argmax(softmax_outputs, dim=-1)

        # Update confusion matrix for each task
        for j in range(labels.shape[0]):
            task_id = torch.argmax(task_index[j]).item()
            task_name_str = task_names[task_id]
            for k in range(labels.shape[1]):
                true_label = labels[j][k].item()
                predicted_label = predicted_actions[j][k].item()
                task_confusion_matrices[task_name_str][int(true_label), int(predicted_label)] += 1

            if task_name_str == 'dms_identity':
                identity1 = torch.argmax(inputs[j][0][4:20]).item()
                identity2 = torch.argmax(inputs[j][1][4:20]).item()
                dms_confusion[task_name_str][identity1][identity2].append(softmax_outputs[j][1][1].item())
                dms_embs[task_name_str][identity1].append(h[j][0].detach().cpu().numpy())
            elif task_name_str == 'dms_category':
                category1 = torch.argmax(inputs[j][0][0:4]).item()
                category2 = torch.argmax(inputs[j][1][0:4]).item()
                dms_confusion[task_name_str][category1][category2].append(softmax_outputs[j][1][1].item())
                dms_embs[task_name_str][category1].append(h[j][0].detach().cpu().numpy())
            elif task_name_str == 'dms_position':
                position1 = torch.argmax(inputs[j][0][20:24]).item()
                position2 = torch.argmax(inputs[j][1][20:24]).item()
                dms_confusion[task_name_str][position1][position2].append(softmax_outputs[j][1][1].item())
                dms_embs[task_name_str][position1].append(h[j][0].detach().cpu().numpy())

    # Plot confusion matrices
    for task_id, cm in task_confusion_matrices.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for Task {task_id}')
        plt.show()

    for task in ['dms_category', 'dms_identity', 'dms_position']:
      lim = 16 if task == 'dms_identity' else 4
      for i in range(lim):
          for j in range(lim):
              if len(dms_confusion[task][i][j]) > 0:
                dms_confusion[task][i][j] = np.mean(dms_confusion[task][i][j])
              else:
                dms_confusion[task][i][j] = 0
      dms_confusion[task] = np.array(dms_confusion[task])
      plt.figure(figsize=(8, 6))
      sns.heatmap(dms_confusion[task], annot=True, fmt='.2f')
      plt.xlabel('Predicted')
      plt.ylabel('True')
      plt.title(f'Probability Matrix for Task {task}')
      plt.show()

      emb_heatmap = np.zeros((lim, lim))
      for i in range(lim):
          for j in range(lim):
              emb1 = np.mean(dms_embs[task][i], axis=0)
              emb2 = np.mean(dms_embs[task][j], axis=0)
              emb_heatmap[i, j] = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
      plt.figure(figsize=(8, 6))
      sns.heatmap(emb_heatmap, annot=True, fmt='.2f')
      plt.xlabel('Predicted')
      plt.ylabel('True')
      plt.title(f'Cosine Similarity of {task} embeddings')
      plt.show()

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


train_datasets = []
val_datasets = []

batch_size=2048

for name, (dataset_class, kwargs) in dataloaders.items():
    total_size = batch_size
    train_size = int(0.7 * total_size)
    val_size = total_size - train_size
    
    train_dataset = dataset_class(
        dataset_size=train_size,
        category_size=4,
        identity_size=4,
        std=0,
        pad_to=6,
        **kwargs
    )
    
    val_dataset = dataset_class(
        dataset_size=val_size,
        category_size=4,
        identity_size=4,
        std=0.15,
        pad_to=6,
        **kwargs
    )
    
    train_datasets.append(train_dataset)
    val_datasets.append(val_dataset)


# merge datasets
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
val_dataset = torch.utils.data.ConcatDataset(val_datasets)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

hidden_states = [256]

for hidden_size in hidden_states:
  for rnn_type in rnn_types:
    print(rnn_type, hidden_size)
    model = CustomRNN(input_size, hidden_size, output_size, rnn_type).to('cuda')
    train_model(model, train_dataloader, val_dataloader, num_epochs=500, learning_rate=0.001, verbose=True)
    plt.savefig(f'figs/{rnn_type}_{hidden_size}_mixed.png')
    print('-' * 80)
