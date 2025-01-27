import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import pickle
from categorical_nback_task import Nback_Dataset_categorical
from RNN_model import CustomRNN
import torch.nn.functional as F

class CustomRNNWithActivations(CustomRNN):
    def forward(self, x):
        h0 = self.init_hidden(x.size(0)).to(x.device)
        hidden_states = []

        if self.rnn_type == 'LSTM':
            c0 = self.init_hidden(x.size(0)).to(x.device)
            out, (hn, cn) = self.rnn(x, (h0, c0))
        else:
            out, hn = self.rnn(x, h0)

        # Store hidden states for each timestep
        for t in range(out.size(1)):  # Iterate over the timesteps
            hidden_states.append(out[:, t, :].detach().cpu().numpy())

        out = self.fc(out)  # Apply the fully connected layer to each timestep
        
        return out, hidden_states  # Return output and all hidden states

def save_activations_to_dataframe(model, dataloader):
    model.eval()  # Set model to evaluation mode

    inputs_list = []
    hidden_states_list = []
    output_actions_list = []

    with torch.no_grad():
        for inputs, labels, task_index in dataloader:
            
            # Step 1: Extend the task_index tensor by adding a new dimension of size seq_len
            task_index_extended = task_index.unsqueeze(1)  # Shape: (batch_size, 1, task_index_len)

            # Step 2: Repeat the task_index along the new dimension to match seq_len
            task_index_repeated = task_index_extended.repeat(1, inputs.shape[1], 1)  # Shape: (batch_size, seq_len, task_index_len)

            # Step 3: Concatenate along the last dimension (input_size + task_index_len)
            concatenated = torch.cat((inputs, task_index_repeated), dim=-1)  # Shape: (batch_size, seq_len, input_size + task_index_len)
            
            outputs, hidden_states = model(concatenated)
            
            softmax_outputs = F.softmax(outputs, dim=-1)
            predicted_actions = torch.argmax(softmax_outputs, dim=-1)

            # Store the data
            print(inputs.shape)
            print(len(hidden_states))
            print(hidden_states[0].shape)
            inputs_list.append(inputs.detach().cpu().numpy())
            hidden_states_list.append(hidden_states)
            output_actions_list.append(predicted_actions.detach().cpu().numpy())

    # Create a DataFrame
    data = {
        'inputs': inputs_list,
        'hidden_states': hidden_states_list,
        'output_actions': output_actions_list
    }
    df = pd.DataFrame(data)
    
    # Save to a pickle file
    with open('1back_model_activations.pkl', 'wb') as f:
        pickle.dump(df, f)

# Load the dataset and initialize DataLoader
nback_dataset = Nback_Dataset_categorical(dataset_size=512*8, n = 0)
dataloader = DataLoader(nback_dataset, batch_size=512, shuffle=False)

# Initialize the model with the desired RNN type
input_size = 14  # Since each sequence element is a scalar
hidden_size = 256 # You can adjust the hidden layer size
output_size = 3  # Three possible actions
rnn_type = 'GRU'  # Can be 'RNN', 'GRU', or 'LSTM'

model = CustomRNNWithActivations(input_size, hidden_size, output_size, rnn_type)

# Load the trained model
model.load_state_dict(torch.load('1back_final_model.pth'))

# Record activations and save to pickle
save_activations_to_dataframe(model, dataloader)
