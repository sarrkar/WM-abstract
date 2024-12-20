import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size = 256, output_size = 3, rnn_type='RNN', num_layers=1):
        super(CustomRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'RNN', 'GRU', or 'LSTM'")
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.rnn_type = rnn_type

    def init_hidden(self, batch_size):
        return nn.init.kaiming_uniform_(torch.empty(1, batch_size, self.hidden_size))
    
    def forward(self, x):
        h0 = self.init_hidden(x.size(0)).to(x.device)

        if self.rnn_type == 'LSTM':
            c0 = self.init_hidden(x.size(0)).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        
       
        lout = self.fc(out)  # Take the output of the last time step
        
        return lout, out


