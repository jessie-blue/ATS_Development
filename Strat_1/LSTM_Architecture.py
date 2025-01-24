# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:12:21 2024

@author: ktsar
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LSTM(nn.Module): 
    
    def __init__(self, input_size, output_size, hidden_size, num_layers, device):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # Layers
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)
        
        self.relu = nn.ReLU()
        
        self.linear_layer = nn.Linear(in_features = hidden_size, 
                                      out_features = output_size)

    
    def forward(self, x):
       
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # 1 instead of x.size(0) on datacamp 
        # Initialize long-term memory
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # 1 instead of x.size(0)
        # Pass all inputs to lstm layer
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(out)
        out = self.linear_layer(out[:, -1, :])
        return out
    
    
    
    
class LSTM_V1(nn.Module): 
    
    def __init__(self, input_size, output_size, hidden_size, num_layers, device):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # Layers
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)
        
        self.relu1 = nn.ReLU()
        
        self.lstm = nn.LSTM(input_size = hidden_size,
                            hidden_size = hidden_size * 2,
                            num_layers= num_layers + 1,
                            batch_first=True)
        
        self.relu2 = nn.ReLU()
        
        self.linear_layer = nn.Linear(in_features = hidden_size, 
                                      out_features = output_size)

    
    def forward(self, x):
       
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # 1 instead of x.size(0) on datacamp 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) # 1 instead of x.size(0)
        out, _ = self.lstm(x, (h0, c0))
        #out = self.relu(out)
        out = self.linear_layer(out[:, -1, :])
        return out
    

class TimeSeriesDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,i):
        return self.X[i], self.y[i]
        
        
        
        
class LSTM_V2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, device):
        super(LSTM_V2, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        
        # Define the LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Define ReLU activation functions
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        # Define the linear layer
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through the first LSTM layer
        out, _ = self.lstm1(x, (h0, c0))
        out = self.relu1(out)
        
        # Forward pass through the second LSTM layer
        out, _ = self.lstm2(out, (h0, c0))
        out = self.relu2(out)
        
        # Flatten the output from the LSTM layers
        out = out[:, -1, :]
        
        # Forward pass through the linear layer
        out = self.linear(out)
        return out

        
        
class LSTM_V3(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, output_size, device):
        super(LSTM_V3, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        
        # Define the LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers, batch_first=True)
        
        # Define ReLU activation functions
        self.relu = nn.ReLU()
        
        # Define the linear layer
        self.linear = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(x.device)
        
        # Forward pass through the first LSTM layer
        out, _ = self.lstm1(x, (h0, c0))
        out = self.relu(out)
        
        # Forward pass through the second LSTM layer
        h0_2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(x.device)
        c0_2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(x.device)
        out, _ = self.lstm2(out, (h0_2, c0_2))
        
        # Flatten the output from the LSTM layers
        out = out[:, -1, :]
        
        # Forward pass through the linear layer
        out = self.linear(out)
        return out
        
        
class CustomLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Args:
            input_size (int): The number of input features.
            hidden_sizes (list): A list of hidden sizes for each LSTM layer (e.g., [128, 64, 32]).
            output_size (int): The number of output features.
        """
        super(CustomLSTMModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        
        # Define the first LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        
        # Define subsequent LSTM layers
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        
        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        # Pass through the first LSTM layer
        out, _ = self.lstm1(x)
        
        # Pass through the second LSTM layer
        out, _ = self.lstm2(out)
        
        # Pass through the third LSTM layer
        out, _ = self.lstm3(out)
        
        # Pass the final output through the fully connected layer
        out = self.fc(out[:, -1, :])  # Use the last time-step's output
        return out
        
        
        
class CustomLSTMModel_2(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Args:
            input_size (int): The number of input features.
            hidden_sizes (list): A list of hidden sizes for each LSTM layer (e.g., [128, 64]).
            output_size (int): The number of output features.
        """
        super(CustomLSTMModel_2, self).__init__()
        
        # Define the first LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        
        # Define the second LSTM layer
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        
        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        # Pass through the first LSTM layer
        out, _ = self.lstm1(x)
        
        # Pass through the second LSTM layer
        out, _ = self.lstm2(out)
        
        # Pass the final output through the fully connected layer
        out = self.fc(out[:, -1, :])  # Use the last time-step's output
        return out 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        