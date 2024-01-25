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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        