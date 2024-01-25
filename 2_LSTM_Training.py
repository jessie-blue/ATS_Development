# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:24:36 2024

@author: ktsar
"""

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


from pathlib import Path
from Preprocessing_functions import create_multivariate_rnn_data;
from torch.utils.data import DataLoader, TensorDataset
from helper_functions import accuracy_fn
from LSTM_Architecture import *
from sklearn.preprocessing import MinMaxScaler


df_model = pd.read_parquet("df_model_XLU_v1")

df_model.head()
df_model = df_model.reset_index()
df_model['Date'] = pd.to_datetime(df_model['Date']).dt.date
df_model = df_model.set_index("Date")


## Experiment with a smaller feature space 
# df_model = df_model[['labels', 
#                      'open_low', 
#                      'open_close', 
#                      'gap']]

end_date = df_model.index.max()
#start_date = df_model.index.min()

model_number = 4
data_scaling = True
seq_length =  1
test_size_pct = 0.15

df_model = df_model.sort_index(ascending = False)

## SCALING THE DATA BEFORE CONVERTING IT INTO SUITABLE INPUT FOR RNN 
if data_scaling == True:
    x = df_model.drop(labels = "labels", axis = 1)
    scaler = MinMaxScaler()
    x_fit = scaler.fit(x)
    x = scaler.transform(x)
    x = pd.DataFrame(x)
    y = df_model.labels.to_frame().reset_index()
    
    del df_model
    df_model = y.merge(x, left_index = True, right_index = True)
    df_model = df_model.set_index("Date")
    del x, x_fit,y 

df_model['last_day'] = (df_model.index == end_date).astype(int)

X, y  = create_multivariate_rnn_data(df_model, seq_length)

test_size = int(X.shape[0] * test_size_pct) 
train_size = X.shape[0] - test_size


X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]


# Convert data to PyTorch tensors AND CHECK DIMENSIONS OF TENSOR AND CHECK THE REQUIRED INPUT FOR LSTM 
X_train_tensor = torch.from_numpy(X_train).type(torch.float)#.unsqueeze(1)
y_train_tensor = torch.from_numpy(y_train.values).type(torch.LongTensor)#.unsqueeze(1)

X_test_tensor = torch.from_numpy(X_test).type(torch.float)#.unsqueeze(1)
y_test_tensor = torch.from_numpy(y_test.values).type(torch.LongTensor)#.unsqueeze(1)



### HYPERPARAMETERS
input_feat = X_train.shape[2]
hidden_size = 32
num_layers = 2 
learning_rate = 0.01
epochs =  3000
num_classes = 3
batch_size = 32

train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)




train_loader = DataLoader(train_dataset, 
                          batch_size = batch_size,
                          shuffle = False)


test_loader = DataLoader(test_dataset, 
                          batch_size = batch_size,
                          shuffle = False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for _, batch in enumerate(train_loader):
    
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break 

# INSTANTIATE MODEL
model = LSTM(input_size=input_feat, 
             output_size=num_classes, 
             hidden_size=hidden_size, 
             num_layers=num_layers,
             device=device).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

torch.manual_seed(42)

best_test_accuracy = 0 
best_epoch = 0 

# TRAIN AND TEST MODEL 
for epoch in range(epochs):
    
    model.train(True)
    running_loss = 0.0 
    acc = 0.0
    
    for batch_index, batch in enumerate(train_loader): 
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
       
        output = model(x_batch)
        pred = torch.softmax(output, dim = 1).argmax(dim = 1)
        
        loss = loss_fn(output, y_batch[:,0])
        running_loss += loss.item()
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        batch_accuracy = accuracy_fn(y_true = y_batch[:,0], y_pred = pred) 
        acc += batch_accuracy
        
        #print(f"Epoch: {epoch}, Batch: {batch_index}, Batch Accuracy: {batch_accuracy:.2f}")
        
        #print(epoch, batch_index)
    
    
    avg_loss = running_loss / len(train_loader) # batch_index
    avg_acc = acc / len(train_loader) #batch_index

    #print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f} ")


    ### Testing
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            # 1 Forward pass
            test_pred = model(X)
            ## DO WE NEED TO TRANSFORM USING SOFTMAX? SEE test_pred tensor
            
            # 2 Calculate loss
            test_loss += loss_fn(test_pred, y[:,0])
            
            # 3 Calculate accuracy
            test_acc += accuracy_fn(y_true = y[:,0], 
                                    y_pred = test_pred.argmax(dim = 1)) # may need to add argmax(dim = 1) in here
            
        test_loss /= len(test_loader)
        test_acc  /= len(test_loader)
    
    if test_acc > best_test_accuracy:
        
        # UPDATE BEST MODEL
        best_test_accuracy = test_acc
        best_epoch = epoch
        
        # CREATE MODELS DIRECTORY 
        MODEL_PATH = Path("models")
        MODEL_PATH.mkdir(parents = True, exist_ok = True)
        
        # CREATE MODEL SAVE PATH
        MODEL_NAME = f"LSTM_Classification_model_{model_number}_Epoch_{epoch}_TestAcc_{test_acc:.2f}_TrainAcc_{avg_acc:.2F}"
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
        
        # SAVE MODEL STATE DICT
        print(f"Saving model to: {MODEL_SAVE_PATH}")
        torch.save(obj = model.state_dict(), f = MODEL_SAVE_PATH)
    
    #print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f} ")
    print(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}" )


    if avg_loss == 100:
        break

ac = pd.DataFrame(y_train_tensor[:,0].to("cpu").numpy(), columns = ["train_label"]).iloc[-batch_size:,].reset_index(drop = True)
ac1 = pd.DataFrame(pred.to("cpu").numpy(), columns = ["preds"]).reset_index(drop = True)
ac = ac.merge(ac1, left_index = True, right_index = True)



