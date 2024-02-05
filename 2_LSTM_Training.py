# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:24:36 2024

@author: ktsar
"""
import os
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim 

from datetime import datetime 
from pathlib import Path
from Preprocessing_functions import min_max_scaling, create_multivariate_rnn_data, accuracy_fn
from torch.utils.data import DataLoader #, TensorDataset
from LSTM_Architecture import LSTM, TimeSeriesDataset

ticker = "BTC-USD"
DF_NAME = "df_BTC-USD_k3_202402021844.parquet"
#DF_NAME = f"df_{ticker}_k3_202401251838.parquet"
FILE_PATH = f"Data/{ticker}/"
FILE_PATH_NAME = FILE_PATH + DF_NAME

df_model = pd.read_parquet(FILE_PATH_NAME)
df_model = df_model.reset_index()
df_model['Date'] = pd.to_datetime(df_model['Date']).dt.date
df_model = df_model.set_index("Date")


end_date = df_model.index.max()
seq_length =  1
test_size_pct = 0.15

df_model = df_model.sort_index(ascending = False)

model_feat = pd.DataFrame(list(df_model.columns) + ["last_day"])

df_model = min_max_scaling(df_model)

df_model['last_day'] = (df_model.index == end_date).astype(int)

X, y  = create_multivariate_rnn_data(df_model, seq_length)

# Train, Test Split 
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
learning_rate = 0.1
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


### CHECKING INPUT DIMENSIONS
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

# TRAIN AND TEST MODEL (NEEDS TO BE REFACTORED IN FNS)
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
        DATE = datetime.today().strftime('%Y%m%d%H%M')
        MODEL_PATH = Path(f"lstm_models/{ticker}")
        MODEL_PATH.mkdir(parents = True, exist_ok = True)
        
        # CREATE MODEL SAVE PATH
        MODEL_NAME = f"LSTM_Class_{DF_NAME.replace('.parquet','')}_Epoch_{epoch}_TestAcc_{test_acc:.2f}_TrainAcc_{avg_acc:.2F}_{DATE}"
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
        
        # SAVE THE INPUT FEATURES OF THE MODEL
        FEAT_NAME = f"LSTM_{DF_NAME.replace('.parquet','')}_NFEAT{model_feat.shape[0]}.csv"
        if FEAT_NAME not in os.listdir():
            FEAT_SAVE_PATH = MODEL_PATH / FEAT_NAME
            model_feat.to_csv(FEAT_NAME, index = False)
        
        # SAVE MODEL STATE DICT
        print(f"Saving model to: {MODEL_SAVE_PATH}")
        torch.save(obj = model.state_dict(), f = MODEL_SAVE_PATH)
    
    #print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f} ")
    print(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}" )


    if avg_loss == 100:
        break
    



