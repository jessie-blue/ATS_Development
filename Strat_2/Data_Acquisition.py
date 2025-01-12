# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:20:27 2024

@author: ktsar
"""

import os 
from datetime import datetime

directory = os.getcwd().replace("\\", "/")
os.chdir(directory)
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim 

from pathlib import Path
from ALGO_KT1 import Preprocessing_functions as pf 
from ALGO_KT1 import LSTM_Architecture as ls
from torch.utils.data import DataLoader #, TensorDataset
from techinical_analysis import * 

ticker = 'XLE'

if ticker != 'BTC-USD':

    df = pf.downlaod_symbol_data(ticker)

else:
    
    try:
        df = pd.read_csv('Strat_2/data/BTC-USD/BTCUSD_15.csv')
    except FileNotFoundError:
        df = pd.read_csv('data/BTC-USD/BTCUSD_15.csv')
    
    del df['Timestamp'], df['datetime.1']
    df = df.rename(columns={'datetime' : 'Date'})
    df = df.set_index('Date')

df = pf.create_momentum_feat(df, ticker)
df = pf.technical_indicators(df).dropna()
df = reversal_patterns(df)
df = continuation_patterns(df)
df = magic_doji(df)
df = pf.add_market_feature('SPY', data = df, time_period = '120mo')

if ticker != 'BTC-USD':
    df = pf.format_idx_date(df)
    
else: 
    df.index = pd.to_datetime(df.index)

df['labels'] = ((df['Close'] - df['Open']) >= 0).astype(int) 
df['open_high'] = df['open_high'] * (-1)

print(f"0 - red bar n/ 1 - green bar",df['labels'].value_counts())
# =============================================================================
# BAR STATS 
# =============================================================================
df_green, green_day_stats = pf.cluster_stats(df, 1, "open_close", "open_high", "open_low")
df_Red, red_day_stats = pf.cluster_stats(df, 0, "open_close", "open_high", "open_low")

green_day_stats.columns = ['open_close_green', "open_high_green", "open_low_green"]
red_day_stats.columns = ['open_close_red', "open_high_red", "open_low_red"]
stats = green_day_stats.merge(red_day_stats, left_index = True, right_index = True)

DATE = datetime.today().strftime('%Y%m%d%H%M')
MODEL_PATH = Path(f"stats/{ticker}")
MODEL_PATH.mkdir(parents = True, exist_ok = True)
FILENAME = f"{ticker}_stats_{DATE}.csv"
if FILENAME not in os.listdir(MODEL_PATH):
    stats.to_csv(MODEL_PATH / FILENAME)
# =============================================================================
# END
# =============================================================================

cols = df.columns[4:]

try:
    df1 = df[cols].dropna().drop(columns = ["Capital Gains", "Stock Splits"])
except KeyError:
    df1 = df[cols].dropna()


end_date = df1.index.max()
seq_length =  1
test_size_pct = 0.30

#df1['last_day'] = (df1.index == end_date).astype(int)
df1 = df1.sort_index(ascending = False)

### make a directory of the symbol if not there 
MODEL_PATH = Path(f"data/{ticker}")
MODEL_PATH.mkdir(parents = True, exist_ok = True)
try:
    df1.to_parquet(f"Strat_2/data/{ticker}/DF_{end_date.strftime('%Y_%m_%d')}.parquet")

except FileNotFoundError:
    df1.to_parquet(f"data/{ticker}/DF_{end_date.strftime('%Y_%m_%d')}.parquet")
except OSError:
    df1.to_parquet(f"data/{ticker}/DF_{end_date.strftime('%Y_%m_%d')}.parquet")

model_feat = pd.DataFrame(list(df1.columns))

df1 = pf.min_max_scaling(df1)

X, y  = pf.create_multivariate_rnn_data(df1, seq_length)

# Train, Test Split 
test_size = int(X.shape[0] * test_size_pct) 
train_size = X.shape[0] - test_size

X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Get dates FROM/TO for training dataset 
train_start_date = df1.head(train_size).index.min()
train_end_date = df1.head(train_size).index.max()

print("Start date of Training Data: ", train_start_date)
print("End date of Training Data: ", train_end_date)

# Convert data to PyTorch tensors AND CHECK DIMENSIONS OF TENSOR AND CHECK THE REQUIRED INPUT FOR LSTM 
X_train_tensor = torch.from_numpy(X_train).type(torch.float)#.unsqueeze(1)
y_train_tensor = torch.from_numpy(y_train.values).type(torch.LongTensor)#.unsqueeze(1)

X_test_tensor = torch.from_numpy(X_test).type(torch.float)#.unsqueeze(1)
y_test_tensor = torch.from_numpy(y_test.values).type(torch.LongTensor)#.unsqueeze(1)


### HYPERPARAMETERS
input_feat = X_train.shape[2]
hidden_size = 32
num_layers = 2 
learning_rate = 0.001
momentum = 0.9
epochs =  int(15e3)
num_classes = 2
batch_size = 32
hidden_size1 = 32
hidden_size2 = 64

train_dataset = ls.TimeSeriesDataset(X_train_tensor, y_train_tensor)
test_dataset = ls.TimeSeriesDataset(X_test_tensor, y_test_tensor)

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


#INSTANTIATE MODEL
base_lstm = True

if base_lstm is True:
    model = ls.LSTM(input_size=input_feat, 
                  output_size=num_classes, 
                  hidden_size=hidden_size, 
                  num_layers=num_layers,
                  device=device).to(device)

else:
    model = ls.LSTM_V3(input_size=input_feat, 
                  output_size=num_classes, 
                  hidden_size1=hidden_size1, 
                  hidden_size2=hidden_size2,
                  num_layers=num_layers,
                  device=device).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), 
                            lr=learning_rate, 
                            momentum = momentum)

torch.manual_seed(42)

best_test_accuracy = 0 
best_epoch = 0 
best_avg_acc = 0

results = {}

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
        
        
        batch_accuracy = pf.accuracy_fn(y_true = y_batch[:,0], y_pred = pred) 
        acc += batch_accuracy
        
        #print(f"Epoch: {epoch}, Batch: {batch_index}, Batch Accuracy: {batch_accuracy:.2f}")
        
        #print(epoch, batch_index)
    
    
    avg_loss = running_loss / len(train_loader) # batch_index
    avg_acc = acc / len(train_loader) #batch_index
    #break
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
            test_acc += pf.accuracy_fn(y_true = y[:,0], 
                                    y_pred = test_pred.argmax(dim = 1)) # may need to add argmax(dim = 1) in here
            
        test_loss /= len(test_loader)
        test_acc  /= len(test_loader)
    
        if test_acc > best_test_accuracy: #or best_avg_acc > avg_acc: # reverse the second condition
            #if test_acc >= 0.68 and best_avg_acc >= 0.68:
            # UPDATE BEST MODEL
            best_test_accuracy = test_acc
            best_epoch = epoch
            
            # CREATE MODELS DIRECTORY 
            DATE = datetime.today().strftime('%Y%m%d%H%M')
            MODEL_PATH = Path(f"lstm_models/{ticker}")
            MODEL_PATH.mkdir(parents = True, exist_ok = True)
            
            # CREATE MODEL SAVE PATH
            MODEL_NAME = f"LSTM_Class_Epoch_{epoch}_TestAcc_{test_acc:.2f}_TrainAcc_{avg_acc:.2F}_{DATE}"
            MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
            
            # SAVE THE INPUT FEATURES OF THE MODEL
            FEAT_PATH = Path(f"model_features/{ticker}")
            FEAT_PATH.mkdir(parents= True, exist_ok= True)
            FEAT_NAME = f"LSTM_{DATE}_NFEAT{model_feat.shape[0]}.csv"
            if FEAT_NAME not in os.listdir(FEAT_PATH):
                FEAT_SAVE_PATH = FEAT_PATH / FEAT_NAME
                model_feat.to_csv(FEAT_SAVE_PATH, index = False)
            
            # SAVE MODEL STATE DICT
            print(f"Saving model to: {MODEL_SAVE_PATH}")
            torch.save(obj = model.state_dict(), f = MODEL_SAVE_PATH)

    results[epoch]  = [(avg_acc, test_acc)]
    #print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f} ")
    print(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}" )

    if avg_loss == 100:
        break
    


### plot training 
res = pd.DataFrame(results.values())
res.columns = ['tuple']

res['train_acc'] = res['tuple'].apply(lambda x : x[0])
res['test_acc'] = res['tuple'].apply(lambda x : x[0])

del res['tuple']
res.index = results.keys()


import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
plt.plot(res['train_acc'], res['test_acc'])
plt.xlabel('Training Accuracy')
plt.ylabel('Test Accuracy')

#y_test[52].value_counts()
#y_test['labels'].value_counts()












