# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:09:00 2024

@author: ktsar
"""

import datetime
import joblib

import os 
from datetime import datetime
directory = "C:/Users/ktsar/Downloads/Python codes/Python codes/Git_Repos/ATS_Development/Strat_2"
os.chdir(directory)
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim 

from pathlib import Path
from ALGO_KT1 import Preprocessing_functions as pf 
from ALGO_KT1 import LSTM_Architecture as ls
from torch.utils.data import DataLoader #, TensorDataset

ticker = "SPY"
time_period = "360mo"

### LOAD FEAT LIST TO ORDER THE DATA ###
FEAT_PATH = f"model_features/{ticker}/"
print(os.listdir(FEAT_PATH))
idx = 0 if len(os.listdir(FEAT_PATH)) < 2 else int(input("Select file index (e.g. 0,1,2)"))
FEAT_NAME = os.listdir(FEAT_PATH)[idx]
MODEL_FEAT = pd.read_csv(FEAT_PATH + FEAT_NAME)['0'].to_list()

# LOAD DF FOR MODEL BUILDING TO CHECK DATE RANGES 
DF_PATH = f"data/{ticker}/"
print("DataFrame for model building: ", os.listdir(DF_PATH))
idx = 0 if len(os.listdir(DF_PATH)) < 2 else int(input("Select file index: "))
DF_NAME = os.listdir(DF_PATH)[idx] 
print("Chosen DataFrame file: ", DF_NAME)
df_dates = pd.read_parquet(DF_PATH + DF_NAME)
df_dates = pf.format_idx_date(df_dates)

# LOAD LSTM MODEL STATE DICT  
MODEL_PATH = f"lstm_models/{ticker}/"
print(os.listdir(MODEL_PATH))
idx = 0 if len(os.listdir(MODEL_PATH)) < 2 else int(input("Select file index: "))
MODEL_NAME = os.listdir(MODEL_PATH)[idx]
print("Chosen LSTM, MODEL file: ", MODEL_NAME)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pf.downlaod_symbol_data(ticker, period = time_period)
df = pf.create_momentum_feat(df, ticker)
df = pf.technical_indicators(df).dropna()
df = pf.format_idx_date(df)

# REMOVE DATA SNOOPING 
out_sample = True

if out_sample is True:
    start_date = df_dates.index.min()
    df = df[df.index <= start_date]
    #df = df[df.index >= '2010-01-01']
    
    del DF_NAME, df_dates 


seq_length =  1
df = df.sort_index(ascending = False)
df['labels'] = (df['open_close'] >=0).astype(int)

# preserve the price features to use in the backtest data
drop_cols = ['Open', 'High', 'Low', 'Close', 'Stock Splits']
df1 = df[drop_cols]



df = df[MODEL_FEAT]
df_model = df.copy()

assert list(df_model.columns) == MODEL_FEAT

## SCALING THE DATA BEFORE CONVERTING IT INTO SUITABLE INPUT FOR RNN 
df_model = pf.min_max_scaling(df_model)

df_model.columns = ['labels'] + MODEL_FEAT[0:-1]

df_model = df_model.sort_index(ascending = True)

del drop_cols

X, y  = pf.create_multivariate_rnn_data(df_model, seq_length)
del y



############################ PREDICTION #######################################


X_tensor = torch.from_numpy(X).type(torch.float).to(device).squeeze(0)

input_feat = df_model.shape[1]
hidden_size = 32
num_layers = 2 
num_classes = 2

# INSTANTIATE MODEL
model = ls.LSTM(input_size=input_feat, 
             output_size=num_classes, 
             hidden_size=hidden_size, 
             num_layers=num_layers,
             device=device).to(device)

# LOAD LSTM MODEL STATE DICT  
model.load_state_dict(torch.load(f = MODEL_PATH + MODEL_NAME ))
del MODEL_PATH, MODEL_NAME

#### PREDICTION #### 
model.eval()

with torch.inference_mode():

    output = model(X_tensor)
    pred = torch.softmax(output, dim = 1).argmax(dim = 1)

## possible mistake in creating the predictions df - dates might not align properly
predictions = pd.DataFrame(pred.to("cpu").numpy(), columns = ["predictions"], index = df_model.index[:-1])

df2 = df_model.copy()

df2 = df2.merge(predictions, left_index = True, right_index = True)
df1 = df1.merge(df2, left_index = True, right_index = True)

del pred, output, predictions

# cluster_stats = pd.read_csv(STATS_PATH + STATS_NAME).set_index("Unnamed: 0")
ACC = (df1['labels'] == df1['predictions']).sum() / df1.shape[1]



######## BACKTESTING #########
import numpy as np

df1 = df1.sort_index()
df1_cols = [i for i in df1.columns if "mom" not in i]
df1 = df1[df1_cols]
del df1_cols

df1 = df1.sort_index(ascending = True)

TC = 3 

start_capital = 1e4
df = pd.DataFrame()
half_kelly = pf.kelly_criterion(ticker, df1.index.min(), period = "360mo") / 2 

for date, row in df1.iterrows():
    
    print(date, row)
    half_kelly = pf.kelly_criterion(ticker, date, period = "360mo") / 2 
    
    row['half_kelly'] = half_kelly
    row['shares'] = (start_capital * half_kelly) // row['Close'] ## you need to divide cluster stats from target with USO - check clusters stats df for % or decimals 
    row['return'] = row['Close'] / row['Open'] - 1 
    row['pnl'] = np.where(row['predictions'] == 1, round(start_capital* half_kelly * row['return'], 2), 0).astype("float64")
    row['eod_capital'] = start_capital + row['pnl'].item()
    
    start_capital = row['eod_capital']
    
    row = row.to_frame()

    df = pd.concat([df, row], axis =1)
    # break 
   # df['date'] = date


df = df.transpose()
df = df.infer_objects()


import matplotlib.pyplot as plt 
plt.figure(figsize = (10,7))
plt.plot(df.index, df['eod_capital'], color = 'b')
plt.xlabel('Date')
plt.ylabel("USD")













