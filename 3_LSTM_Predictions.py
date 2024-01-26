# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:14:23 2024

@author: ktsar
"""
import os 
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn

from mpl_toolkits import mplot3d
from Preprocessing_functions import *;
from scipy.stats import skew, norm, kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from LSTM_Architecture import LSTM

ticker = "XLU"

df = downlaod_symbol_data(ticker) # period = "1day"
df = create_momentum_feat(df, ticker)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date']).dt.date
#df = df.set_index("Date")

### LOAD KMEANS MODEL ###
data = df[["open_low", "open_close", "gap"]].dropna()
model_filename = "kmeans_model_XLU_3_clusters.joblib"
loaded_kmeans = joblib.load(model_filename)
k_predictions = pd.DataFrame(loaded_kmeans.predict(data), columns = ["labels"])
data = data.merge(k_predictions, left_index = True, right_index = True)


df_model = merge_dfs(data, df, ticker)
df_model = df_model.set_index("Date")

end_date = df_model.index.max()
df_model['last_day'] = (df_model.index == end_date).astype(int)
del df, data

seq_length =  1
#test_size_pct = 0.15

df_model = df_model.sort_index(ascending = False)

## SCALING THE DATA BEFORE CONVERTING IT INTO SUITABLE INPUT FOR RNN 
df_model = min_max_scaling(df_model)

X, y  = create_multivariate_rnn_data(df_model, seq_length)

X1 = X[0]

input_feat = df_model.shape[1]
hidden_size = 32
num_layers = 2 
learning_rate = 0.01
epochs =  3000
num_classes = 3
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# INSTANTIATE MODEL
model = LSTM(input_size=input_feat, 
             output_size=num_classes, 
             hidden_size=hidden_size, 
             num_layers=num_layers,
             device=device).to(device)

# LOAD LSTM MODEL STATE DICT  
MODEL_LOC = "C:/Users/ktsar/Downloads/Python codes/Python codes/LSTM_Model_4_Class_XLU_SL_1_LAST_DAY_FEAT/LSTM_Classification_model_4_Epoch_2709_TestAcc_81.07_TrainAcc_82.03"

model.load_state_dict(torch.load(f = MODEL_LOC ))

X_tensor = torch.from_numpy(X).type(torch.float).to(device).unsqueeze(0)

#X_tensor = torch.from_numpy(X).type(torch.float).to(device)#.unsqueeze(0)

model.eval()

with torch.inference_mode():

    output = model(X_tensor)
    pred = torch.softmax(output, dim = 1).argmax(dim = 1)


actions = {
    0 : f"Place a SELL ORDER in {ticker} on the OPEN and aim for a gain of 30 cents",
    1 : f"Plce a SELL ORDER in {ticker} on the OPEN and aim for a gain of 60 cents",
    2 : f"DO NOT TRADE {ticker}"
           }
            
print(actions[pred[0].item()])












