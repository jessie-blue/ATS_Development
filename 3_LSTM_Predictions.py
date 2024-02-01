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
n_clusters = 3 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = downlaod_symbol_data(ticker) # period = "1day"
df = create_momentum_feat(df, ticker)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date']).dt.date
#df = df.set_index("Date")

### LOAD KMEANS MODEL ###
KMEANS_PATH = f"kmeans_models/{ticker}/"
#KMEANS_NAME = f"kmeans_model_df_{ticker}_k{n_clusters}_202401251838.joblib"
KMEANS_NAME = "kmeans_model_df_XLU_k3_202402011414.joblib"
FILE = KMEANS_PATH + KMEANS_NAME
loaded_kmeans = joblib.load(FILE)

### ASSIGN CLUSTER TO OBSERVATION
data = df[["Date","open_low", "open_close", "gap"]].dropna().set_index("Date")
k_predictions = pd.DataFrame(loaded_kmeans.predict(data), columns = ["labels"], index = data.index)
data = data.merge(k_predictions, left_index = True, right_index = True).reset_index()

df_model = merge_dfs(data.set_index("Date"), df.set_index("Date"), ticker)
#df_model = df_model.set_index("Date")

end_date = df_model.index.max()
df_model['last_day'] = (df_model.index == end_date).astype(int)
del df, data

seq_length =  1
df_model = df_model.sort_index(ascending = False)

## SCALING THE DATA BEFORE CONVERTING IT INTO SUITABLE INPUT FOR RNN 
df_model = min_max_scaling(df_model)

X, y  = create_multivariate_rnn_data(df_model, seq_length)

# X = X[0]
#X_tensor = torch.from_numpy(X[0]).type(torch.float).to(device).unsqueeze(0)
X_tensor = torch.from_numpy(X).type(torch.float).to(device).squeeze(0)

input_feat = df_model.shape[1]
hidden_size = 32
num_layers = 2 
#learning_rate = 0.01
#epochs =  3000
num_classes = 3
#batch_size = 32

# INSTANTIATE MODEL
model = LSTM(input_size=input_feat, 
             output_size=num_classes, 
             hidden_size=hidden_size, 
             num_layers=num_layers,
             device=device).to(device)

# LOAD LSTM MODEL STATE DICT  
MODEL_PATH = f"lstm_models/{ticker}/"
#print(os.listdir(f"lstm_models/{ticker}"))
MODEL_NAME = 'LSTM_Class_df_XLU_k3_202402011414_Epoch_1775_TestAcc_85.06_TrainAcc_82.18_202402011426'
interactive = False

if interactive is True:
    MODEL_IDX = int(input('Choose model index:'))
    MODEL_NAME = os.listdir(f"lstm_models/{ticker}")[MODEL_IDX]


model.load_state_dict(torch.load(f = MODEL_PATH + MODEL_NAME ))

# PREDICTION 
model.eval()

with torch.inference_mode():

    output = model(X_tensor)
    pred = torch.softmax(output, dim = 1).argmax(dim = 1)


STATS_PATH = f"Data/{ticker}/"
#print("KMEANS Stats files: ", os.listdir(f"Data/{ticker}"))
STATS_NAME = 'KMEANS_Stats_df_XLU_k3_202402011414.csv'

cluster_stats = pd.read_csv(STATS_PATH + STATS_NAME).set_index("Unnamed: 0")


actions = {}

for cluster in range(n_clusters):

    mean_profit = cluster_stats.loc["mean", f"open_low_{cluster}"]
    mean_loss = cluster_stats.loc["mean", f"open_close_{cluster}"]
    
    if mean_profit > mean_loss and mean_loss > 0:
        actions[cluster] = f"Place a SELL ORDER in {ticker} on the OPEN. Profit target: {mean_profit} pct"
    
    else:
        actions[cluster] = f"DO NOT TRADE {ticker}"
        
            
print(actions[pred[0].item()])

predictions = pd.DataFrame(pred.to("cpu").numpy(), columns = ["predictions"])










