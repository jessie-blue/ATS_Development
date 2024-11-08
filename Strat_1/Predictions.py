# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:37:37 2024

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

from datetime import datetime, timedelta 
from mpl_toolkits import mplot3d
from scipy.stats import skew, norm, kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

try:
    from Preprocessing_functions import *;
    from LSTM_Architecture import LSTM
    
except ModuleNotFoundError:
    from Strat_1.Preprocessing_functions import *;
    from Strat_1.LSTM_Architecture import LSTM

cwd = os.getcwd().replace("\\", "/"  )
os.chdir(cwd)

tickers = ["XLU", "USO", "XLI", "AMLP", "SPY"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prediction_date = input("Choose date to predict for: today or YYYY-MM-DD: ")

# prediction_date = "2024-02-13"
# ticker = "XLU"

for ticker in tickers: 
    
    # =============================================================================
    # PULL DATA FROM DB
    # =============================================================================
    df = downlaod_symbol_data(ticker) # period = "1day"
    df = create_momentum_feat(df, ticker)
    df = format_idx_date(df)
    
    if prediction_date != "today":
        #date = "2024-02-29"
        df = df[df.index < prediction_date]
    
    # =============================================================================
    # LOAD KMEANS MODEL FOR LABELLING 
    # =============================================================================
    ### LOAD KMEANS MODEL ###
    KMEANS_PATH = f"kmeans_models/{ticker}/"
    print(os.listdir(KMEANS_PATH))
    idx = 0 if len(os.listdir(KMEANS_PATH)) < 2 else int(input("Select file index: "))
    KMEANS_NAME = os.listdir(KMEANS_PATH)[idx]
    print("Chosen K_MEANS MODEL file: ", KMEANS_NAME)
    FILE = KMEANS_PATH + KMEANS_NAME
    loaded_kmeans = joblib.load(FILE)
    del KMEANS_PATH, KMEANS_NAME, idx, FILE
    
    ### ASSIGN CLUSTER TO OBSERVATION
    data = df[["open_low", "open_close", "gap"]].dropna()
    k_predictions = pd.DataFrame(loaded_kmeans.predict(data), columns = ["labels"], index = data.index)
    
    df_model = df.merge(k_predictions, left_index = True, right_index = True)
    del data, k_predictions, loaded_kmeans
    # =============================================================================
    # DATA TRANSFORMATION  - SCALING AND LSTM FORMATTING 
    # =============================================================================
    ### LOAD FEAT LIST TO ORDER THE DATA ###
    FEAT_PATH = f"model_features/{ticker}/"
    print(os.listdir(FEAT_PATH))
    idx = 0 if len(os.listdir(FEAT_PATH)) < 2 else int(input("Select file index (e.g. 0,1,2)"))
    FEAT_NAME = os.listdir(FEAT_PATH)[idx]
    print('Selected Feature list: ', FEAT_NAME)
    MODEL_FEAT = pd.read_csv(FEAT_PATH + FEAT_NAME)['0'].to_list()
    
    end_date = df_model.index.max()
    df_model['last_day'] = (df_model.index == end_date).astype(int)
    df_model = df_model[MODEL_FEAT].dropna()
    
    # MIGHT NOT BE REQUIRED (seq - lenght)
    seq_length =  1
    df_model = df_model.sort_index(ascending = False)
    
    ## SCALING THE DATA BEFORE CONVERTING IT INTO SUITABLE INPUT FOR RNN 
    df_model = min_max_scaling(df_model)
    
    X, y  = create_multivariate_rnn_data(df_model, seq_length)
    
    # X = X[0]
    #X_tensor = torch.from_numpy(X[0]).type(torch.float).to(device).unsqueeze(0)
    X_tensor = torch.from_numpy(X).type(torch.float).to(device).squeeze(0)
    del FEAT_PATH, idx, FEAT_NAME, MODEL_FEAT, end_date, y, X
    # =============================================================================
    # LOAD LSTM MODEL TO PREDICT 
    # =============================================================================
    
    # LOAD LSTM MODEL STATE DICT  
    MODEL_PATH = f"lstm_models/{ticker}/"
    print(os.listdir(MODEL_PATH))
    idx = 0 if len(os.listdir(MODEL_PATH)) < 2 else int(input("Select file index: "))
    MODEL_NAME = os.listdir(MODEL_PATH)[idx]
    print("Chosen LSTM, MODEL file: ", MODEL_NAME)
    
    input_feat = df_model.shape[1]
    hidden_size = 32
    num_layers = 2 
    num_classes = 3
    
    # INSTANTIATE MODEL
    model = LSTM(input_size=input_feat, 
                 output_size=num_classes, 
                 hidden_size=hidden_size, 
                 num_layers=num_layers,
                 device=device).to(device)
    
    model.load_state_dict(torch.load(f = MODEL_PATH + MODEL_NAME ))
    
    del MODEL_PATH, idx, MODEL_NAME
    # PREDICTION 
    model.eval()
    
    with torch.inference_mode():
    
        output = model(X_tensor)
        pred = torch.softmax(output, dim = 1).argmax(dim = 1)
    
    
    # Cluster stats
    STATS_PATH = f"Data/{ticker}/k_stats/"
    print("KMEANS Stats files: ", os.listdir(STATS_PATH))
    idx = 0 if len(os.listdir(STATS_PATH)) < 2 else int(input("Select file index: "))
    STATS_NAME = os.listdir(STATS_PATH)[idx]
    print("Chosen K_STATS file: ", STATS_NAME)
    cluster_stats = pd.read_csv(STATS_PATH + STATS_NAME).set_index("Unnamed: 0")
    
    del STATS_PATH, idx, STATS_NAME
    
    n_clusters = 3 
    
    actions = {}
    
    for cluster in range(n_clusters):
    
        mean_profit = cluster_stats.loc["mean", f"open_low_{cluster}"]
        mean_loss = cluster_stats.loc["mean", f"open_close_{cluster}"]
        
        if mean_profit > mean_loss and mean_loss > 0:
            # actions[cluster] = f"Place a SELL ORDER in {ticker} on the OPEN. Profit target: {mean_profit} pct"
            actions[cluster] = f"SELL"
        
        else:
            # actions[cluster] = f"DO NOT TRADE {ticker}"
            actions[cluster] = f"HOLD"
    
    
    print(ticker, actions[pred[0].item()])
    
    predictions = pd.DataFrame(pred.to("cpu").numpy(), columns = ["predictions"])
    
    # =============================================================================
    # Create a file with positions for the day  
    # =============================================================================
    
   # if "SELL" in actions[pred[0].item()]:
    strats_path = cwd.replace("Strat_1", "")
    strats = pd.read_csv(strats_path + "/strategies.csv")
    
    strats = strats[strats['symbol'] == ticker]
    
    kelly = kelly_criterion(ticker, period = "6mo")
    
    strat = 'Short_Open'
    symbol = ticker
    last_price = df['Close'].iloc[-1]
    capital = strats['current_capital'].item()
    half_kelly = kelly / 2
    if half_kelly < 1:
        half_kelly = 1 
    bp_used = round(capital * half_kelly,2)
    n_shares = int(bp_used // last_price) 
    open_position_price = 'at_open'
    target_price = 1 - cluster_stats.loc["median", f"open_low_{pred[0].item()}"] / 100
    exp_ret = 1 - target_price
    stop_price = 'at_close'
    
    
    orders = {
        "strat" : strat,
        "ticker" : ticker,
        "direction" : actions[pred[0].item()],
        "last_close_price" : df['Close'].iloc[-1],
        "capital" : capital, # to be determined by a portfolio optimization engine
        "half_kelly" : half_kelly,
        "bp_used" : bp_used,
        "n_shares" : n_shares ,
        "open_position": open_position_price,
        "target_price"  : target_price,
        "expected_return" : round(exp_ret, 6),
        "stop_price" : stop_price
        }
    
    orders = pd.DataFrame(orders, columns = orders.keys(), index = [1] )
    
    if prediction_date == "today":
        date = datetime.today().strftime('%Y_%m_%d')
    else:
        date = (df.index.max() + timedelta(days = 1)).strftime('%Y_%m_%d')
    
    FILE_PATH = cwd.replace("Strat_1", "orders/")
    FILENAME = "Orders_" + date.replace("-", "_") + ".csv"
    
    if FILENAME not in os.listdir(FILE_PATH):
        orders.to_csv(FILE_PATH + FILENAME, index = False)
        continue 

    orders_file = pd.read_csv(FILE_PATH + FILENAME)
    
    orders_file = pd.concat([orders_file, orders], axis = 0).reset_index(drop = True)
    
    orders_file.to_csv(FILE_PATH + FILENAME, index = False)










