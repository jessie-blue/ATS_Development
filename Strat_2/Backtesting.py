# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:09:00 2024

@author: ktsar
"""

import datetime
import joblib

import os 
from datetime import datetime
#directory = "C:/Users/ktsar/Downloads/Python codes/Python codes/Git_Repos/ATS_Development/Strat_2"
directory = os.getcwd().replace("\\", "/")
os.chdir(directory)
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim 

from pathlib import Path
from ALGO_KT1 import Preprocessing_functions as pf 
from ALGO_KT1 import LSTM_Architecture as ls
from torch.utils.data import DataLoader #, TensorDataset
from techinical_analysis import * 

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

# LOAD BAR STATS 
STATS_PATH = f"stats/{ticker}/"
idx = 0 if len(os.listdir(STATS_PATH)) < 2 else int(input("Select file index: "))
STATS_NAME = os.listdir(STATS_PATH)[idx]
print("Chosen STATS file: ", STATS_NAME)
stats = pd.read_csv(STATS_PATH + STATS_NAME, index_col = ['Unnamed: 0'])

del FEAT_PATH, idx, DF_PATH, FEAT_NAME, directory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pf.downlaod_symbol_data(ticker, period = time_period)
df = pf.create_momentum_feat(df, ticker)
df = pf.technical_indicators(df).dropna()
df = reversal_patterns(df)
df = continuation_patterns(df)
df = magic_doji(df)
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

# Labels
#df['labels'] = (df['open_close'] >=0).astype(int) # this might be wrong!
# labels used in the Data_Acquistion
df['labels'] = ((df['Close'] - df['Open']) >= 0).astype(int) 

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

######## INFER HIGH OR LOW WAS FIRST ######
### NOT SURE WE NEED THIS 
#df1['low_first'] = df1['Close'] >= df1['Open']
#df1['low_last'] = df1['Close'] <= df1['Open'] 

#df1['labels'] = (df1['Close'] > df1['Open']).astype(int)

######## BACKTESTING #########

df1 = df1.sort_index()


# DROPPING Momentum features - not sure why!
#df1_cols = [i for i in df1.columns if "mom" not in i]
#df1 = df1[df1_cols]
#del df1_cols

df1 = df1.sort_index(ascending = True)

TC = 3 

start_capital = 1e4
df = pd.DataFrame()
half_kelly = pf.kelly_criterion(ticker, df1.index.min(), period = "360mo") / 2 

# for date, row in df1.iterrows():
    
#     print(date, row)
#     half_kelly = pf.kelly_criterion(ticker, date, period = "360mo") / 2 
    
#     row['half_kelly'] = half_kelly
#     row['shares'] = (start_capital * half_kelly) // row['Close'] ## you need to divide cluster stats from target with USO - check clusters stats df for % or decimals 
#     row['return'] = row['Close'] / row['Open'] - 1 
#     row['pnl'] = np.where(row['predictions'] == 1, round(start_capital* half_kelly * row['return'], 2), 0).astype("float64")
#     row['eod_capital'] = start_capital + row['pnl'].item()
    
#     start_capital = row['eod_capital']
    
#     row = row.to_frame()

#     df = pd.concat([df, row], axis =1)
#     # break 
#    # df['date'] = date



# for date, row in df1.iterrows():
    
#     print(date, row)
#     half_kelly = pf.kelly_criterion(ticker, date, period = "360mo") / 2 
    
#     row['half_kelly'] = half_kelly
#     row['shares'] = (start_capital * half_kelly) // row['Close'] ## you need to divide cluster stats from target with USO - check clusters stats df for % or decimals 
#     # row['return'] = row['Close'] / row['Open'] - 1 
    
#     row['pnl_green'] = start_capital * stats.loc['median', 'open_high_green'] / 100 if row['predictions'] == 0 and row['low_last'] else start_capital * stats.loc['median', 'open_low_green']  
#     row['pnl_0'] = start_capital * stats.loc['median', 'open_high_red'] / 100 if row['predictions'] == 1 and row['low_last'] else start_capital * stats.loc['median', 'open_low_red']  
    
#     row['pnl'] = np.where(row['predictions'] == 1, round(start_capital* half_kelly * row['return'], 2), 0).astype("float64")
#     row['eod_capital'] = start_capital + row['pnl'].item()
    
#     start_capital = row['eod_capital']
    
#     row = row.to_frame()

#     df = pd.concat([df, row], axis =1)
#     # break 
#    # df['date'] = date
end_date_for_test = df1.head(200).index.max()


for date, row in df1.iterrows():
    
    #print(date, row)
    print(date)
    half_kelly = pf.kelly_criterion(ticker, date, period = "360mo") / 2 
    row['half_kelly'] = half_kelly
    row['shares'] = (start_capital * half_kelly) // row['Open'] ## you need to divide cluster stats from target with USO - check clusters stats df for % or decimals 
    row['pnl'] = (row['Open'] - row['Close']) * row['shares'] if row['predictions'] == 0 else (row['Close'] - row['Open']) * row['shares']
    row['pnl'] -= 3 
    
    # Capital adjustments 
    row['eod_capital'] = start_capital + row['pnl'].item()
    
    row['eod_equity'] = start_capital + row['pnl']
    row['daily_ret'] = row['eod_equity'] / start_capital - 1
    start_capital = row['eod_capital']
    
    
    # Add the row to the new df 
    row = row.to_frame()
    df = pd.concat([df, row], axis =1)
    
    ### only for testing 
    #if date == end_date_for_test:
    #    break
    
    # LEGACY CODE 
    #row['return'] = row['Close'] / row['Open'] - 1 
    #capital = start_capital * half_kelly
    #row['usd_return'] = row['return'] * capital
    
    #row['pnl'] = np.where(row['predictions'] == 1, row['usd_return']*(-1), row['usd_return'])
    #row['eod_capital'] = start_capital + row['pnl'].item()
    
    #start_capital = row['eod_capital']
    
    #row = row.to_frame()

    #df = pd.concat([df, row], axis =1)
    
    
    # break 
   # df['date'] = date



df = df.transpose()
#df = df.infer_objects()

df1 = df.copy()

#####   MAX DRAWDOWN
from calculateMaxDD import calculateMaxDD

cum_ret = np.cumprod(1+ df1['daily_ret']) - 1
maxDrawdown, maxDrawdownDuration, startDrawdownDay=calculateMaxDD(cum_ret.values)

#####   SHARPE RATIO
sharpe_ratio = round(np.sqrt(252) * np.mean(df1['daily_ret']) / np.std(df1['daily_ret']),2)

#####   AVG YEARLY RETURN AND STD
mean_ret = df1['daily_ret'].mean() * 252
std = df1['daily_ret'].std()*np.sqrt(252)

import numpy as np
p_change = df1['Close'].pct_change().dropna() #/ df1['Close'].shift(1)
corr = np.corrcoef(p_change, df1['Close'][1:])

print(f"Correlation Price / Return: " , round(corr[1][0], 2))
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Maximum Drawdown: {round(maxDrawdown,4)}')
print(f'Max Drawdown Duration: {maxDrawdownDuration} days' )
print(f'Start day Drawdown: {startDrawdownDay}')
print(f"Average Yearly Return: {round(mean_ret*100, 2)} %")


#
# PLOTTING
#

# Create figure and axis objects
plt.rcParams.update({'font.size': 12})

fig, ax1 = plt.subplots(figsize=(10, 7))
plt.title(f"Green / Red Bar Prediction Strategy - {ticker}")

# Plot data on the first y-axis
ax1.plot(df1.index, df1['Close'], 'g-', alpha = 0.5)
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price ', color='g')

# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(df1.index, df1['eod_equity'], 'b-')
ax2.set_ylabel('Equity USD', color='b')

# Add black dotted line at y=0
#ax1.axhline(y=0, color='k', linestyle='--')
ax2.axhline(y=1e4, color='k', linestyle='--')

#Remove box lines around the chart area
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Add text box
stats_text = f'Sharpe Ratio: {sharpe_ratio} :\n'
stats_text += f'Maximum Drawdown: {round(maxDrawdown*100,2)}% \n'
stats_text += f'Start day Drawdown: {startDrawdownDay} day \n'
stats_text += f"Drawdown Duration: {int(maxDrawdownDuration)} days \n"
stats_text += f"Average Yearly Return: {round(mean_ret*100, 2)} % \n"
stats_text += f"Average Yearly STD: {round(std*100, 2)} % \n"
fig.text(0.1, 0.03, stats_text, fontsize=12,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(facecolor='white', alpha=0.5,edgecolor='none'))





save = False
if save is True:
    plt.savefig(f"Green- Red Bar Prediction/Backtest_{ticker}_hk{half_kelly}_V2", bbox_inches='tight')

plt.show()

import matplotlib.pyplot as plt 
plt.figure(figsize = (10,7))
plt.plot(df.index, df['eod_capital'], color = 'b')
plt.xlabel('Date')
plt.ylabel("USD")



acc = (df1['labels'] == df1['predictions']).sum() / df.shape[1]










