

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

from ALGO_KT1 import Preprocessing_functions as pf 
#from ALGO_KT1.Preprocessing_functions import *;
from ALGO_KT1.LSTM_Architecture import LSTM
from techinical_analysis import *;

cwd = os.getcwd().replace("\\", "/"  )
os.chdir(cwd)

tickers = ["XLU","SPY"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prediction_date = input("Choose date to predict for: today or YYYY-MM-DD: ")

#prediction_date = "2024-02-13"


### for ticker in tickers from here
ticker = "SPY"

#df = pf.downlaod_symbol_data(ticker)

# Alternative to yfinance
df = pf.download_data(ticker, days=365).sort_index()
df['Dividends'] =  0

df = pf.create_momentum_feat(df, ticker)
df = pf.technical_indicators(df, MA_DIVERGENCE = True).dropna()
df = reversal_patterns(df)
df = continuation_patterns(df)
df = magic_doji(df)
df = pf.format_idx_date(df)


if prediction_date != "today":
    #date = "2024-02-29"
    df = df[df.index < prediction_date]
    
    
df['labels'] = ((df['Close'] - df['Open']) >= 0).astype(int) 
df['open_high'] = df['open_high'] * (-1)

df = df.tail(48)

print(f"0 - red bar n/ 1 - green bar",df['labels'].value_counts())
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

df1 = df.copy()

end_date = df1.index.max()
df1['last_day'] = (df1.index == end_date).astype(int)
df1 = df1[MODEL_FEAT].dropna()

seq_length =  1
df1 = df1.sort_index(ascending = False)

## SCALING THE DATA BEFORE CONVERTING IT INTO SUITABLE INPUT FOR RNN 
df_model = pf.min_max_scaling(df1)

X, y  = pf.create_multivariate_rnn_data(df_model, seq_length)

# X = X[0]
#X_tensor = torch.from_numpy(X[0]).type(torch.float).to(device).unsqueeze(0)
X_tensor = torch.from_numpy(X).type(torch.float).to(device).squeeze(0)
del FEAT_PATH, idx, FEAT_NAME, MODEL_FEAT, end_date, y, X

# LOAD LSTM MODEL STATE DICT  
MODEL_PATH = f"lstm_models/{ticker}/"
lstm_files = os.listdir(MODEL_PATH)
print(lstm_files)
idx = 0 if len(lstm_files) < 2 else int(input("Select file index: "))
MODEL_NAME = os.listdir(MODEL_PATH)[idx]
print("Chosen LSTM, MODEL file: ", MODEL_NAME)

input_feat = df_model.shape[1]
hidden_size = 32
num_layers = 2 
num_classes = 2
    
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
    

if pred[0].item() == 0:
    action = 'SELL'

if pred[0].item() == 1:
    action = 'BUY'


print(f"Symbol: {ticker}: {action}")


strat_name = 'Strat_2'

ATS_PATH = cwd.replace('/Strat_2', '')
capital = pd.read_csv(ATS_PATH + '/strategies.csv')
capital = capital[capital['strategy_name'] == strat_name]
capital = capital[capital['symbol'] == ticker]
#capital['current_capital'] = capital['current_capital'].astype(float)
capital = capital['current_capital'].item()

kelly = pf.kelly_criterion(ticker, period='6mo', date_to=df.index.max()) / 2
bp_used = round(capital * kelly, 2)
last_close_price = df['Close'].iloc[-1]
n_shares = int(bp_used // last_close_price) 
open_position_price = 'at_open'
target_price = 'at_close'
stop_price = 'at_close'

stats_dir = f'stats/{ticker}'
statsfile = os.listdir(stats_dir)[0]
expected_return = pd.read_csv(f'stats/{ticker}/{statsfile}', index_col='Unnamed: 0')
if action == 'BUY':
    expected_return = expected_return['open_close_green']['median'] * (-1)
    
else:
    expected_return = expected_return['open_close_red']['median'] * (-1)


orders = {
        "strat" : strat_name,
        "ticker" : ticker,
        "direction" : action, # Feed that from the predictions script 
        "last_close_price" : df['Close'].iloc[-1],
        "capital" : round(capital, 2), # to be determined by a portfolio optimization engine
        "half_kelly" : kelly,
        "bp_used" : bp_used,
        "n_shares" : n_shares ,
        "open_position": open_position_price,
        "target_price"  : target_price,
        "expected_return" : expected_return,
        "stop_price" : stop_price
        }


orders = pd.DataFrame(orders, columns = orders.keys(), index = [1] )

if prediction_date == "today":
    date = datetime.today().strftime('%Y_%m_%d')
else:
    date = (df.index.max() + timedelta(days = 1)).strftime('%Y_%m_%d')

FILE_PATH = cwd.replace("Strat_2", "orders/")
FILENAME = "Orders_" + date.replace("-", "_") + ".csv"

if FILENAME not in os.listdir(FILE_PATH):
    orders.to_csv(FILE_PATH + FILENAME, index = False)
    #continue 

orders_file = pd.read_csv(FILE_PATH + FILENAME)

orders_file = pd.concat([orders_file, orders], axis = 0).reset_index(drop = True)

orders_file.to_csv(FILE_PATH + FILENAME, index = False)

print(f"file saved in ", FILE_PATH + FILENAME)