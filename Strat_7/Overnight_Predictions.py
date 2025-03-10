

import os
from datetime import datetime, timedelta 
import numpy as np 
import pandas as pd 
import joblib
import sys
from sklearn.ensemble import RandomForestRegressor

directory = os.getcwd()

cwd = directory.replace('Strat_7', 'module_1')
#os.chdir(cwd)
sys.path.append(cwd)

import Preprocessing_functions as pf 
import calculateMaxDD 
#os.chdir(directory)

from ..module_1.Preprocessing_functions import * 
from ..module_1 import Preprocessing_functions as pf  


ticker = 'SPY'

df = pf.downlaod_symbol_data(ticker, period= '120mo')
try:
    df = df.drop(columns=['Stock Splits', 'Dividends', 'Capital Gains'])
except KeyError:
    print("Columns not available (see above line of code)")
    
df = pf.create_momentum_feat(df, symbol=ticker) ### need to inspect in more detail how the create momemntum features work and the shift in this case
df = pf.technical_indicators(df,MA_DIVERGENCE=True)
df = pf.format_idx_date(df)

#df['overnight_pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

df = df.dropna()

df.head()

########
prediction_date = input()

if prediction_date != "today":
    #date = "2024-02-29"
    df = df[df.index < prediction_date]

#######

features = pd.read_csv(cwd.replace('\module_1', f'\Strat_7\model_features\{ticker}_features_30.csv'))
features = features['features'].to_list()

X = df[features].tail(1)

rf_predictor = joblib.load(directory + "\models" + f"\{ticker}_overnight_regression_random_forest.pkl")

X['prediction'] = rf_predictor.predict(X)
X['action'] = np.where(X['prediction'] > 0, 'BUY', 'SELL')

print(f"{ticker}: " ,X['action'].item())


##################################
### CREATE A FILE WITH THE ORDERS
##################################

#kelly = pf.kelly_criterion(ticker, period = "6mo")
kelly = 2
strat = 'Strat_7' # this was changes from 'Short_Open' in case smtng breaks
symbol = ticker
last_price = df['Close'].iloc[-1]
capital = 1 #strats['current_capital'].item()
half_kelly = kelly / 2
#if half_kelly < 1: removed this on 27.11.2024 to allow for less size to be used
#    half_kelly = 1 
bp_used = round(capital * half_kelly,2)
n_shares = int(bp_used // last_price) 
open_position_price = 'at_close_today'
target_price = last_price * (1 + X['prediction'].item())
exp_ret = (target_price - last_price) / last_price 
stop_price = 'tomorrow_open'


orders = {
    "strat" : strat,
    "ticker" : ticker,
    "direction" : X['action'].item(),
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

    
FILE_PATH = directory.replace("Strat_7", "orders/Testing/")
FILENAME = "Orders_" + date.replace("-", "_") + ".csv"

if FILENAME not in os.listdir(FILE_PATH):
    orders.to_csv(FILE_PATH + FILENAME, index = False)
    #continue 
    

orders_file = pd.read_csv(FILE_PATH + FILENAME)

orders_file = pd.concat([orders_file, orders], axis = 0).reset_index(drop = True)

orders_file.to_csv(FILE_PATH + FILENAME, index = False)




import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(cwd)
