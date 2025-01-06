
import os 
import datetime
import numpy as np
import pandas as pd 
from ALGO_KT1.Preprocessing_functions import *;
from datetime import timedelta

cwd = os.getcwd().replace("\\", "/"  )

# function parameters to fill in 
prediction_date = datetime.date.today()

ticker = 'SPY'
strat_name = 'Strat_2'
action = 'output from LSTM'

capital = pd.read_csv('strategies.csv')
capital = capital[capital['strategy_name'] == strat_name]
capital = capital[capital['symbol'] == ticker]
capital = capital['current_capital'].iloc[0]

kelly = 1 #kelly_criterion(ticker, period='6mo')
bp_used = capital * kelly
last_close_price = 400 # need to add this from Yahoo or from API
n_shares = int(bp_used // last_close_price) 
open_position_price = 'at_open'
target_price = 'at_close'
expected_return = 'nan - the average green bar'
stop_price = 'at_close'


def create_orders(ticker,
                  strat_name,
                  action,
                  df):



    orders = {
        "strat" : strat_name,
        "ticker" : ticker,
        "direction" : action, # Feed that from the predictions script 
        "last_close_price" : df['Close'].iloc[-1],
        "capital" : capital, # to be determined by a portfolio optimization engine
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