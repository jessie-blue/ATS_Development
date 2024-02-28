# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:20:46 2024

@author: ktsar
"""

import os 
import numpy as np
import pandas as pd 
from datetime import datetime , timedelta 
from Strat_1.Preprocessing_functions import downlaod_symbol_data

date = input("Choose date (today or YYYY_MM_DD, default: today - 1): ")

if date == "today":
    date = datetime.today().strftime("%Y-%m-%d")

if len(date) < 3:
    date = (datetime.today() - timedelta(days = 1)).strftime("%Y_%m_%d")


# dates = pd.date_range('2024-02-14', '2024-02-27')

# for date in dates:
    
#     date = date.strftime("%Y_%m_%d")


FILENAME = f"orders/Orders_{date}.csv"
try:
    file = pd.read_csv(FILENAME)

except FileNotFoundError:
    print(f"Date {date} not a trading day!")

hi_lo = pd.DataFrame()
# print(f"run for loop for date {date}")
for idx, row in file.iterrows():
    
    ticker = row['ticker']
    
    df = downlaod_symbol_data(ticker, period= "1mo")
    df = df[df.index.strftime("%Y_%m_%d") == date]
    
    file.loc[file["ticker"] ==  ticker, "open_position"] = df['Open'].item()
    file.loc[file["ticker"] ==  ticker, "stop_price"] = df['Close'].item()
    
    prices = pd.DataFrame([row['ticker'], df['Low'].item(), df['High'].item()], 
                           index = ['ticker', 'low', 'high']).transpose()
    
    hi_lo = pd.concat([hi_lo, prices], axis = 0)
    del prices 
    
# DTYPES     
hi_lo['low'] = hi_lo['low'].astype("float64")
hi_lo['high'] = hi_lo['high'].astype("float64")
file['stop_price'] = file['stop_price'].astype("float64")
file['target_price'] = file['target_price'].astype("float64")
file['open_position'] = file['open_position'].astype("float64")

file['target_price'] = round(file['open_position'] * file['target_price'],2)

#### MERGE 
file = file.merge(hi_lo, on = 'ticker')
try:
    file['pnl'] = file['bp_used']*file['expected_return'] if (file['low'] < file['target_price']).all() else (file['open_position'] - file['stop_price']) * file['n_shares']
except KeyError:
    file['pnl'] = file['n_shares']*(file['open_position'] - file['target_price']) if (file['low'] < file['target_price']).all() else (file['open_position'] - file['stop_price']) * file['n_shares']

file['pnl'] = round(file['pnl'], 2)
file['eod_capital'] = round(file['capital'] + file['pnl'], 2)

strats = pd.read_csv("strategies.csv")
for ticker in file.ticker:
    strats.loc[strats['symbol'] == ticker, "current_capital"] = file.loc[file["ticker"] == ticker, "eod_capital"].item()

FILENAME = f"orders/eod/Orders_{date}.csv"
file.to_csv(FILENAME, index = False)
strats.to_csv('strategies.csv', index = False)