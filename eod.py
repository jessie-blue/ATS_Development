# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:20:46 2024

@author: ktsar
"""

import os 
cwd = os.getcwd().replace("\\", "/"  )
os.chdir(cwd)

import numpy as np
import pandas as pd 
from datetime import datetime , timedelta, date 
from Strat_1.Preprocessing_functions import downlaod_symbol_data

date = input("Choose date (today, yesterday or YYYY_MM_DD): ")

if date == "today":
    date = datetime.today().strftime("%Y_%m_%d")

if date == "yesterday":
    date = (datetime.today() - timedelta(days = 1)).strftime("%Y_%m_%d")


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

#### CALCULATE PNL AND EQUITY
target_reached = file['bp_used']*file['expected_return']
close_pnl = (file['open_position'] - file['stop_price']) * file['n_shares']
file['target_reached'] = (file['low'] <= file['target_price'])
file['pnl'] = np.where(file['target_reached'] == True, target_reached, close_pnl)

file['pnl'] = round(file['pnl'], 2)
file['eod_capital'] = round(file['capital'] + file['pnl'], 2)

## FIX THE EOD_CAPITAL 
file["eod_capital"] = np.where(file['direction'] == "HOLD", file['capital'], file['eod_capital'])
file["pnl"] = np.where(file['direction'] == "HOLD", 0 , file['pnl'])

strats = pd.read_csv("strategies.csv")
for ticker in file.ticker:
    strats.loc[strats['symbol'] == ticker, "current_capital"] = file.loc[file["ticker"] == ticker, "eod_capital"].item()

FILENAME = f"orders/eod/Orders_{date}.csv"
file.to_csv(FILENAME, index = False)
strats.to_csv('strategies.csv', index = False)

file['ret'] = file['eod_capital'] / file['capital'] - 1 

for idx, row in file.iterrows():
    
    ticker = row['ticker']
    
    print(ticker)

    df_ret = pd.read_csv(f'Strat_1/strat_returns/{ticker}.csv', header = 0, names = ['date', 'ret'])
    
    data = file[file['ticker'] == ticker]
    
    # data['ret'] = data['eod_capital'].max() / data['capital'].max() - 1 
    data['date'] = datetime.today().strftime("%Y-%m-%d")
    data = data[['date', 'ret']]
    
    df_ret = pd.concat([df_ret, data], axis = 0)
    
    df_ret.to_csv(f'Strat_1/strat_returns/{ticker}.csv', index = False)


print(file)