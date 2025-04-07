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
from Strat_1.Preprocessing_functions import * # downlaod_symbol_data


date = input("Choose date (today, yesterday or YYYY_MM_DD): ")

if date == "today":
    date = datetime.today().strftime("%Y_%m_%d")

if date == "yesterday":
    date = (datetime.today() - timedelta(days = 1)).strftime("%Y_%m_%d")


FILENAME = f"orders/Orders_{date}.csv"
try:
    file = pd.read_csv(FILENAME)
    
    file_2 = file[file['strat'] == 'Strat_2']
    
    file = file[file['strat'] == 'Strat_1']

except FileNotFoundError:
    print(f"Date {date} not a trading day!")

hi_lo = pd.DataFrame()
# print(f"run for loop for date {date}")
for idx, row in file.iterrows():
    
    ticker = row['ticker']
    
    print(ticker)
    
    df = downlaod_symbol_data(ticker, period= "3mo")
    #df = download_data(ticker, days = 10)
    df = df[df.index.strftime("%Y_%m_%d") == date] # this is where the code breaks / check conversions 
    
    #break
    
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


file["eod_capital"] = np.where(file['direction'] == "HOLD", file['capital'], file['eod_capital'])
file["pnl"] = np.where(file['direction'] == "HOLD", 0 , file['pnl'])

#strats = pd.read_csv("strategies.csv")
#strats_2 = strats[strats['strategy_name'] == 'Strat_2']
#strats = strats[strats['strategy_name'] == 'Strat_1']
#for ticker in file.ticker:
#    strats.loc[strats['symbol'] == ticker, "current_capital"] = file.loc[file["ticker"] == ticker, "eod_capital"].item()

### Strat 2 - file_2 EOD Calculations for one symbol only 
ticker = file_2['ticker'].item()

mkt_data = downlaod_symbol_data(ticker, period='6mo')
#mkt_data = download_data(ticker, days=10)
date2 = date.replace('_', '-')
#date2 = datetime.strptime(date2, '%Y-%m-%d')
mkt_data  = mkt_data[mkt_data.index == date2]
file_2.loc[file_2["ticker"] ==  ticker, "stop_price"] = mkt_data['Close'].item()
file_2.loc[file_2["ticker"] ==  ticker, "open_position"] = mkt_data['Open'].item()
file_2['high'] = mkt_data['High'].item()
file_2['low'] = mkt_data['Low'].item()
file_2['target_reached'] = 'nan'
file_2['pnl'] = file_2['n_shares'] * (file_2['open_position'] - file_2['stop_price'])
file_2['pnl'] = file_2['pnl'].item() * (-1) if file_2['direction'].item() == 'BUY' else file_2['pnl'].item()
file_2['eod_capital'] = file_2['capital'] + file_2['pnl']

### Adjust the strategies file 
strats = pd.read_csv("strategies.csv")
strats = strats[strats['status'] == 'active'] ### need to adjust active and testing strats
strats['current_capital'] = strats['current_capital'].astype(float)

# strat 2 
strats_2 = strats[strats['strategy_name'] == 'Strat_2']
strats_2.loc[strats_2['symbol'] == ticker, "current_capital"] = file_2.loc[file_2["ticker"] == ticker, "eod_capital"].item()

# strat 1 
strats = strats[strats['strategy_name'] == 'Strat_1']

strat_1_capital = strats['starting_capital'].sum()

for ticker in file.ticker:
    strats.loc[strats['symbol'] == ticker, "current_capital"] = file.loc[file["ticker"] == ticker, "eod_capital"].item()

### Concatenate the two strategies 
file = pd.concat([file, file_2], axis=0)
strats = pd.concat([strats, strats_2], axis = 0 )
del file_2, strats_2 

### Save the file with the updates 
FILENAME = f"orders/eod/Orders_{date}.csv"
file.to_csv(FILENAME, index = False)
strats.to_csv('strategies.csv', index = False)

### Returns updates and calculation
file['ret'] = file['eod_capital'] / file['capital'] - 1 

file['date'] = date.replace('_', '-') # new way of setting date - used to be in the for loop

for idx, row in file.iterrows():
    
    ticker = row['ticker']
    strat = row['strat']
    
    print(f'Symbol: {ticker}, Strategy: {strat}')

    if strat == 'Short_Open': # if this works to be deleted - the if condition
        strat = 'Strat_1'
        
    df_ret = pd.read_csv(f'{strat}/strat_returns/{ticker}.csv', header = 0, names = ['date', 'ret'])
    
    data = file[file['ticker'] == ticker]
    data = data[data['strat'] == strat]
    # old way of setting date  data['date'] = datetime.today().strftime("%Y-%m-%d")
    data = data[['date', 'ret']]
    
    df_ret = pd.concat([df_ret, data], axis = 0)
    df_ret = df_ret.reset_index(drop=True)
    
    df_ret.to_csv(f'{strat}/strat_returns/{ticker}.csv', index = False)

    

print(file)
file = file[file['strat'] != 'Strat_2']
print('Strat_1 Daily PNL: ', round(file['pnl'].sum(),2))
#adj_uso = 20
#adj290125 = 132+9 + adj_uso
#adj280225 = 300
#adj050325 = 37
#adj170325 = 200
#adjustments = adj_uso + adj290125 + adj280225 + adj050325 + adj170325
print('Strat_1 PNL', round(file['eod_capital'].sum() - strat_1_capital, 2))