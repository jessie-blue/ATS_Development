# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:07:23 2024

@author: ktsar
"""

import os 
import pandas as pd 
import numpy as np 

from Strat_1 import Preprocessing_functions as pf 
from datetime import datetime 
from Portfolio_functions import *

date = datetime.today().strftime('%Y_%m_%d')

#date = "2024_02_22"

FILE_PATH = "C:/Users/ktsar/Downloads/Python codes/Python codes/Git_Repos/ATS_Development/orders/"
FILENAME = "Orders_" + date + ".csv"

orders = pd.read_csv(FILE_PATH + FILENAME)
tickers = list(orders['ticker'].unique())

orders.ticker = orders.ticker.drop_duplicates()
bp = orders['bp_used'].sum()

df = pd.DataFrame()

#ticker = tickers[0]

for ticker in tickers:
    
    df1 = pf.downlaod_symbol_data(ticker, period = "360mo")
    #df1[f'ret_{ticker}' ] = df1['Close'] /  df1['Open'] - 1       #np.log(df1['Close'] / df1['Close'].shift(1))
    df1[f'ret_{ticker}' ] = df1['open_high'] / 100      #np.log(df1['Close'] / df1['Close'].shift(1))
    
    df =  pd.concat([df,df1[f'ret_{ticker}']], axis = 1)

df = df.dropna()
df_ret = df.mean().to_numpy()
df_std = df.std().to_numpy()

# MONTE CARLO APPROACH 
portfolio_value = orders['bp_used'].sum()
num_simulations = int(10e3)
num_assets = len(tickers)
confidence_level = 0.99

asset_returns = np.random.normal(df_ret, df_std, size=(num_simulations, num_assets))

# Calculate portfolio returns
portfolio_returns = np.mean(asset_returns, axis=1)

# Calculate portfolio value at risk
portfolio_value_at_risk = np.percentile(portfolio_value * portfolio_returns, 100 * (1 - confidence_level))

print(f"Value at Risk (VaR) at {confidence_level * 100:.2f}% confidence level: ${portfolio_value_at_risk:.2f}")

import matplotlib.pyplot as plt

# Plot histogram of portfolio returns
plt.figure(figsize=(10, 6))
plt.hist(portfolio_value * portfolio_returns, bins=50, density=True, alpha=0.9, color='b')

# Plot vertical line for VaR
plt.axvline(x=portfolio_value_at_risk, color='r', linestyle='--', linewidth=2, label='Value at Risk (95%)')

plt.title(f'Portfolio Returns Distribution - {date}')
plt.xlabel('Portfolio Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()









