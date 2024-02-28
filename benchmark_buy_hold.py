# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:14:40 2024

@author: ktsar
"""

import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

from Strat_1 import Preprocessing_functions as pf 
from Strat_1.calculateMaxDD import calculateMaxDD

ticker = 'SPY'

df = pf.downlaod_symbol_data(ticker, period = "360mo")
df['ret'] = df['Close'].pct_change()
df = pf.format_idx_date(df)

start_date = "2006-01-01"
end_date = "2024-12-31"

df = df[(df.index >= start_date) & (df.index <= end_date)]

#####   AVG YEARLY RETURN
mean_ret = round(df['ret'].mean()* 252 * 100 , 2)

#####   SHARPE RATIO
sharpe_ratio = round(np.sqrt(252) * np.mean(df['ret']) / np.std(df['ret']),2)

#####   DRAWDOWN METRICS
cum_ret = np.cumprod(1+ df['ret']) - 1
maxDrawdown, maxDrawdownDuration, startDrawdownDay=calculateMaxDD(cum_ret.values)

print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Maximum Drawdown: {round(maxDrawdown,4)}')
print(f'Max Drawdown Duration: {maxDrawdownDuration} days' )
print(f'Start day Drawdown: {startDrawdownDay}')
print(f"Average Yearly Return: {round(mean_ret, 2)} %")

#####   PLOTTING
plt.figure(figsize=(10, 7))
plt.plot(df.index, df['Close'], color='blue', linestyle='-')
plt.title(f'{ticker} BUY AND HOLD STRATEGY')
plt.xlabel('Date')
plt.ylabel('USD')
plt.grid(True)
plt.show()