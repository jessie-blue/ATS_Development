# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:54:23 2024

@author: ktsar
"""
import pandas as pd

timeframe = 60
    
filename = f'C:/Users/User/Desktop/crypto_data/OHLCVT/Q1_2024/ADAUSD_{timeframe}.csv'

cols = ['timestamp', "open", "high", "low", "close", "volume", "trades"]

df = pd.read_csv(filename, names = cols)

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') # / pd.Timedelta(minutes=1)

df.to_csv(filename.replace(".csv", "_v1.csv"), index = False)
