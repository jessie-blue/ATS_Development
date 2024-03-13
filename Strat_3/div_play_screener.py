# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:04:26 2024

@author: ktsar
"""

import os 
import datetime 
import numpy as np
import pandas as pd 
import yfinance as yf
import time 
# import re 

from ALGO_KT1 import Preprocessing_functions as pf 

start_time = time.time()

prefs = pd.read_csv("prefs_240311.csv")


for i in range(0, prefs.shape[0]):
    
        
    ticker = prefs['Symbol'][i].split("\n")[0].replace("-", "-P")
    
    if ticker == "Symbol":
        continue
    
    df = pf.downlaod_symbol_data(ticker, period = "360mo")
    
    if df.empty:
        continue
    
    pre_div_days = 15
    
    df['ex_div_date'] = df['Dividends'] != 0
    df['pre_div_1'] = df['ex_div_date'].shift(-1)
    df['pre_div_15'] = df['ex_div_date'].shift(-pre_div_days)
    df['ret'] = np.where(df['pre_div_1'] == True, df['Close'] / df['Close'].shift(-pre_div_days - 1) - 1 , 0 )
    
    min_cumret = df['ret'].cumsum().min()
    max_cumret = df['ret'].cumsum().max()
    #print(f"{ticker} [ {min_cumret} : {max_cumret} ]")
    
    df1 = {}
    capital = 1e4
    
    for idx, row in df.iterrows():
        
        if row['ret'] != 0:
            
            pnl = row['ret'] * capital 
            
            capital += pnl 
            
            df1[idx] = [pnl, capital]
    
    if len(df1) < 2:
        print(f"Not enough data for {ticker}")
        continue
        
    df1 = pd.DataFrame(df1).transpose()
    df1.columns = ['pnl', 'capital']
    df1 = df1.dropna()
    
    hit_rate = (df1['pnl'] > 0).sum() / df1.shape[0]
    
    n_divs = df1.shape[0]
    
    end_capital = df1['capital'].iloc[-1]
    
    strat_return = round(end_capital/ 1e4 - 1, 4) * 100
    
    print(f"{ticker}: {strat_return}%")
    
    if strat_return >= 10:
        
        
        df2 = [ticker, strat_return, n_divs, hit_rate, min_cumret, max_cumret]
        
        cols = ['symbol', 'pct', 'n_divs','hit_rate', 'min_cum', 'max_cum']
        df2 = pd.DataFrame(df2).transpose()
        df2.columns = cols
        
        filename = "div_play_screened_list.csv" 
        if filename in os.listdir("Div_play"):
            file = pd.read_csv('Div_play/div_play_screened_list.csv')        
        else:
            df2.to_csv('Div_play/' + filename, index = False)
            continue
        
        file = pd.concat([file, df2], axis = 0)
        
        file.to_csv('Div_play/div_play_screened_list.csv', index = False)
    

end_time = time.time()

print(f"Time taken: {(end_time - start_time) / 60} mins")