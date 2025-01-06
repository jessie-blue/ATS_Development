# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:16:36 2024

@author: ktsar
"""

import os 
import numpy as np
import pandas as pd 
import datetime as dt 
import matplotlib.pyplot as plt

path = 'orders/eod/'

df = pd.DataFrame()


for filename in os.listdir('orders/eod'):
    
    file = pd.read_csv(path + filename)
    
    date = filename.replace("Orders_", '').replace(".csv", "").replace("_","-")
    
    file['date'] = date
    
    file = file[["date","strat", "ticker","n_shares", "pnl", "eod_capital"]]
    
    df = pd.concat([df,file], axis = 0)
    
        
def strat_perfomance(ticker, df):

    df1 = df[df['ticker'] == ticker]
    strategy = df1.strat.unique().item()
    
    cum_sum = df1.pnl.cumsum()
    
    plt.figure(figsize = (20,14))
    plt.plot(df1.date, df1['eod_capital'], color = "b")
    #plt.plot(df1.date, cum_sum, color = "b")
    plt.title(f"{strategy}, {ticker}, Equity Curve")
    plt.xlabel("Date")
    plt.xticks(rotation = 90)
    plt.xticks(fontsize=9, fontname='Arial')
    plt.ylabel("USD")
    plt.yticks(fontsize=13, fontname='Arial')




strat_perfomance("XLU", df)
strat_perfomance("XLI", df)
strat_perfomance("USO", df)
strat_perfomance("AMLP", df)
strat_perfomance("SPY", df)






















    
    
    