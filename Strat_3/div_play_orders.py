# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:41:43 2024

@author: ktsar
"""

import os 
import datetime 
import pandas as pd 
import yfinance as yf
import time 
from urllib.error import HTTPError
from ALGO_KT1 import Preprocessing_functions as pf 

start_time = time.time()

df = pd.read_csv("Div_play/div_play_screened_list.csv")
# df = pd.read_csv("prefs_240311.csv")

# df['symbol'] = df['Symbol'].str.split("\n").str[0].replace("-", "-P")
# df['symbol'] = df['symbol'].replace("-", "-P")


for idx, row in df.iterrows():
    
    symbol = yf.Ticker(row['symbol'])
    
    # symbol.info['averageVolume']
    
    try:
        last_div = datetime.datetime.utcfromtimestamp(symbol.info['lastDividendDate']).strftime("%Y-%m-%d")
    except KeyError:
        last_div = symbol.dividends.index[-1].strftime("%Y-%m-%d")
            
    try:
        next_div = datetime.datetime.utcfromtimestamp(symbol.info['exDividendDate'])#.strftime("%Y-%m-%d")
        next_div = next_div if next_div.strftime("%Y-%m-%d") != last_div else "Div not available yet"
        
    except KeyError:
       next_div = f"Div not available yet {row['symbol']}"
       
    if isinstance(next_div,str):
        print(f"No div date yet {row['symbol']}")
        # print(row['symbol'] , next_div)
        continue 
       
    
    today = int(time.time()) / (60*60*24)
    ex_date_epoch = int(next_div.timestamp()) / (60*60*24)
    div_rate = symbol.info['dividendRate']
    
    days_diff = (ex_date_epoch - today)
    print(f"{row['symbol']}: days_diff = {days_diff}")
        
    #     print(f"{row['symbol']} : {next_div.strftime('%Y-%m-%d')} : {div_rate}")
    # symbol.info['shortRatio']
    # symbol.info['sharesShort']
    
