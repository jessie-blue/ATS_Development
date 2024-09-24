# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:26:49 2024

@author: ktsar
"""

import os 
import datetime 
import numpy as np
import pandas as pd 
import mplfinance as mpf
import matplotlib.pyplot as plt
import yfinance as yf
import time 
import re 

from ALGO_KT1 import Preprocessing_functions as pf 
from pathlib import Path

start_time = time.time()

prefs = pd.read_csv("prefs_240924.csv")

prefs = prefs.rename(columns={'Symbol\nCUSIP' : 'Symbol'})

days = range(2,21)

for idx, row in prefs.iterrows():
    
    #print(row)
    print(row["Symbol"].split('\n')[0].replace('-', '-P'))
    ticker = row["Symbol"].split('\n')[0].replace('-', '-P')
    #print(ticker)
    if ticker == "Symbol":
        continue

    #call_date = row['Call Date'].split('\r\n')[0]
    call_date = row['Call Date\nMatur Date'].split('\r\n')[0]
    #mat_date = row['Call Date'].split('\n')[1]
    mat_date = row['Call Date\nMatur Date'].split('\n')[1]
    #cpn_rate = row['Cpn Rate\r\nAnn Amt'].split("\n")[0].strip()
    cpn_rate = row['Cpn Rate\nAnn Amt'].split("\n")[0].strip()
    #ann_rate = row['Cpn Rate\r\nAnn Amt'].split("\n")[1].strip()
    ann_rate = row['Cpn Rate\nAnn Amt'].split("\n")[1].strip()

    df = pf.downlaod_symbol_data(ticker, period = "6mo")
    
    if df.empty:
        continue

    df['ex_date'] = np.where(df['Dividends'] != 0, True, False)
    df['label'] = np.where(df['Close'] > df['Close'].shift(1), "green", "red")
    
    try:
        df['post_div_1'] = (df['ex_date'].shift(1) * df['Close'].pct_change()) * 100
        for day in days:
            df[f'post_div_{day}'] = (df['ex_date'].shift(day) * df['Close'] / df['Close'].shift(day) - 1) * 100 
        
    except ZeroDivisionError: 
        print(ticker, ': No useful data for this symbol')
        continue
    
    for day in days:
        df[f'post_div_{day}'] = np.where(df[f'post_div_{day}'] == -100, 0 , df[f'post_div_{day}'])
    
    symbol = yf.Ticker(ticker)
    
    today = df.tail(1)
    try:
        current_yield = np.round(float(ann_rate.replace("$",'')) / today['Close'].item(),4)*100
    
    except ValueError:
        current_yield = 'NA'
    except ZeroDivisionError:
        current_yield = 'NA'
    
    if abs(today['open_high'].iloc[0]) > 4:
        
        
        print(f"Unusual Activity: {ticker}")
        
        try: 
            last_div = datetime.datetime.utcfromtimestamp(symbol.info['lastDividendDate']).strftime("%Y-%m-%d")
        except KeyError:    
            last_div = "nan"
        
        DATE = datetime.datetime.today().strftime("%Y_%m_%d")
        MODEL_PATH = Path(f"Orders/{DATE}/")
        MODEL_PATH.mkdir(parents = True, exist_ok = True)
    
        mpf.plot(df, type='candle', 
                  style='charles', 
                  volume=True,
                  figsize = (20,10),
                  title = f"-- \n {ticker}\n Current Yield: {current_yield}%, Annual Rate: {ann_rate}, Cpn Rate: {cpn_rate}, Call Date: {call_date}",
                  xlabel = f"Date: {DATE.replace('_', '-')}   (Last dividend date: {last_div})",
                  ylabel = "Price USD",
                  addplot = mpf.make_addplot(df['Dividends']),
                  savefig= MODEL_PATH / f'{ticker}.png'
                  )
    
    
    
    for n in days:
    
        if today[f'post_div_{n}'].item() < -2:
            
            print(f"BUY: {ticker} - on the BID")
            
            try: 
                last_div = datetime.datetime.utcfromtimestamp(symbol.info['lastDividendDate']).strftime("%Y-%m-%d")
            except KeyError:    
                last_div = "nan"
            #date_obj = datetime.datetime.utcfromtimestamp(msft.history_metadata['firstTradeDate']).strftime("%Y-%m-%d")
            print("Last Dividend Date: ", last_div)
            
            DATE = datetime.datetime.today().strftime("%Y_%m_%d")
            MODEL_PATH = Path(f"Orders/{DATE}/")
            MODEL_PATH.mkdir(parents = True, exist_ok = True)
            
            mpf.plot(df, type='candle', 
                      style='charles', 
                      volume=True,
                      figsize = (20,10),
                      title = f"-- \n {ticker} \n Current Yield: {current_yield}%, Annual Rate: {ann_rate}, Cpn Rate: {cpn_rate}, Call Date: {call_date}",
                      xlabel = f"Date: {DATE.replace('_', '-')}   (Last dividend date: {last_div})",
                      ylabel = "Price USD",
                      addplot = mpf.make_addplot(df['Dividends']),
                      savefig= MODEL_PATH / f'{ticker}.png'
                      )
            
        # for date in dividend_dates:
            # mpf.plot([], [], marker='o', markersize=10, color='green', label='Dividend', linestyle='None', linewidth=0)
    # if ticker == "MITT-PC":
    #     break
#symbol.dividends    
end_time = time.time()

print(f"Time taken: {(end_time - start_time)/ 60}")

# =============================================================================
# YAHOO FINANCE DATA USE 
# =============================================================================
# import yfinance as yf

# msft = yf.Ticker("CHSCL")

# # get all stock info
# msft.info

# date_obj = datetime.datetime.utcfromtimestamp(msft.info['lastDividendDate']).strftime("%Y-%m-%d")
# #date_obj = datetime.datetime.utcfromtimestamp(msft.history_metadata['firstTradeDate']).strftime("%Y-%m-%d")
# print("Last Dividend Date: ", date_obj )

# date_obj = datetime.datetime.utcfromtimestamp(msft.info['exDividendDate']).strftime("%Y-%m-%d")
# print("Next Dividend Date: ", date_obj )

# # get historical market data
# hist = msft.history(period="1mo")

# # show meta information about the history (requires history() to be called first)
# msft.history_metadata

# # show actions (dividends, splits, capital gains)
# msft.actions
# msft.dividends
# msft.splits
# msft.capital_gains  # only for mutual funds & etfs

# # show share count
# msft.get_shares_full(start="2022-01-01", end=None)

# # show financials:
# # - income statement
# msft.income_stmt
# msft.quarterly_income_stmt
# # - balance sheet
# msft.balance_sheet
# msft.quarterly_balance_sheet
# # - cash flow statement
# msft.cashflow
# msft.quarterly_cashflow
# # see `Ticker.get_income_stmt()` for more options

# # show holders
# msft.major_holders
# msft.institutional_holders
# msft.mutualfund_holders
# msft.insider_transactions
# msft.insider_purchases
# msft.insider_roster_holders

# # show recommendations
# msft.recommendations
# msft.recommendations_summary
# msft.upgrades_downgrades


# =============================================================================
# 
# =============================================================================
































