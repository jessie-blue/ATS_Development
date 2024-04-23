# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:49:55 2024

@author: ktsar
"""

import os 
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import talib


from ALGO_KT1 import Preprocessing_functions as pf


ticker = "SPY"
df = pf.downlaod_symbol_data(ticker, period = "3mo")

def reversal_patterns(df):
    
    df['hammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    df['hanging_man'] = talib.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])
    df['engulfing_pattern'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
    df['dark_cloud'] = talib.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'])
    df['piercing_line'] = talib.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])
    
    ### STARS ###
    df['morning_star'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['evening_star'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['inverted_hammer'] = talib.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    
    ### LESS POWERFUL REVERSAL PATTERNS ###
    df['harami'] = talib.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])
    df['harami_cross'] = talib.CDLHARAMICROSS(df['Open'], df['High'], df['Low'], df['Close'])
    df['belt_hold'] = talib.CDLBELTHOLD(df['Open'], df['High'], df['Low'], df['Close'])
    df['upsidegap_two_crows'] = talib.CDLUPSIDEGAP2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
    df['three_black_crows'] = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])
    df['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])
    df['advance_block'] = talib.CDLADVANCEBLOCK(df['Open'], df['High'], df['Low'], df['Close'])
    df['stalled_pattern'] = talib.CDLSTALLEDPATTERN(df['Open'], df['High'], df['Low'], df['Close'])
    df['counterattack'] = talib.CDLCOUNTERATTACK(df['Open'], df['High'], df['Low'], df['Close'])
    
    return df 

def continuation_patterns(df):
    
    df['tasuki'] = talib.CDLTASUKIGAP(df['Open'], df['High'], df['Low'], df['Close'])
    df['rf_three_methods'] = talib.CDLRISEFALL3METHODS(df['Open'], df['High'], df['Low'], df['Close'])
    df['separating_lines'] = talib.CDLSEPARATINGLINES(df['Open'], df['High'], df['Low'], df['Close'])

    return df

def magic_doji(df):
    
    df['long_legged_doji'] = talib.CDLLONGLEGGEDDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['gravestone_doji'] = talib.CDLGRAVESTONEDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['dragonfly_doji'] = talib.CDLDRAGONFLYDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['tristar_doji'] = talib.CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    
    return df
    

df = reversal_patterns(df)

# df = df[df['hm'] != 0]

mpf.plot(df, type='candle', 
          style='charles', 
          volume=True,
          figsize = (20,10),
          title = f"{ticker}",
          xlabel = f"Date",
          ylabel = "Price USD",
          #addplot = mpf.make_addplot(df['hammer'])
          #savefig= MODEL_PATH / f'{ticker}.png'
          )




