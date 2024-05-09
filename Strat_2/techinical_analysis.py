# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:20:12 2024

@author: ktsar
"""

import os 
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import talib


### MOMENTUM
def ROC(df, periods = [1,2,3,4,5, 10,15,20,40,60,100,125,185,252]):
    """Momentum indicator for different timeframe"""
    
    for period in periods:
        df[f'roc_{period}'] = talib.ROC(df['Close'], timeperiod=period)
        
    return df

def EMA(df, periods = [8, 20, 50, 100, 200]):
    
    for period in periods:
        df[f'EMA{period}'] = talib.EMA(df['Close'], timeperiod=period)
    
    return df

def momentum_oscillators(df):
    
    ## ABOVE 70 OVERBOUTH, BELOW 30 OVERSLD, ABOVE/BELOW 50 STRONG TREND
    df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
    
    #MACD > SL BULLISH, MACD < SL BEARISH
    df["macd_values"], df["macd_signal_line"], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # VOLUME
    df['vwap'] = talib.VWAP
    
    return df


### VOLATILITY 

def volatility(df):
    
    # VOLATILITY INDICATOR # HIGH ATR ==> HIGH VOL AND LOW ATR ==> LOW VOL
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'])
    
    # Define the parameters
    period = 20  # Lookback period for the moving average
    num_std_dev = 2  # Number of standard deviations for the bands
    df['bband_up'], df['bband_mid'], df['bband_low'] = talib.BBANDS(df['Close'], 
                                                                    timeframe = period, 
                                                                    nbdevup = num_std_dev, 
                                                                    nbdevdn = num_std_dev)
    return df



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
    