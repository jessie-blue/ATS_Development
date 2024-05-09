# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:21:15 2024

@author: ktsar
"""

import os 
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
import ta
import talib
from statsmodels.tsa.stattools import adfuller

from techinical_analysis import *
from ALGO_KT1 import Preprocessing_functions as pf


ticker = "BTC-USD"
df = pf.downlaod_symbol_data(ticker, period = "24mo")


#df = ROC(df)

df = EMA(df)

df = momentum_oscillators(df) ## now vwap 

df = volatility(df)

df = reversal_patterns(df)

df = continuation_patterns(df)

df = magic_doji(df)

df['vwap'] = ta.volume.volume_weighted_average_price(df['High'],
                                                     df['Low'],
                                                     df['Close'],
                                                     df['Volume'])



df['vwap_diff'] = df['vwap'] - df['Close'] # if +ve then VWAP > Close
df['vwap_pct_diff'] =  (df['vwap_diff']) /  df['Close'].shift(-1)

df = df.dropna()

stats = pf.dist_stats(df, "vwap_diff")
stats_pct = pf.dist_stats(df, "vwap_pct_diff")


vwap_band_low = 0.1 * (-1)
vwap_band_high = 0.08

profit_low = 0.07 * (-1)
profit_high = 0.06


df['buy_vwap'] = df['vwap_pct_diff'] < vwap_band_low
df['sell_vwap'] = df['vwap_pct_diff'] > vwap_band_high 
df['vwap_strat'] = 0

df['vwap_strat'] = np.where(df["buy_vwap"] == True, 1, df['vwap_strat'])
df['vwap_strat'] = np.where(df["sell_vwap"] == True, -1, df['vwap_strat'])




















# Create the plot
plt.figure(figsize=(20, 10))  # Set the figure size
plt.plot(df.index, df['Close'], label="Price", color = "blue")
plt.plot(df.index, df['vwap'], label="VWAP", color = 'green')
# Add labels and title
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"Close vs VWAP for {ticker}")
# Add legend
plt.legend()
# Show the plot
plt.grid(True)
plt.show()

















# =============================================================================
#  TESTING FOR UNIT ROOT
# =============================================================================

def test_unit_root(price_series, significance_level=0.05):
    """
    Test if a price series contains a unit root using Augmented Dickey-Fuller test.
    
    Parameters:
    - price_series: A pandas Series containing the price series.
    - significance_level: The significance level for the test (default is 0.05).
    
    Returns:
    - A tuple (test_statistic, p_value, is_unit_root), where:
        - test_statistic: The test statistic from the Augmented Dickey-Fuller test.
        - p_value: The p-value from the test.
        - is_unit_root: True if the null hypothesis (presence of a unit root) is not rejected, False otherwise.
    """
    adf_result = adfuller(price_series, autolag='AIC')
    test_statistic, p_value = adf_result[0], adf_result[1]
    
    is_unit_root = p_value > significance_level
    
    return test_statistic, p_value, is_unit_root

# Example usage:
# Assuming you have a pandas Series named 'price_series' containing your price data
# Replace this with your actual price series data
# price_series = pd.Series([...])

# Call the function to test for a unit root
test_statistic, p_value, is_unit_root = test_unit_root(df['vwap_diff'].dropna())

# Print the results
print("Test Statistic:", test_statistic)
print("P-value:", p_value)
if is_unit_root:
    print("The series likely contains a unit root.")
else:
    print("The series likely does not contain a unit root.")
























