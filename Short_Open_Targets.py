# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:00:43 2024

@author: ktsar
"""

import os
import pandas as pd 

from datetime import datetime 
from Preprocessing_functions import downlaod_symbol_data


date = datetime.today().strftime('%Y_%m_%d')

FILE_PATH = "orders/"
FILENAME = "Orders_" + date + ".csv"

orders = pd.read_csv(FILE_PATH + FILENAME)
tickers = list(orders['ticker'])

for ticker in tickers:
    
    target_pct = orders[orders['ticker'] == ticker]['target_price'].item()
    
    open_price = downlaod_symbol_data(ticker, period = "1day")['Open'].item()
    target = round(open_price * target_pct, 2)

    print(f"{ticker} - Price target = {target}")

