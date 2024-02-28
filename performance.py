# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:16:36 2024

@author: ktsar
"""

import os 
import numpy as np
import pandas as pd 
import datetime as dt 

date = (dt.datetime.today() - dt.timedelta(days = 1)).strftime("%Y_%m_%d")

dates = pd.date_range("2024-02-14", "2024-02-27")

date = dates[0].strftime("%Y_%m_%d")

orders = f"orders/eod/Orders_{date}.csv"

orders = pd.read_csv(orders)

orders['eod_capital'] = orders['capital'] + orders['pnl']

strats = pd.read_csv("strategies.csv")
strats['date'] = date.strftime("%Y_%m_%d")

for ticker in orders.ticker:
    strats.loc[strats['symbol'] == ticker, "current_capital"] = orders.loc[orders["ticker"] == ticker, "eod_capital"]

