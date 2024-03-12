# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:39:30 2022

@author: ktsar
"""


import plotly.graph_objects as go
import os 
import re
import yfinance as yf
import matplotlib.pyplot as plt 
import pandas as pd
from datetime import datetime
os.getcwd()

os.chdir('C:/Users/ktsar/OneDrive/Desktop/ATS/Development')


df = pd.read_excel('preferreds.xlsx')
del df['Unnamed: 4']
df = df.rename(columns = { 'Symbol\nCUSIP' : 'Symbol', 'Stock\nExchange' : 'Exchange'})
df.head()


df['Symbol'] = df['Symbol'].str.replace('\d+', '')
df['Symbol'] = df['Symbol'].str.replace('\n.*', '')
df['Symbol'] = df['Symbol'].str.replace('-', '-P')
df['Exchange'] = df['Exchange'].str.replace('\nChart', '')
df.head()


print(df['Exchange'].value_counts())


df = df[df['Exchange'].isin(['NYSE', 'NGM', 'NGS'])]
print(df['Exchange'].value_counts())

symbols = tuple(zip(df['Symbol'], df['Security'], df['Exchange']))

#stock = symbols[0]

for stock in symbols:

    # Set your ticker 
    symbol = stock[0]
    ticker = yf.Ticker(symbol)
    print(pd.Series(ticker.info).head(5))

    # Pull the data - set period and interval 
    data = ticker.history(period = '10000000d',
                          interval = '1d',
                          actions = True,
                          auto_adjust= True)

    # Bring data into the df from the index 
    data = data.reset_index()


    # Create the figure 
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'])])

    fig.update_layout(
        title={
            'text': stock[0] + ' ' + stock[1] + ' ' + stock[2],
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    
    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.show()

    # Save the figure
    fig.write_image('Charts/{}.png'.format(stock[0]), width=1400, height=800, scale=2)
    
    fig.close()