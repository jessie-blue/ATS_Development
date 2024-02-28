# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:05:35 2024

@author: ktsar
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from Strat_1 import Preprocessing_functions as pf 
import seaborn as sns
import matplotlib.pyplot as plt

ticker = 'XLU'

df = pf.downlaod_symbol_data(ticker, period = "360mo")

df = pf.create_momentum_feat(df, ticker)

df['ret'] = df['Close'].pct_change()

df = df.dropna()

# =============================================================================
# SELECT TARGET AND EXPLANATORY VARS 
# =============================================================================

target = df.pop('ret')

cols = df.columns

X = df.drop(columns = ['Open', 'Close', 'High',
                       'Low', 'Capital Gains', 
                       'Stock Splits', f"{ticker}_mom1" ])

# X = df[['Volume', 'gap', f"{ticker}_mom1"]]
# =============================================================================
# CHECK CORRELATIONS BETWEEN FEATURES
# =============================================================================

CORR = X.corr()

# Create heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(CORR, annot=False, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Heatmap')
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.show()

# =============================================================================
# RUN OLS 
# =============================================================================

X = sm.add_constant(X)  # Adding constant for intercept
model = sm.OLS(target, X).fit(cov_type='HC3')
print("Univariate Linear Regression Results:")
print(model.summary())
    
    
    
