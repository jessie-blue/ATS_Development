# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:01:21 2024

@author: ktsar
"""

import os
import pandas as pd 

file = os.listdir('Data/XLU/k_stats')[0]


xlu = pd.read_csv('Data/XLU/k_stats/' + file).set_index("Unnamed: 0")

b4 = xlu.copy()

stats = ['mean', "median", "max", "min", "std"]

for stat in stats:
    xlu.loc[stat, :] = xlu.loc[stat, :]*100

xlu.to_csv('Data/XLU/k_stats/' + file)
