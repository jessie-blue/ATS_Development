# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:42:24 2024

@author: ktsar
"""
import os 
import joblib
from datetime import datetime
import pandas as pd
from pathlib import Path
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt 
from scipy.stats import skew, norm, kurtosis
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from Preprocessing_functions import *
#os.getcwd()

ticker = "XLU"
n_clusters = 3

df = downlaod_symbol_data(ticker)
df = create_momentum_feat(df, ticker)

data, _, kmeans = k_means_clustering(df, n_clusters + 1)

cube_clusters_plot(data)

cluster_dist = pd.DataFrame()

for cluster_label in range(0,data.labels.nunique()):
    
    cluster_df, stat = cluster_stats(data, cluster_label, "open_low", "open_close", "gap")
    cluster_df = cluster_df.reset_index(drop = True)
    del cluster_df['labels']
    stat = stat.rename(columns = {"open_low": f"open_low_{cluster_label}",
                                              "open_close" : f"open_close_{cluster_label}",
                                              "gap" : f"gap_{cluster_label}"})
                          
    cluster_dist = pd.concat([cluster_dist,stat], axis = 1)

df_model = merge_dfs(data, df, ticker)

save = False
day = datetime.today().strftime('%Y%m%d%H%M')

if save == True:
    
    DATA_MODEL_PATH = Path(f"Data/{ticker}")
    DATA_MODEL_PATH.mkdir(parents = True, exist_ok = True)
    
    DATA_MODEL_NAME =  f"df_{ticker}_k{n_clusters}_{day}.parquet"
    DATA_MODEL_SAVE_PATH = DATA_MODEL_PATH / DATA_MODEL_NAME
    df_model.to_parquet(DATA_MODEL_SAVE_PATH, index = True)
    
    # Save Cluster stats
    STATS_MODEL_NAME = f"KMEANS_Stats_{DATA_MODEL_NAME.replace('.parquet', '')}.csv"
    STATS_SAVE_PATH = DATA_MODEL_PATH / STATS_MODEL_NAME
    cluster_dist.to_csv(STATS_SAVE_PATH, index = True)   
    
    # Save kmeans model to a file using joblib
    KMEANS_MODEL_PATH =  Path(f"kmeans_models/{ticker}")
    KMEANS_MODEL_PATH.mkdir(parents = True, exist_ok = True)
    
    KMEANS_MODEL_NAME = f"kmeans_model_{DATA_MODEL_NAME.replace('.parquet', '')}.joblib"
    KMEANS_MODEL_SAVE_PATH = KMEANS_MODEL_PATH / KMEANS_MODEL_NAME
    joblib.dump(kmeans, KMEANS_MODEL_SAVE_PATH)

open_low_stats = dist_stats(df, "open_low")
open_close_stats = dist_stats(df, "open_close")

######## ADDED LATER ##############
testing = False

if testing == True:
    df2 = df_model[df_model['labels'] == 2]
    plt.figure(figsize = (14,8))
    plt.scatter(df2['open_low'], 
                df2['open_close'], 
                c = "b",
                s = df2['gap']*200)
    plt.title("Cluster 2 Distribution")
    plt.xlabel('open_low')
    plt.ylabel("open_close")
    
    cluster_inspection(df_model, 0)
    cluster_inspection(df_model, 1)


