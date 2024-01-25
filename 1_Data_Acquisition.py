# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:22:39 2024

@author: ktsar
"""
import os 
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt 
from scipy.stats import skew, norm, kurtosis
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from Preprocessing_functions import *
os.getcwd()

ticker = "XLU"

df = downlaod_symbol_data(ticker)
df = create_momentum_feat(df, ticker)

data, _, kmeans = k_means_clustering(df, 4)

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

if save == True:
    
    name = input("Type a name for the data to be used for modelling: ")
    df_model.to_parquet("data_models/df_model_{ticker}_" + name, index = True)
    
    # Save Cluster stats
    cluster_dist.to_csv("data_models/Cluster_Statistics.csv", index = True)   
    
    # Save kmeans model to a file using joblib
    model_filename = "data_models/kmeans_model_XLU_3_clusters.joblib"
    joblib.dump(kmeans, model_filename)

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




