# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:22:39 2024

@author: ktsar
"""

import joblib
from datetime import datetime
from pathlib import Path 
from Preprocessing_functions import *
from techinical_analysis import *

ticker = "IVE"
n_clusters = 3
specific_timeframe = True # usually used to campare against a specific model build 
save = True # save the data 

df = downlaod_symbol_data(ticker, period = "360mo")
df = create_momentum_feat(df, ticker)
df = momentum_oscillators(df)
df = volatility(df)
df = reversal_patterns(df) 
df = continuation_patterns(df)
df = magic_doji(df)

df = df.drop(columns= ['Open', 
                       'High', 
                       'Low', 
                       'Close',
                       'Stock Splits', 
                       'Capital Gains'])

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

#df_model = merge_dfs(data, df, ticker)
data = data[['labels']]
df_model = data.merge(df, left_index= True, right_index=True)


if specific_timeframe:
    df_model = df_model[df_model.index <= "2024-02-01"]

day = datetime.today().strftime('%Y%m%d%H%M')

if save is True:
        
    DATA_MODEL_PATH = Path(f"Strat_1/Data/{ticker}/df")
    DATA_MODEL_PATH.mkdir(parents = True, exist_ok = True)
    
    DATA_MODEL_NAME =  f"df_{ticker}_k{n_clusters}_{day}.parquet"
    DATA_MODEL_SAVE_PATH = DATA_MODEL_PATH / DATA_MODEL_NAME
    df_model.to_parquet(DATA_MODEL_SAVE_PATH, index = True)
    
    # Save Cluster stats
    STATS_MODEL_NAME = f"KMEANS_Stats_{DATA_MODEL_NAME.replace('.parquet', '')}.csv"
    STATS_PATH = Path(f"Strat_1/Data/{ticker}/k_stats")
    STATS_PATH.mkdir(parents = True, exist_ok= True)
    STATS_SAVE_PATH = STATS_PATH / STATS_MODEL_NAME
    cluster_dist.to_csv(STATS_SAVE_PATH, index = True)   
    
    # Save kmeans model to a file using joblib
    KMEANS_MODEL_PATH =  Path(f"Strat_1/kmeans_models/{ticker}")
    KMEANS_MODEL_PATH.mkdir(parents = True, exist_ok = True)
    
    KMEANS_MODEL_NAME = f"kmeans_model_{DATA_MODEL_NAME.replace('.parquet', '')}.joblib"
    KMEANS_MODEL_SAVE_PATH = KMEANS_MODEL_PATH / KMEANS_MODEL_NAME
    joblib.dump(kmeans, KMEANS_MODEL_SAVE_PATH)

open_low_stats = dist_stats(df, "open_low")
open_close_stats = dist_stats(df, "open_close")

######## ADDED LATER FOR VISUAL CLUSTERS INSPECTION ##############
testing = False

if testing is True:
    df2 = df_model[df_model['labels'] == 2]
    plt.figure(figsize = (14,8))
    plt.scatter(df2['open_low'], 
                df2['open_close'], 
                c = "b")
                #s = df2['gap'])
    plt.title("Cluster 2 Distribution")
    plt.xlabel('open_low')
    plt.ylabel("open_close")
    
   # cluster_inspection(df_model, 2)
    cluster_inspection(df_model, 0)
    cluster_inspection(df_model, 1)
   # cluster_inspection(df_model, 3)


