# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:33:02 2024

@author: ktsar
"""

import os 
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn

from datetime import date 
from scipy.stats import skew, norm, kurtosis
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans


def downlaod_symbol_data(symbol = "XLU", period = "120mo"):
    
    """
    Download historical market data for a given stock symbol.
    
    Parameters:
    - symbol (str): Stock symbol to retrieve data for (default is "XLU").
    - period (str): Historical data period to retrieve (default is "120mo" for 10 years).
    
    Returns:
    pandas.DataFrame: DataFrame containing historical market data for the specified stock symbol.
                      The DataFrame includes additional columns:
                      - 'open_low': Difference between Open and Low prices.
                      - 'open_close': Difference between Open and Close prices.
                      - 'open_high': Difference between Open and High prices.
                      - 'high_low': Difference between High and Low prices.
                      - 'low_close': Difference between Low and Close prices.
                      - 'high_close': Difference between High and Close prices.
    """
    
    data = yf.Ticker(symbol)
    
    # get all stock info (slow)
    #data.info
    
    # get historical market data
    df = data.history(period=period).round(2)
    
    df['open_low'] = 100*((df['Open'] - df['Low']) / df['Open']) 
    
    df['open_close'] = 100*((df['Open'] - df['Close']) / df['Open'])
    
    df['open_high'] = 100*((df['Open'] - df['High']) / df['Open'])
    
    df['high_low'] = 100*(abs(df['High'] - df['Low']) / df['High'])
    
    df['low_close'] = 100*((df['Low'] - df['Close']) / df['Low'])
    
    df['high_close'] = 100*((df['Close'] - df['High']) / df['High'])
    
    df['gap'] = 100*((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1))

    return df

def download_data(ticker, days):
    import pandas_datareader.data as web
    import datetime

    # Define stock ticker, start, and end dates
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=days)

    # Fetch historical data from Stooq
    df = web.DataReader(ticker, "stooq", start_date, end_date)
    
    df = df.sort_index()
    
    df['open_low'] = 100*((df['Open'] - df['Low']) / df['Open']) 
    
    df['open_close'] = 100*((df['Open'] - df['Close']) / df['Open'])
    
    df['open_high'] = 100*((df['Open'] - df['High']) / df['Open'])
    
    df['high_low'] = 100*(abs(df['High'] - df['Low']) / df['High'])
    
    df['low_close'] = 100*((df['Low'] - df['Close']) / df['Low'])
    
    df['high_close'] = 100*((df['Close'] - df['High']) / df['High'])
    
    df['gap'] = 100*((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1))
    
    df['Dividends'] = 0 

    return df


def create_momentum_feat(df, symbol):
    
    """
    Create momentum features for a DataFrame based on the closing prices of a specified stock symbol.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing historical market data.
    - symbol (str): Stock symbol used for creating momentum features.

    Returns:
    pandas.DataFrame: DataFrame with added momentum features.
                      For each specified day in 'mom_days', columns are added with names
                      in the format '{symbol}_mom{day}', representing the percentage change
                      in closing prices over the specified number of days.
    """
    
    mom_days = [1,2,3,4,5,10,15,20,60,120,180,240]
    
    for day in mom_days:
        
        df[symbol + '_mom' + str(day)] = round(100 * (df['Close'] / df['Close'].shift(day) - 1),2)
    
    return df 


def dist_stats(data, col):
    """
    Calculate descriptive statistics for a selected column in a DataFrame.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing the data.
    - col (str): Name of the column for which descriptive statistics are calculated.

    Returns:
    pandas.DataFrame: DataFrame containing descriptive statistics for the specified column.
                      Statistics include count, minimum, maximum, mean, median, standard deviation,
                      skewness, and kurtosis.
    """    
    stats = { 

        'count' : data[col].count(),
        'min' : data[col].min(),
        'max' : data[col].max(),
        'mean' : data[col].mean(),
        'median' : data[col].median(),
        'std' : data[col].std(),
        'skew' : skew(data[col]),
        'kurtosis' : kurtosis(data[col])
    
    }
    
    stats = pd.DataFrame(stats.values(), index = stats.keys(), columns= [col]).round(4)
    
    return stats
        

def k_means_clustering(df, clusters = 4):
    """
    Perform K-means clustering on selected columns of a DataFrame and visualize the elbow method.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the data.
    - clusters (int): Number of clusters for K-means clustering (default is 7).

    Returns:
    tuple: A tuple containing two elements:
           1. pandas.DataFrame: DataFrame with selected columns and an additional 'labels' column
              indicating the cluster assignment for each data point.
           2. numpy.ndarray: Array containing the cluster labels assigned by K-means.
    """
    np.random.seed(42)
    
    inertias = []
    
    data = df[['open_low', "open_close", "gap"]].dropna()
    
    for i in range(1, clusters):
        kmeans = KMeans(n_clusters=i, 
                        random_state = 42,
                        verbose = False)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    plt.plot(range(1,clusters), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show() 
    
    data['labels'] = kmeans.labels_
    #print(f"Names of features used to train model: {data.columns}")

    return data, kmeans.labels_, kmeans



    
def cube_clusters_plot(data):
    """
    Create a 3D scatter plot to visualize clusters in a DataFrame.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing the data with cluster labels.

    Returns:
    None: Displays a 3D scatter plot using the 'open_low', 'open_close', and 'gap' columns.
    """
    
    fig = plt.figure(figsize=(10, 7)) 
    ax = plt.axes(projection='3d')

    # Iterate through unique labels and create separate scatter plots for each cluster
    for label in data['labels'].unique():
        cluster_data = data[data['labels'] == label]
        ax.scatter(cluster_data['open_low'], 
                   cluster_data['open_close'], 
                   cluster_data['gap'], 
                   label=f'Cluster {label}')

    ax.set_title('3D Scatter Plot')
    ax.set_xlabel('open_low')
    ax.set_ylabel('open_close')
    ax.set_zlabel('gap')
    ax.legend()
    ax.view_init(-145)  # -140, 45
    plt.show()


def cluster_stats(data, cluster_label, col1, col2, col3):
    
    """ 
    Subsets the df which was used for KMEANS fit and performs descriptive stats for the
    select columns (i.e open_low, open_close, gap etc) for a given label of a cluster (i.e. 1,2,3)
    
    Returns: A df with the cluster data, a df with descriptive stats 
    """
    avg = data[data['labels'] == cluster_label]
    avg_stats = dist_stats(avg, col1)
    avg_stats = avg_stats.merge(dist_stats(avg ,col2), left_index = True, right_index = True)
    avg_stats = avg_stats.merge(dist_stats(avg ,col3), left_index = True, right_index = True)

    return avg, avg_stats


def format_idx_date(df_model):
    
    df_model = df_model.reset_index()
    df_model['Date'] = pd.to_datetime(df_model['Date']).dt.date
    df_model = df_model.set_index("Date")
    df_model.index = pd.to_datetime(df_model.index)
    
    return df_model


# =============================================================================
# Build a dataframe for predictions 
# =============================================================================


def merge_dfs(data, df, symbol):
    """
    Merge cluster labels and selected columns from two DataFrames and handle look-ahead bias.

    Parameters:
    - data (pandas.DataFrame): DataFrame containing cluster labels.
    - df (pandas.DataFrame): DataFrame containing additional data columns.

    Returns:
    pandas.DataFrame: Merged DataFrame with cluster labels and selected columns from both input DataFrames.
                      Look-ahead bias is handled by shifting the 'labels' column and removing NaN values.
    """
    # Merge two clusters together - 1 and 2 in this case but might need to adjust - look at 3d graph
    #data['labels'] = data['labels'].replace(2,1)
    
    df1 = data[['labels']]
    try:
        df1 = df1.merge(df[[
            'Date',
            'open_low', 
            'open_close',
            'gap',
            'open_high',
            'low_close',
            'high_close',
            'high_low',
            'Dividends',
            'Volume',
            symbol+'_mom1', symbol+'_mom2', symbol+'_mom3', symbol+'_mom4', symbol+'_mom5', symbol+'_mom10', symbol+'_mom15',
            symbol+'_mom20', symbol+'_mom60', symbol+'_mom120', symbol+'_mom180',symbol+'_mom240'        
            ]], left_index = True, right_index = True)
    
    except KeyError:
        df1 = df1.merge(df[[
            'open_low', 
            'open_close',
            'gap',
            'open_high',
            'low_close',
            'high_close',
            'high_low',
            'Dividends',
            'Volume',
            symbol+'_mom1', symbol+'_mom2', symbol+'_mom3', symbol+'_mom4', symbol+'_mom5', symbol+'_mom10', symbol+'_mom15',
            symbol+'_mom20', symbol+'_mom60', symbol+'_mom120', symbol+'_mom180',symbol+'_mom240'        
            ]], left_index = True, right_index = True)
        
    # Remove look ahead bias 
   # df1['labels'] = df1['labels'].shift(-1)
    df1 = df1.dropna()
    
    
    return df1




def one_hot_target(y):
   """
   One-hot encodes the target variable.

   Parameters:
   - y (numpy.ndarray or pandas.Series): Target variable to be one-hot encoded.

   Returns:
   numpy.ndarray: One-hot encoded representation of the input target variable.

   Example:
   >>> import numpy as np
   >>> y = np.array([0, 1, 2, 1, 0])
   >>> one_hot_target(y)
   array([[1., 0., 0.],
          [0., 1., 0.],
          [0., 0., 1.],
          [0., 1., 0.],
          [1., 0., 0.]])
   """
   y = y.astype(int)
    
   def one_hot_encode(vector):
       num_classes = np.max(vector) + 1  # Assuming values range from 0 to max_value
       return np.eye(num_classes)[vector]
    
   y = one_hot_encode(y)
    
   return y 



def create_multivariate_rnn_data(data, window_size):
    """
    Creates input-output pairs for training a multivariate Recurrent Neural Network (RNN).

    Parameters:
    - data (numpy.ndarray): The input data, a 2D array with shape (num_samples, num_features),
                           where num_samples is the total number of data points and
                           num_features is the number of features for each data point.
    - window_size (int): The size of the sliding window used to create input sequences.

    Returns:
    - X (numpy.ndarray): Input sequences for the RNN, a 3D array with shape
                        (num_samples - window_size, window_size, num_features).
    - y (numpy.ndarray): Output sequences for the RNN, a 2D array with shape
                        (num_samples - window_size, num_features), corresponding to the next
                        data points after each input sequence in X.
    """
    y = data[window_size:]
    n = data.shape[0]
    X = np.stack([data[i: j] for i, j in enumerate(range(window_size, n))], axis=0)
    return X, y




def train_test_seq_split(df_model, pct): #### NEDS TO BE FIZED FOR THE NUMPY TO TORCH PART!!!
    """
    Splits a sequential dataset into training and testing sets.

    Parameters:
    - df_model (str or pandas.DataFrame): Path to a parquet file or a DataFrame containing the dataset.
    - pct (float): Percentage of data to be used for training. Should be in the range (0, 1).

    Returns:
    tuple: A tuple containing four torch tensors - (your_train_data, your_test_data, your_train_labels, your_test_labels).

    Example:
    >>> # Load the dataset from a parquet file or use a DataFrame
    >>> df_model = pd.read_parquet("path/to/your/dataset.parquet")
    >>> 
    >>> # Split the dataset into training and testing sets
    >>> train_data, test_data, train_labels, test_labels = train_test_seq_split(df_model, 0.8)
    """
    
    train_size = int(df_model.shape[0] * pct)
    
    X = df_model.drop("labels", axis =1)
    y = df_model["labels"].values 
    
    your_train_data = torch.from_numpy(X.iloc[0: train_size, :].values.astype(""))
    your_test_data = torch.from_numpy(X.iloc[train_size:, :].values)
    your_train_labels = torch.from_numpy(y[0: train_size, :])
    your_test_labels = torch.from_numpy(y[train_size:, :])
    
    return your_train_data, your_test_data, your_train_labels, your_test_labels


def min_max_scaling(df_model):
    
    from sklearn.preprocessing import MinMaxScaler
    
    x = df_model.drop(labels = "labels", axis = 1)
    scaler = MinMaxScaler()
    x_fit = scaler.fit(x)
    x = scaler.transform(x)
    x = pd.DataFrame(x)
    y = df_model.labels.to_frame().reset_index()
    
    del df_model
    df_model = y.merge(x, left_index = True, right_index = True)
    df_model = df_model.set_index("Date")
    del x, x_fit,y 

    return df_model


def cluster_inspection(df_model, cluster_number):
    
    df2 = df_model[df_model['labels'] == cluster_number]
    plt.figure(figsize = (12,7))
    plt.scatter(df2['open_low'], 
                df2['open_close'], 
                c = "b",
                s = df2['gap']*20)
    plt.title(f"Cluster {cluster_number} Distribution")
    plt.xlabel('open_low')
    plt.ylabel("open_close")
    
    return df2


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


from datetime import date 

def kelly_criterion(ticker, 
                    date_to = date.today(), 
                    #period = "120mo",
                    path = 'strat_retuns'):
    
    from dateutil.relativedelta import relativedelta
    import datetime    
    #ticker = 'AMLP'
    #path = 'strat_returns'
    #df = pd.read_csv(f'{path}/{ticker}.csv', header = 0, names = ['date', 'daily_ret'])
    
    df = pd.read_csv(f'{path}/{ticker}.csv', header = 0)
    df.date = pd.to_datetime(df.date)
    
    date_to = pd.to_datetime(date_to)    
    date_from = pd.to_datetime(date_to) - relativedelta(months=6)
    
    # Slice the data for the 6month period
    df = df[df.date <= date_to]
    df = df[df.date >= pd.to_datetime(date_from)]

    try:
        mean_ret = df["daily_ret"].mean()
        #std = df["daily_ret"].std() 
        sigma_sq = np.var(df['daily_ret'], ddof = 1)
    
    except KeyError:
        mean_ret = df["ret"].mean()
        sigma_sq = np.var(df['ret'], ddof = 1)
    
    
    if sigma_sq == 0:
        return 0  # Avoid division by zero
    
    kelly = mean_ret / sigma_sq
    
    if abs(kelly) > 8:
        kelly = 8
        
    # if abs(kelly) < 1 :
    #     kelly = 1
    df = df.set_index('date')
    print(f"Kelly Calculation window: From: {df.index.min()} To: {df.index.max()}")
    return round(abs(kelly) , 2)
    

def intraday_data(ticker = 'SPY', interval = '1min', size = 'compact'):
    from alpha_vantage.timeseries import TimeSeries
    import pandas as pd
    
    API_KEY = "HHZPLP7IGWVH4SQW"
    
    # Initialize Alpha Vantage
    ts = TimeSeries(key=API_KEY, output_format="pandas")
    
    # Get 1-minute interval intraday data (adjust interval as needed)
    data, meta_data = ts.get_intraday(symbol=ticker, interval=interval, outputsize=size)
    
    return data 


def realtime_data(ticker = 'SPY'):
    from alpha_vantage.timeseries import TimeSeries
    import pandas as pd
    
    API_KEY = "HHZPLP7IGWVH4SQW"
    
    # Initialize Alpha Vantage
    ts = TimeSeries(key=API_KEY, output_format="pandas")
    
    # Fetch real-time stock data
    data, meta_data = ts.get_quote_endpoint(symbol=ticker)
    
    # Rename columns for better readability
    data = data.rename(columns={
        "01. symbol": "Symbol",
        "02. open": "Open",
        "03. high": "High",
        "04. low": "Low",
        "05. price": "Price",
        "06. volume": "Volume",
        "07. latest trading day": "Latest Trading Day",
        "08. previous close": "Previous Close",
        "09. change": "Change",
        "10. change percent": "Change Percent"
    })

    return data 
    
    
