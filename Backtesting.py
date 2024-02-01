# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:23:43 2024

@author: ktsar
"""

import datetime
import joblib
import torch
import torch.nn

from LSTM_Architecture import LSTM
from pathlib import Path
from Preprocessing_functions import *


ticker = "SPY"
n_clusters = 3 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = downlaod_symbol_data(ticker, period = "240mo")
df = format_idx_date(df)

# Needs refactoring - take start date in the training data and select days before that
start_date = df.index.min() + datetime.timedelta(days=3985)

df = df[df.index <= start_date]

df = create_momentum_feat(df, ticker).dropna()

### LOAD KMEANS MODEL ###
KMEANS_PATH = f"kmeans_models/{ticker}/"
KMEANS_NAME = f"kmeans_model_df_SPY_k3_202402012133.joblib"
FILE = KMEANS_PATH + KMEANS_NAME
loaded_kmeans = joblib.load(FILE)

### ASSIGN CLUSTER TO OBSERVATION ###
data = df[["open_low", "open_close", "gap"]].dropna()
k_predictions = pd.DataFrame(loaded_kmeans.predict(data), columns = ["labels"], index = data.index)
#data = data.merge(k_predictions, left_index = True, right_index = True)#.reset_index()
del FILE, KMEANS_NAME, KMEANS_PATH, loaded_kmeans

df_model = df.merge(k_predictions, left_index = True, right_index = True)

end_date = df_model.index.max()
df_model['last_day'] = (df_model.index == end_date).astype(int)
del df, data, k_predictions

seq_length =  1
df_model = df_model.sort_index(ascending = False)

# preserve the price features to use in the backtest data
drop_cols = ['Open', 'High', 'Low', 'Close', 'Stock Splits', 'Capital Gains']
df1 = df_model[drop_cols]

### ORDER THE DATA ###
#MODEL_PATH = Path(f"lstm_models/{ticker}")
#FEAT_NAME = f"LSTM_df_XLU_k3_202401251838_NFEAT{model_feat.shape[0]}.csv"
FEAT_NAME = "LSTM_df_SPY_k3_202402012133_NFEAT23.csv"
#FEAT_SAVE_PATH = MODEL_PATH / FEAT_NAME
MODEL_FEAT = pd.read_csv(FEAT_NAME)['0'].to_list()

df_model = df_model[MODEL_FEAT]
df2 = df_model.copy()

## SCALING THE DATA BEFORE CONVERTING IT INTO SUITABLE INPUT FOR RNN 
# df_model = df_model.drop(columns = drop_cols)
# columns_in = list(df_model.columns)
# columns_in = [item for item in columns_in if item != "labels"]
# columns_in.insert(0, "labels")

df_model = min_max_scaling(df_model)
df_model.columns = MODEL_FEAT
del drop_cols

X, y  = create_multivariate_rnn_data(df_model, seq_length)
del y

############################ PREDICTION #######################################


X_tensor = torch.from_numpy(X).type(torch.float).to(device).squeeze(0)

input_feat = df_model.shape[1]
hidden_size = 32
num_layers = 2 
num_classes = 3

# INSTANTIATE MODEL
model = LSTM(input_size=input_feat, 
             output_size=num_classes, 
             hidden_size=hidden_size, 
             num_layers=num_layers,
             device=device).to(device)

# LOAD LSTM MODEL STATE DICT  
MODEL_PATH = f"lstm_models/{ticker}/"
#print(os.listdir(f"lstm_models/{ticker}"))
MODEL_NAME = 'LSTM_Class_df_SPY_k3_202402012133_Epoch_215_TestAcc_80.80_TrainAcc_72.78_202402012138'
interactive = False

if interactive is True:
    MODEL_IDX = int(input('Choose model index:'))
    MODEL_NAME = os.listdir(f"lstm_models/{ticker}")[MODEL_IDX]


model.load_state_dict(torch.load(f = MODEL_PATH + MODEL_NAME ))

del MODEL_PATH, MODEL_NAME
#### PREDICTION #### 
model.eval()

with torch.inference_mode():

    output = model(X_tensor)
    pred = torch.softmax(output, dim = 1).argmax(dim = 1)


## possible mistake in creating the predictions df - dates might not align properly
predictions = pd.DataFrame(pred.to("cpu").numpy(), columns = ["predictions"], index = df_model.index[:-1])


df2 = df2.merge(predictions, left_index = True, right_index = True)
df1 = df1.merge(df2, left_index = True, right_index = True)

del pred, output, predictions

STATS_PATH = f"Data/{ticker}/"
#print("KMEANS Stats files: ", os.listdir(f"Data/{ticker}"))
STATS_NAME = 'KMEANS_Stats_df_SPY_k3_202402012133.csv'

cluster_stats = pd.read_csv(STATS_PATH + STATS_NAME).set_index("Unnamed: 0")


#ACC = (df1['labels'] == df1['predictions']).sum() / df1.shape[1]

#### BACKTESTING ####
import numpy as np

df1 = df1.sort_index()
shares = 100
tc = 3

df1['target_1'] = round((1 - cluster_stats.loc["median" , "open_low_0"]/100) * df1['Open'], 2) 
df1['target_2'] = round((1 - cluster_stats.loc["median" , "open_low_2"]/100) * df1['Open'], 2) 

df1['k1_true'] = (df1['target_1'] >= df1['Low']) 
df1['k1_profit'] = (df1['k1_true'] * (df1['Open'] - df1['target_1']))* shares
df1['k1_loss'] = (df1['Open'] - df1['Close']) * shares
df1['k1_pnl'] = np.where(df1['k1_true'] == True, df1['k1_profit'], df1['k1_loss'])
del df1['k1_profit'], df1['k1_loss']


df1['k2_true'] = df1['target_2'] >= df1['Low'] 
df1['k2_profit'] = (df1['k2_true'] * (df1['Open'] - df1['target_2']))* shares
df1['k2_loss'] = (df1['Open'] - df1['Close']) * shares
df1['k2_pnl'] = np.where(df1['k2_true'] == True, df1['k2_profit'], df1['k2_loss'])
del df1['k2_profit'], df1['k2_loss']


df1['k1_k2'] = np.where(df1['predictions'] == 0, df1['k1_pnl'], df1['k2_pnl'])

df1['k0_k1_k2'] = np.where(df1['predictions'] == 1, 0, df1['k1_k2'] )

df1['net_pnl'] = np.where(df1['k0_k1_k2'] != 0, df1['k0_k1_k2'] - tc, 0)

df1['pnl_cumsum'] = df1['net_pnl'].cumsum()



plt.figure(figsize = [10,7])
plt.plot(df1['pnl_cumsum'], color = 'b')
plt.axhline(0, color = 'black', linewidth = 1)
plt.xlabel('Date')
plt.ylabel('Cummulative PNL')
plt.title("Backtest Short Open Strategy")



plt.figure(figsize = [10,7])
plt.plot(df1['Close'], color = 'b')
plt.axhline(0, color = 'black', linewidth = 1)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title("Backtest Short Open Strategy")





