# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:24:40 2024

@author: ktsar
"""

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


ticker = "XLU"
n_clusters = 3 

### LOAD KMEANS MODEL ###
KMEANS_PATH = f"kmeans_models/{ticker}/"
KMEANS_NAME = f"kmeans_model_df_XLU_k3_202402011414.joblib"
FILE = KMEANS_PATH + KMEANS_NAME
loaded_kmeans = joblib.load(FILE)

### LOAD FEAT LIST TO ORDER THE DATA ###
#MODEL_PATH = Path(f"lstm_models/{ticker}")
#FEAT_NAME = f"LSTM_df_XLU_k3_202401251838_NFEAT{model_feat.shape[0]}.csv"
FEAT_NAME = f"model_features/{ticker}/LSTM_df_XLU_k3_202402011414_NFEAT23.csv"
#FEAT_SAVE_PATH = MODEL_PATH / FEAT_NAME
MODEL_FEAT = pd.read_csv(FEAT_NAME)['0'].to_list()

# Cluster stats
STATS_PATH = f"Data/{ticker}/k_stats/"
#print("KMEANS Stats files: ", os.listdir(f"Data/{ticker}"))
STATS_NAME = 'KMEANS_Stats_df_XLU_k3_202402011414.csv'
cluster_stats = pd.read_csv(STATS_PATH + STATS_NAME).set_index("Unnamed: 0")

# LOAD DF FOR MODEL BUILDING TO CHECK DATE RANGES 
DF_PATH = f"Data/{ticker}/df/"
DF_NAME = "df_XLU_k3_202402011414.parquet"
df_dates = pd.read_parquet(DF_PATH + DF_NAME)
df_dates = format_idx_date(df_dates)

# LOAD LSTM MODEL STATE DICT  
MODEL_PATH = f"lstm_models/{ticker}/"
MODEL_NAME = 'LSTM_Class_df_XLU_k3_202402011414_Epoch_2283_TestAcc_86.21_TrainAcc_85.79_202402061559'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = downlaod_symbol_data(ticker, period = "240mo")
df = format_idx_date(df)

# Needs refactoring - take start date in the training data and select days before that
#start_date = df.index.min() + datetime.timedelta(days=3985)
start_date = df_dates.index.min()
df = df[df.index <= start_date]
del DF_NAME, df_dates 


df = create_momentum_feat(df, ticker).dropna()

# ### LOAD KMEANS MODEL ###
# KMEANS_PATH = f"kmeans_models/{ticker}/"
# KMEANS_NAME = f"kmeans_model_df_BTC-USD_k3_202402021844.joblib"
# FILE = KMEANS_PATH + KMEANS_NAME
# loaded_kmeans = joblib.load(FILE)

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
drop_cols = ['Open', 'High', 'Low', 'Close', 'Stock Splits']
df1 = df_model[drop_cols]

### ORDER THE DATA ###
#MODEL_PATH = Path(f"lstm_models/{ticker}")
#FEAT_NAME = f"LSTM_df_XLU_k3_202401251838_NFEAT{model_feat.shape[0]}.csv"
# FEAT_NAME = "LSTM_df_BTC-USD_k3_202402021844_NFEAT23.csv"
#FEAT_SAVE_PATH = MODEL_PATH / FEAT_NAME
# MODEL_FEAT = pd.read_csv(FEAT_NAME)['0'].to_list()

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
# MODEL_PATH = f"lstm_models/{ticker}/"
#print(os.listdir(f"lstm_models/{ticker}"))
# MODEL_NAME = 'LSTM_Class_df_BTC-USD_k3_202402021844_Epoch_1080_TestAcc_97.06_TrainAcc_94.82_202402021853'
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

# STATS_PATH = f"Data/{ticker}/"
#print("KMEANS Stats files: ", os.listdir(f"Data/{ticker}"))
# STATS_NAME = 'KMEANS_Stats_df_BTC-USD_k3_202402021844.csv'

cluster_stats = pd.read_csv(STATS_PATH + STATS_NAME).set_index("Unnamed: 0")


#ACC = (df1['labels'] == df1['predictions']).sum() / df1.shape[1]

#### BACKTESTING ####
import numpy as np

df1 = df1.sort_index()
df1_cols = [i for i in df1.columns if "mom" not in i]
df1 = df1[df1_cols]
del df1_cols

capital = 25000
tc = 3

#create a list of clusters to use in the backtesting df1
k_names = []

for n in range(0,3):
    
    open_low = cluster_stats[f'open_low_{n}']['median']
    open_close = cluster_stats[f'open_close_{n}']['median']
    
    if open_low and open_close >= 0:
        k_names.append(n)
        
    if open_low > open_close and open_low > 0 and abs(open_close)*3 < open_low:
        if n not in k_names:
            k_names.append(n)
    
###### N SHARES CODE ############
# K1 RES 
static_shares = False

if static_shares is True:
    shares = 100
    
    df1['target_1'] = round((1 - cluster_stats.loc["median" , "open_low_0"]/100) * df1['Open'], 2) 
    df1['target_2'] = round((1 - cluster_stats.loc["median" , "open_low_1"]/100) * df1['Open'], 2) 
    
    df1['k1_true'] = (df1['target_1'] >= df1['Low']) 
    df1['k1_profit'] = (df1['k1_true'] * (df1['Open'] - df1['target_1']))* shares
    df1['k1_loss'] = (df1['Open'] - df1['Close']) * shares
    df1['k1_pnl'] = np.where(df1['k1_true'] == True, df1['k1_profit'], df1['k1_loss'])
    del df1['k1_profit'], df1['k1_loss']
    
    # K2 RES
    df1['k2_true'] = df1['target_2'] >= df1['Low'] 
    df1['k2_profit'] = (df1['k2_true'] * (df1['Open'] - df1['target_2']))* shares
    df1['k2_loss'] = (df1['Open'] - df1['Close']) * shares
    df1['k2_pnl'] = np.where(df1['k2_true'] == True, df1['k2_profit'], df1['k2_loss'])
    del df1['k2_profit'], df1['k2_loss']
    
    # Combining results 
    df1['k1_k2'] = np.where(df1['predictions'] == 0, df1['k1_pnl'], df1['k2_pnl'])
    df1['k0_k1_k2'] = np.where(df1['predictions'] == 2, 0, df1['k1_k2'] )
    df1['net_pnl'] = np.where(df1['k0_k1_k2'] != 0, df1['k0_k1_k2'] - tc, 0)
    df1['pnl_cumsum'] = df1['net_pnl'].cumsum()

################### DYNAMIC SHARES ######################################

dynamic_shares_raw = False 

if dynamic_shares_raw is True:

    df1['shares'] = capital // df1['Close']
    
    # K1 RES 
    df1['target_1'] = round((1 - cluster_stats.loc["median" , "open_low_0"]/100) * df1['Open'], 2) 
    df1['target_2'] = round((1 - cluster_stats.loc["median" , "open_low_1"]/100) * df1['Open'], 2) 
    
    df1['k1_true'] = (df1['target_1'] >= df1['Low']) 
    df1['k1_profit'] = (df1['k1_true'] * (df1['Open'] - df1['target_1']))* df1['shares']
    df1['k1_loss'] = (df1['Open'] - df1['Close']) * df1['shares']
    df1['k1_pnl'] = np.where(df1['k1_true'] == True, df1['k1_profit'], df1['k1_loss'])
    del df1['k1_profit'], df1['k1_loss']
    
    # K2 RES
    df1['k2_true'] = df1['target_2'] >= df1['Low'] 
    df1['k2_profit'] = (df1['k2_true'] * (df1['Open'] - df1['target_2']))* df1['shares']
    df1['k2_loss'] = (df1['Open'] - df1['Close']) * df1['shares']
    df1['k2_pnl'] = np.where(df1['k2_true'] == True, df1['k2_profit'], df1['k2_loss'])
    del df1['k2_profit'], df1['k2_loss']
    
    # Combining results 
    df1['k1_k2'] = np.where(df1['predictions'] == 0, df1['k1_pnl'], df1['k2_pnl'])
    df1['k0_k1_k2'] = np.where(df1['predictions'] == 2, 0, df1['k1_k2'] )
    df1['net_pnl'] = np.where(df1['k0_k1_k2'] != 0, df1['k0_k1_k2'] - tc, 0)
    df1['pnl_cumsum'] = df1['net_pnl'].cumsum()
    
    df1['daily_ret'] = ((df1['net_pnl'] + capital) - capital)  / capital # for sharpe ratio calc


################### DYNAMIC SHARES EXPERIMENTATION ######################################

experimentation = False

if experimentation is True:
    df3 = df1.copy()
    
    df3['shares'] = capital // df3['Close']
    
    # K1 RES 
    df3['target_1'] = round((1 - cluster_stats.loc["median" , "open_low_0"]/100) * df3['Open'], 2) 
    df3['target_2'] = round((1 - cluster_stats.loc["median" , "open_low_1"]/100) * df3['Open'], 2) 
    
    df3['k1_true'] = (df3['target_1'] >= df3['Low']) 
    df3['k1_profit'] = (df3['k1_true'] * (df3['Open'] - df3['target_1']))* df3['shares']
    df3['k1_loss'] = (df3['Open'] - df3['Close']) * df3['shares']
    df3['k1_pnl'] = np.where(df3['k1_true'] == True, df3['k1_profit'], df3['k1_loss'])
    #del df3['k1_profit'], df3['k1_loss']
    
    # K2 RES
    df3['k2_true'] = df3['target_2'] >= df3['Low'] 
    df3['k2_profit'] = (df3['k2_true'] * (df3['Open'] - df3['target_2']))* df3['shares']
    df3['k2_loss'] = (df3['Open'] - df3['Close']) * df3['shares']
    df3['k2_pnl'] = np.where(df3['k2_true'] == True, df3['k2_profit'], df3['k2_loss'])
    #del df3['k2_profit'], df3['k2_loss']
    
    # Combining results 
    df3['k1_k2'] = np.where(df3['predictions'] == 0, df3['k1_pnl'], df3['k2_pnl'])
    df3['k0_k1_k2'] = np.where(df3['predictions'] == 2, 0, df3['k1_k2'] )
    df3['net_pnl'] = np.where(df3['k0_k1_k2'] != 0, df3['k0_k1_k2'] - tc, 0)
    df3['pnl_cumsum'] = df3['net_pnl'].cumsum()
    
    df3['daily_ret'] = ((df3['net_pnl'] + capital) - capital)  / capital # for sharpe ratio calc
    
    plt.plot(df3['net_pnl'])

################### TILL HERE ######################################

for k in range(len(k_names)):
    
    df1['shares'] = capital // df1['Close'] ## you need to divide cluster stats from target with USO - check clusters stats df for % or decimals 
    df1[f'target_{k_names[k]}'] = round((1 - cluster_stats.loc["median" , f"open_low_{k_names[k]}"]) * df1['Open'], 2) 
    
    df1[f'k{k_names[k]}_true'] = (df1[f'target_{k_names[k]}'] >= df1['Low']) 
    df1[f'k{k_names[k]}_profit'] = (df1[f'k{k_names[k]}_true'] * (df1['Open'] - df1[f'target_{k_names[k]}']))* df1['shares']
    df1[f'k{k_names[k]}_loss'] = round(((df1['Open'] - df1['Close']) * df1['shares']),4)
    df1[f'k{k_names[k]}_pnl'] = np.where(df1[f'k{k_names[k]}_true'] == True, df1[f'k{k_names[k]}_profit'], df1[f'k{k_names[k]}_loss'])
    del df1[f'k{k_names[k]}_profit'], df1[f'k{k_names[k]}_loss']


no_trade_k = [i for i in range(0,3) if i not in k_names][0] # predicted cluster for which we do not trade 

df1[f'k{k_names[0]}_k{k_names[1]}'] = np.where(df1['predictions'] == 0, df1[f'k{k_names[0]}_pnl'], df1[f'k{k_names[1]}_pnl'])
df1['k0_k1_k2'] = np.where(df1['predictions'] == no_trade_k, 0, df1[f'k{k_names[0]}_k{k_names[1]}'] )
df1['net_pnl'] = np.where(df1['k0_k1_k2'] != 0, df1['k0_k1_k2'] - tc, 0)
df1['pnl_cumsum'] = df1['net_pnl'].cumsum()
df1['daily_ret'] = ((df1['net_pnl'] + capital) - capital)  / capital


## datetime slicing 
#df1[pd.to_datetime(df1.index) <= "2007-03-30"]


#####   MAX DRAWDOWN
from calculateMaxDD import calculateMaxDD

cum_ret = np.cumprod(1+ df1['daily_ret']) - 1
#plt.plot(cum_ret)
maxDrawdown, maxDrawdownDuration, startDrawdownDay=calculateMaxDD(cum_ret.values)


#####   SHARPE RATIO
sharpe_ratio = round(np.sqrt(252) * np.mean(df1['daily_ret']) / np.std(df1['daily_ret']),2)

#####   AVG YEARLY RETURN
mean_ret = df1['daily_ret'].mean() * 252

print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Maximum Drawdown: {round(maxDrawdown,4)}')
print(f'Max Drawdown Duration: {maxDrawdownDuration} days' )
print(f'Start day Drawdown: {startDrawdownDay}')
print(f"Average Yearly Return: {round(mean_ret*100, 2)} %")



# Create figure and axis objects
plt.rcParams.update({'font.size': 12})

fig, ax1 = plt.subplots(figsize=(10, 7))
plt.title(f"Backtest Short Open Strategy - {ticker}")

# Plot data on the first y-axis
ax1.plot(df1.index, df1['Close'], 'g-')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price ', color='g')

# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(df1.index, df1['pnl_cumsum'], 'b-')
ax2.set_ylabel('Cummulative USD', color='b')

# Add black dotted line at y=0
#ax1.axhline(y=0, color='k', linestyle='--')
ax2.axhline(y=0, color='k', linestyle='--')

#Remove box lines around the chart area
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Add text box
stats_text = f"Strategy Stats:\n"
stats_text += f'Sharpe Ratio: {sharpe_ratio} :\n'
stats_text += f'Maximum Drawdown: {round(maxDrawdown,4)} \n'
stats_text += f'Start day Drawdown: {startDrawdownDay} \n'
stats_text += f"Average Yearly Return: {round(mean_ret*100, 2)} % \n"
fig.text(0.1, 0.03, stats_text, fontsize=12,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(facecolor='white', alpha=0.5,edgecolor='none'))

plt.show()