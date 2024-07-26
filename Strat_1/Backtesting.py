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
from techinical_analysis import * 

ticker = "VNQ"
n_clusters = 3 
time_period = "36mo"

### LOAD KMEANS MODEL ###
KMEANS_PATH = f"kmeans_models/{ticker}/"
print(os.listdir(KMEANS_PATH))
idx = 0 if len(os.listdir(KMEANS_PATH)) < 2 else int(input("Select file index: "))
KMEANS_NAME = os.listdir(KMEANS_PATH)[idx]
print("Chosen K_MEANS MODEL file: ", KMEANS_NAME)
FILE = KMEANS_PATH + KMEANS_NAME
loaded_kmeans = joblib.load(FILE)

### LOAD FEAT LIST TO ORDER THE DATA ###
FEAT_PATH = f"model_features/{ticker}/"
print(os.listdir(FEAT_PATH))
idx = 0 if len(os.listdir(FEAT_PATH)) < 2 else int(input("Select file index (e.g. 0,1,2)"))
FEAT_NAME = os.listdir(FEAT_PATH)[idx]
MODEL_FEAT = pd.read_csv(FEAT_PATH + FEAT_NAME)['0'].to_list()
#MODEL_FEAT.pop(-1)

# Cluster stats
STATS_PATH = f"Data/{ticker}/k_stats/"
print("KMEANS Stats files: ", os.listdir(STATS_PATH))
idx = 0 if len(os.listdir(STATS_PATH)) < 2 else int(input("Select file index: "))
STATS_NAME = os.listdir(STATS_PATH)[idx]
print("Chosen K_STATS file: ", STATS_NAME)
cluster_stats = pd.read_csv(STATS_PATH + STATS_NAME).set_index("Unnamed: 0")

# LOAD DF FOR MODEL BUILDING TO CHECK DATE RANGES 
DF_PATH = f"Data/{ticker}/df/"
print("DataFrame for model building: ", os.listdir(DF_PATH))
idx = 0 if len(os.listdir(DF_PATH)) < 2 else int(input("Select file index: "))
DF_NAME = os.listdir(DF_PATH)[idx] 
print("Chosen DataFrame file: ", DF_NAME)
df_dates = pd.read_parquet(DF_PATH + DF_NAME)
df_dates = format_idx_date(df_dates)

# LOAD LSTM MODEL STATE DICT  
MODEL_PATH = f"lstm_models/{ticker}/"
print(os.listdir(MODEL_PATH))
idx = 0 if len(os.listdir(MODEL_PATH)) < 2 else int(input("Select file index: "))
MODEL_NAME = os.listdir(MODEL_PATH)[idx]
print("Chosen LSTM, MODEL file: ", MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = downlaod_symbol_data(ticker, period = time_period)
df = format_idx_date(df)

# REMOVE DATA SNOOPING 
out_sample = True

if out_sample is True:
    start_date = df_dates.index.min()
    df = df[df.index <= start_date]
    #df = df[df.index >= '2010-01-01']
    
    del DF_NAME, df_dates 


df = create_momentum_feat(df, ticker).dropna()
df = momentum_oscillators(df)
df = volatility(df)
df = reversal_patterns(df) 
df = continuation_patterns(df)
df = magic_doji(df)
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

df_model = df_model[MODEL_FEAT]
df2 = df_model.copy()

## SCALING THE DATA BEFORE CONVERTING IT INTO SUITABLE INPUT FOR RNN 
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
#df1 = df1.merge(df2, left_index = True, right_index = True)
df1 = df2.copy()
del pred, output, predictions

cluster_stats = pd.read_csv(STATS_PATH + STATS_NAME).set_index("Unnamed: 0")
ACC = (df1['labels'] == df1['predictions']).sum() / df1.shape[1]

# =============================================================================
# #### BACKTESTING ####
# =============================================================================
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
    
################### ADDING KELLY ######################################

half_kelly_metric = True

if half_kelly_metric is True:
    
    start_capital = 1e4
    no_trade_k = [i for i in range(0,3) if i not in k_names][0]
    df = pd.DataFrame()
    #half_kelly = kelly_criterion(ticker, df1.index.min()) / 2 
    
    for date, row in df1.iterrows():
        
      #  try:
            #half_kelly = kelly_criterion(ticker, date) / 2
       # except FileNotFoundError:
           # half_kelly =  1
            
            
        half_kelly = 1
        # if half_kelly < 1:
        #     print("Convert half kelly to 1")
        #     half_kelly = 1
        
        # biyearly_hk = False
        
        # if biyearly_hk:
        
        #     if (date.month == 1 and date.day in (29,30,31)) or (date.month == 7 and date.day in (29,30,31)): 
        #         half_kelly = kelly_criterion(ticker, date) / 2
        #         if half_kelly < 1:
        #             half_kelly = 1
        #         print(date)
            
        for k in range(len(k_names)):
            
            row['shares'] = (start_capital * half_kelly) // row['Close'] ## you need to divide cluster stats from target with USO - check clusters stats df for % or decimals 
            row[f'target_{k_names[k]}'] = round((1 - cluster_stats.loc["median" , f"open_low_{k_names[k]}"]/100) * row['Open'], 2) 
            row[f'k{k_names[k]}_true'] = (row[f'target_{k_names[k]}'] >= row['Low']) 
            row[f'k{k_names[k]}_profit'] = (row[f'k{k_names[k]}_true'] * (row['Open'] - row[f'target_{k_names[k]}']))* row['shares']
            row[f'k{k_names[k]}_loss'] = round(((row['Open'] - row['Close']) * row['shares']),4)
            row[f'k{k_names[k]}_pnl'] = np.where(row[f'k{k_names[k]}_true'] == True, row[f'k{k_names[k]}_profit'], row[f'k{k_names[k]}_loss'])
            del row[f'k{k_names[k]}_profit'], row[f'k{k_names[k]}_loss']
            
        
        row[f'k{k_names[0]}_k{k_names[1]}'] = np.where(row['predictions'] == 0, row[f'k{k_names[0]}_pnl'], row[f'k{k_names[1]}_pnl'])
        row['k0_k1_k2'] = np.where(row['predictions'] == no_trade_k, 0, row[f'k{k_names[0]}_k{k_names[1]}'] )
        row['net_pnl'] = np.where(row['k0_k1_k2'] != 0, row['k0_k1_k2'] - tc, 0)
        row['eod_equity'] = start_capital + row['net_pnl']
        row['daily_ret'] = row['eod_equity'] / start_capital - 1
        row['half_kelly'] = half_kelly
        
        start_capital += row['net_pnl']
        df = pd.concat([df, row.to_frame().transpose()], axis= 0)
    
    #### SET DATATYPES IN THE NEW DF
    for col in list(df.columns):
        
        if ("true" or "last_day") in col:
            df[col] = df[col].astype("bool")
            
        elif ("labels" or "Volume" or "predictions") in col:
            df[col] = df[col].astype("int32")
        
        else:
            df[col] = df[col].astype("float64")
        
    del df1
    
    df1 = df.copy()
    
    df1['pnl_cumsum'] = df1['net_pnl'].cumsum()

# =============================================================================
# END OF ADDING KELLY
# =============================================================================

#####   MAX DRAWDOWN
from calculateMaxDD import calculateMaxDD

cum_ret = np.cumprod(1+ df1['daily_ret']) - 1
maxDrawdown, maxDrawdownDuration, startDrawdownDay=calculateMaxDD(cum_ret.values)

#####   SHARPE RATIO
sharpe_ratio = round(np.sqrt(252) * np.mean(df1['daily_ret']) / np.std(df1['daily_ret']),2)

#####   AVG YEARLY RETURN AND STD
mean_ret = df1['daily_ret'].mean() * 252
std = df1['daily_ret'].std()*np.sqrt(252)

import numpy as np
p_change = df1['Close'].pct_change().dropna() #/ df1['Close'].shift(1)
corr = np.corrcoef(p_change, df1['Close'][1:])

print(f"Correlation Price / Return: " , round(corr[1][0], 2))
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
ax1.plot(df1.index, df1['Close'], 'g-', alpha = 0.5)
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price ', color='g')

# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(df1.index, df1['eod_equity'], 'b-')
ax2.set_ylabel('Equity USD', color='b')

# Add black dotted line at y=0
#ax1.axhline(y=0, color='k', linestyle='--')
ax2.axhline(y=1e4, color='k', linestyle='--')

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
stats_text = f'Sharpe Ratio: {sharpe_ratio} :\n'
stats_text += f'Maximum Drawdown: {round(maxDrawdown*100,2)}% \n'
stats_text += f'Start day Drawdown: {startDrawdownDay} day \n'
stats_text += f"Drawdown Duration: {int(maxDrawdownDuration)} days \n"
stats_text += f"Average Yearly Return: {round(mean_ret*100, 2)} % \n"
stats_text += f"Average Yearly STD: {round(std*100, 2)} % \n"
fig.text(0.1, 0.03, stats_text, fontsize=12,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(facecolor='white', alpha=0.5,edgecolor='none'))

save = True
if save is True:
    plt.savefig(f"Short_Open_Backtests/Backtest_{ticker}_hk{half_kelly}", bbox_inches='tight')

plt.show()

# =============================================================================
# SAVE STRAT RETURNS TO AID CALCULATING HALF KELLY FOR LIVE TRADING
# =============================================================================

if out_sample is False:
    if time_period == "12mo":
        rets = df1['daily_ret'].to_frame()
        rets.to_csv(f'strat_returns/{ticker}.csv')
        #rets.to_csv(f'{ticker}.csv')







