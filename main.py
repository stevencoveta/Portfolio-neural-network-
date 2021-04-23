import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import streamlit as st
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
pd.options.mode.chained_assignment = None 
from sklearn.cluster import KMeans
from technical_indicators_lib import OBV , CCI , CHV , CMF , DPO, EMA , EMV , MACD ,MFI ,MI, PVT, RSI, EMV
from pathlib import Path
import plotly.express as px
from sklearn.metrics import silhouette_score

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv3D, Flatten, Dropout, concatenate,BatchNormalization , LSTM , GRU, RNN, Conv2D, Conv1D, GlobalMaxPooling1D, TimeDistributed,Input,Bidirectional, ConvLSTM2D,GlobalAveragePooling1D,Attention
from keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from stqdm import stqdm
from pandas_datareader import data,wb
from datetime import date , timedelta

from chuncks import * 
from models import *
from bucket import *
from ta import *

st.set_page_config(layout="wide")
col1, col2 = st.beta_columns(2)

col1.header("Date Today")
col1.write(pd.datetime.now().date(), use_column_width=True)

col2.header("Estrategia :bulb:")
col2.write("la idea aqui es hacer un clustering de las empresas del sp500, invertiendo en las empresas en el cluster con la mejor media predicta ", use_column_width=True)


liss = [OBV() , CCI() , CHV(), CMF() , DPO(), EMA() , EMV() , MACD() ,MI(), PVT(), RSI()]

#json.load_s3 = lambda f: json.load(S3().Object(key=f).get()["Body"])
#json.load_s3_t = lambda f: json.load(S3_t().Object(key=f).get()["Body"])
st.title("Portfolio Allocator Neural Network")
print("loading buckets")

data_shape = []
close = []
target = []
ohlcv = []
titles_companies = []
folder="companies"
for file in Path(folder).glob('*.csv'):
    df = pd.read_csv(file)
    
    df = df.set_index("Date")
    data = get_df(df,liss)
    data_shape.append(data.shape[0])
    #print(data_shape[-1])

    if data_shape[0] <= data.shape[0]:
        target.append(data["dfair"].values)
        ohlcv.append(data[["open","high","low","adjclose","volume"]])
        close.append(data.drop(["dfair","open","high","low","close","adjclose","fair"],axis=1))
        titles_companies.append(file)
        print("k",data.shape[0])
    else:
        print(data.shape[0])


close = np.stack(close,axis=1)

pred = []
num=0.7
for i in stqdm(range(close.shape[1]),desc="This is a slow task, training"):

    scale = StandardScaler()
    scaler = StandardScaler()
    
    dff = pd.DataFrame(close[:,i])
    targett = (pd.DataFrame(target).T[i].values.reshape(-1,1))
  
    X_train = dff[:int(len(dff) * num)]
    X_test = dff[int(len(dff) * num):]
    y_train = scaler.fit_transform(targett[:int(len(dff) * num)].reshape(-1,1))
    y_test = scale.fit_transform(targett[int(len(dff) * num):].reshape(-1,1))
    
    X_train = scaler_col(X_train)
    X_test = scaler_col(X_test)
    
    X_train = autoencoder(X_train)
    X_test = autoencoder(X_test)
 
    X_train, y_train = train_chunck(X_train, y_train)
    X_test, y_test = test_chunck(X_test, y_test)
    
    y_pred = model_bilstm_get_prediction(X_train,y_train,X_test,y_test)
    #y_pred = lstm(X_train,y_train,X_test,y_test)
    #print(i)
    pred.append(scale.inverse_transform(y_pred))

portfolio_pred = pd.DataFrame(np.stack(pred,axis=1).reshape(len(pred[0]),np.stack(ohlcv, axis=2).shape[2]))


closes = []

for i in range(close.shape[1]):
    closes.append(ohlcv[i]['adjclose'].values)

close_k = pd.DataFrame(np.stack(closes).T)[-len(y_test):].reset_index(drop=True).pct_change().fillna(0)


kmeans = KMeans(n_clusters=5)
lookback_clusters = 30
clusters = []
for i in stqdm(range(lookback_clusters,X_test.shape[0]),desc="Training KNN"):
    kmeans.fit(close_k.iloc[i-lookback_clusters:i].T.values)
    print(kmeans.labels_)
    clusters.append((kmeans.labels_))
clusters = np.stack(clusters)

portfolio_pred = portfolio_pred[lookback_clusters:].reset_index(drop=True)

close_t = pd.DataFrame(np.stack(closes).T)[-len(y_test)+lookback_clusters:].reset_index(drop=True).pct_change().fillna(0)
list_tk = pd.read_csv("list_tickers.txt")

returns = close_t[-lookback_clusters:].T.mean(axis=1)
variance = close_t[-lookback_clusters:].T.std(axis=1)

returns.columns = ["Returns"]
variance.columns = ["Variance"]

ret_var = pd.concat([returns, variance], axis = 1).dropna()
ret_var["k"] = kmeans.labels_
ret_var["companies"] = list_tk.columns
ret_var.columns = ["Returns","Variance","k","companies"]

st.header("Knn Clusters")

clusters_nav  = st.sidebar.checkbox("Show companies names",False)
if clusters_nav == True:
    st.write("True")

else:
    st.write("false")
fig = px.scatter(ret_var,
                x="Returns",
                y='Variance',
                hover_data='k',
                color="k",
                text="companies",
                title='Clusters stocks - Variance vs Returns')

st.plotly_chart(fig)

best_cluster = []
for i in range(len(clusters)):
    mlo = pd.DataFrame(portfolio_pred.iloc[i], index = clusters[i])
    best_cluster.append(mlo.groupby(mlo.index).sum().idxmax().values[0])
clusters = pd.DataFrame(clusters)

port = []
for i in stqdm(range(clusters.shape[0])):
    price = []
    for j in range(len(clusters.columns)):
        if clusters.iloc[i][j] == best_cluster[i]:
            price.append(close_t.iloc[i][j])
        else:
            price.append(0)
    port.append(price)
closes_filtered = pd.DataFrame(port)

weigh = []
for i in range(len(clusters)):
    price = []
    for j in range(len(clusters.columns)):
        if clusters.iloc[i][j] == best_cluster[i]:
            price.append(portfolio_pred.iloc[i][j])
        else:
            price.append(0)
    weigh.append(price)
pred_filtered = pd.DataFrame(weigh)



allocation = []
for (i,row1) , (row2) in zip(abs(pred_filtered).iterrows(),abs(pred_filtered).sum(axis=1)):
    allocation.append(row1 / row2)
allocation = pd.DataFrame(allocation)

allocation = allocation.replace(1,0.3)

portfolio_allocator = (closes_filtered * allocation) - (closes_filtered * 0.01)

sp = pd.read_csv("benchmark/sp500.csv")
sp_test = pd.DataFrame(sp['Adj Close'][-len(y_test)+lookback_clusters:].pct_change().fillna(0).cumsum()).reset_index(drop=True)

#index_test = pd.DataFrame(sp['Adj Close'][-len(y_test)+lookback_clusters:].pct_change().fillna(0).cumsum()).index


st.header('Stocks returns')
st.line_chart(close_t.cumsum())

st.header('Gains per stock portfolio')
st.line_chart(portfolio_allocator.cumsum())

st.header('Cumsum portfolio total')
st.line_chart(portfolio_allocator.sum(axis=1).cumsum())


st.header("Portfolio weights over time ")
st.dataframe(portfolio_allocator)

sharpe_ratio = portfolio_allocator.sum(axis=1).cumsum().mean() / portfolio_allocator.sum(axis=1).cumsum().std()
#st.title("Sharpe Ratio")
sharpe_ratio_annualize = sharpe_ratio *(252**0.5)
#st.title("Annualized Sharpe Ratio")


sharpe_ratio_sp = (sp_test.cumsum().mean() / sp_test.cumsum().std()) 
sharpe_ratio_annualize_sp = sharpe_ratio_sp *(252**0.5)
portfolio_describe = portfolio_allocator.sum(axis=1).cumsum().describe()
sp_test_describe = sp_test.sum(axis=1).describe()

col1, col2 = st.beta_columns(2)
col11, col22 = st.beta_columns(2)
col111, col222 = st.beta_columns(2)
col1111, col2222 = st.beta_columns(2)
col11111, col22222 = st.beta_columns(2)
col111111, col222222 = st.beta_columns(2)
#col111111, col22222 = st.beta_columns(2)

col1.header("Portfolio NN")
col1.write("Sharp Ratio")
col1.write(sharpe_ratio, use_column_width=True)
col11.write("Annualized Sharp Ratio")
col11.write(sharpe_ratio_annualize, use_column_width=True)
col111.write("std")
col111.write(portfolio_describe["std"], use_column_width=True)
col1111.write("Mean")
col1111.write(portfolio_describe["mean"], use_column_width=True)
col11111.write("Max")
col11111.write(portfolio_describe["max"], use_column_width=True)
col111111.write("Min")
col111111.write(portfolio_describe["min"], use_column_width=True)


col2.header("SP500")
col2.write("Sharpe Ratio")
col2.write(sharpe_ratio_sp.values[0], use_column_width=True)
col22.write("Annualized Sharp Ratio")
col22.write(sharpe_ratio_annualize_sp.values[0], use_column_width=True)
col222.write("std")
col222.write(sp_test_describe["std"], use_column_width=True)
col2222.write("Mean")
col2222.write(sp_test_describe["mean"], use_column_width=True)
col22222.write("Max")
col22222.write(sp_test_describe["max"], use_column_width=True)
col222222.write("Min")
col222222.write(sp_test_describe["min"], use_column_width=True)

alloc_fix = []
for i in range(len(close_t)):
    alloc_fix.append((close_t.iloc[i] * 1/close_t.shape[1]))
alloc_fix = pd.DataFrame(np.stack(alloc_fix))


random_df_dirichlet = pd.DataFrame(np.random.dirichlet(np.ones(np.stack(alloc_fix).shape[0]), len(close_t)))
random_df_dirichlet_ret = random_df_dirichlet * close_t

# Distribucion Random Normal
random_df = []
for i in range(len(close_t)):
    a = abs(np.random.normal(size= np.stack(alloc_fix).shape[0]))
    a /= a.sum()
    random_df.append(a)
random_df = pd.DataFrame(np.stack(random_df))
random_df_returns = random_df * close_t

st.header("Comparacion de carteras")
df_back = pd.concat([sp_test,random_df_returns.cumsum().sum(axis=1),
                        alloc_fix.sum(axis=1).cumsum(),
                        random_df_dirichlet_ret.sum(axis=1).cumsum(),
                        portfolio_allocator.sum(axis=1).cumsum()],axis=1)


df_back.columns = ["SP500","Fix Allocation","Dirichlet","Random Normal Distribution","Cartera NN"]

st.line_chart(df_back)