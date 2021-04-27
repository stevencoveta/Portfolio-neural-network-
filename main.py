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
from numpy import inf
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
import boto3
from chuncks import * 
from models import *
from ta import *
import io
#from get_S3_data import * 


st.set_page_config(layout="wide")

s3c = boto3.client('s3', aws_access_key_id='', aws_secret_access_key='')
s3c._request_signer.sign = (lambda *args, **kwargs: None)

@st.cache(allow_output_mutation=True)
def company():
    company = []
    with open('names.txt', 'r') as filehandle:
        company = [current_place.rstrip() for current_place in filehandle.readlines()]
    return company 

company = company()

numb = st.sidebar.number_input("Input randomly chosen number of stocks",1,30,5)

rand = []
for i in range(numb):
    rand.append(random.choice(company))



col1, col2 = st.beta_columns(2)

col1.header("Date Today")
col1.write(pd.datetime.now().date(), use_column_width=True)

col2.header("Estrategia :bulb:")
col2.write("Usar un kmeans para el clustering de las empresas y invertir en el cluster con la mejor prediccion futura, daily ", use_column_width=True)


liss = [OBV() , CCI() , CHV(), CMF() , DPO(), EMA() , EMV() , MACD() ,MI(), PVT(), RSI()]

st.title("Portfolio Allocator Neural Network")
print("loading buckets")


sp = pd.read_csv("benchmark/sp500.csv")
sp = sp.set_index("Date")

@st.cache(allow_output_mutation=True)
def cache_comp(sp,rand):
    data_shape = []
    close = []
    target = []
    ohlcv = []
    titles_companies = []
    folder="companies"
    #for file in Path(folder).glob('*.csv'):
    for i in rand:
        key = f"{i}.csv"
        obj = s3c.get_object(Bucket= "csv-companies" , Key = key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf8')
        df.columns = ["Date",'high','low', 'open', 'close', 'volume']
        df = df.set_index("Date",drop= True)  
        print(df) 
        #df = pd.read_csv(file)
        #df = df.set_index("Date")
        dat = get_df(df,liss)
        #print(dat.shape)
        data = pd.concat([dat,sp],axis=1).fillna(0).iloc[:,:dat.shape[1]]
        data_shape.append(data.shape[0])
        print("data", data.shape)
       
        if data_shape[-1] == sp.shape[0]:
            target.append(data["dfair"].values)
            ohlcv.append(data[["open","high","low","close","volume"]])
            close.append(data.drop(["dfair","open","high","low","close","fair"],axis=1))
            #titles_companies.append(str(file).split("/")[1].split(".")[0])
            titles_companies.append(i)
            print(i)
            print("k",data.shape[0])
        else:
            print(data.shape[0])
    
    close = np.stack(close,axis=1)
    return close,target,ohlcv,data_shape,titles_companies

catching = cache_comp(sp,rand)
close = catching[0]
target = catching[1]
ohlcv = catching[2]
data_shape = catching[3]
titles_companies = catching[4]

with open("titles_companies.txt", "w") as output:
    output.write(str(titles_companies)[1:-1])


select = st.sidebar.selectbox("Select nn model",["BI-LSTM","LSTM"])

@st.cache(allow_output_mutation=True)
def main_model(close,target,model):
    pred = []
    num=0.7
    for i in stqdm(range(close.shape[1])):

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
        if model == "BI-LSTM":
            y_pred = model_bilstm_get_prediction(X_train,y_train,X_test,y_test)
        elif model == "LSTM":
            y_pred = lstm(X_train,y_train,X_test,y_test)
    
        pred.append(scale.inverse_transform(y_pred))
    return pred , y_test , X_test


main_mod = main_model(close,target,select)
pred = main_mod[0]
y_test = main_mod[1]
X_test = main_mod[2]

portfolio_pred = pd.DataFrame(np.stack(pred,axis=1).reshape(len(pred[0]),np.stack(ohlcv, axis=2).shape[2]))


closes = []

for i in range(close.shape[1]):
    closes.append(ohlcv[i]['close'].values)

close_k = pd.DataFrame(np.stack(closes).T)[-len(y_test):].reset_index(drop=True).pct_change().fillna(0)
close_k[close_k == -inf] = 0
close_k[close_k == inf] = 0

lookback_clusters = 30

@st.cache(suppress_st_warning=True)
def knn(lookback_clustersn, k ):
    kmeans = KMeans(n_clusters=k)
    clusters = []
    for i in stqdm(range(lookback_clusters,X_test.shape[0]),desc="Training KNN"):
        kmeans.fit(close_k.iloc[i-lookback_clusters:i].T.values)
        print(kmeans.labels_)
        clusters.append((kmeans.labels_))
    clusters = np.stack(clusters)
    return clusters , kmeans.labels_


list_tk = pd.read_csv("titles_companies.txt")


kk = st.sidebar.slider("Number of Clusters",1,29,10)
kn = knn(lookback_clusters, kk)

clusters = kn[0]
kmeans_labels_ = kn[1]

portfolio_pred = portfolio_pred[lookback_clusters:].reset_index(drop=True)

close_t = pd.DataFrame(np.stack(closes).T)[-len(y_test)+lookback_clusters:].reset_index(drop=True).pct_change().fillna(0)
close_t[close_t == -inf] = 0
close_t[close_t == inf] = 0


returns = close_t[-lookback_clusters:].T.mean(axis=1)
variance = close_t[-lookback_clusters:].T.std(axis=1)

returns.columns = ["Returns"]
variance.columns = ["Variance"]

ret_var = pd.concat([returns, variance], axis = 1).dropna()

ret_var["k"] = kmeans_labels_
ret_var["companies"] = list_tk.columns
ret_var.columns = ["Returns","Variance","k","companies"]

st.header("Knn Clusters")


clusters_nav  = st.checkbox("Show companies names",False)
if clusters_nav == False:
    fig = px.scatter(ret_var,
                x="Returns",
                y='Variance',
                hover_data='k',
                color="k",
                title='Clusters stocks - Variance vs Returns')

    st.plotly_chart(fig)
else:
    fig = px.scatter(ret_var,
                x="Returns",
                y='Variance',
                hover_data='k',
                color="k",
                text="companies",
                title='Clusters stocks - Variance vs Returns')

    st.plotly_chart(fig)


range_n_clusters = list(range(2,30))
clust = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(close_k.iloc[-lookback_clusters:].T.values)
    #print(cluster_labels)
    silhouette_avg = silhouette_score(close_k.iloc[-lookback_clusters:].T.values, cluster_labels)
    clustt = n_clusters, silhouette_avg
    clust.append(clustt)

silou = pd.DataFrame(clust, columns = ["clusters","Silouhette score"]).set_index("clusters",drop=True)
k = silou.idxmax()[0]



st.header('Kmeans Silhouette Score ')
fig = px.line(silou, x=range_n_clusters , y=silou["Silouhette score"], labels={
                     "value": "Silouhette Score",
                     "index": "Clusters",
                     "variable": "Companies"
                 },title='Optimal number of clusters')

st.write(fig)
st.write("Best number of clusters is" , k)

st.header("Correlation heatmap of SP500 Stocks")
ndf_corr = pd.DataFrame(closes).T.pct_change().dropna().corr()
ndf_corr.columns = list_tk.columns
ndf_corr.index = list_tk.columns
fig = px.imshow(ndf_corr,labels={
                     "value": "Companies (%)",
                     "index": "Companies",
                     "variable": "Correlation "
                 },title='Correlation Matrix')
st.write(fig)


best_cluster = []
for i in range(len(clusters)):
    mlo = pd.DataFrame(portfolio_pred.iloc[i], index = clusters[i])
    best_cluster.append(mlo.groupby(mlo.index).sum().idxmax().values[0])
clusters = pd.DataFrame(clusters)

@st.cache
def close_filtere(clusters, best_cluster):
    port = []
    for i in (range(clusters.shape[0])):
        price = []
        for j in range(len(clusters.columns)):
            if clusters.iloc[i][j] == best_cluster[i]:
                price.append(close_t.iloc[i][j])
            else:
                price.append(0)
        port.append(price)
    closes_filtered = pd.DataFrame(port)
    return closes_filtered

closes_filtered = close_filtere(clusters, best_cluster)

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
sp_date = sp["Date"][-len(y_test)+lookback_clusters:]

close_cum = pd.DataFrame(close_t.cumsum())
close_cum.columns = list_tk.columns
close_cum.index = sp_date.values

st.header('Stocks returns')
fig = px.line(close_cum, x=close_cum.index, y=close_cum.columns, labels={
                     "value": "Returns (%)",
                     "index": "Date",
                     "variable": "Companies"
                 },title='Individual Stocks Returns of SP500')
st.write(fig)


port_cum = pd.DataFrame(portfolio_allocator.cumsum())
port_cum.columns = list_tk.columns
port_cum.index = sp_date.values

st.header('Gains per stocks portfolio')
fig = px.line(port_cum, x=close_cum.index, y=close_cum.columns, labels={
                     "value": "Returns (%)",
                     "index": "Date",
                     "variable": "Companies"
                 },title='Individual Stocks Returns of NN Portfolio')
st.write(fig)


portfolio_allocator_ = portfolio_allocator.copy()
portfolio_allocator_.columns = list_tk.columns
portfolio_allocator_.index = sp_date.values
st.header("Portfolio allocator weights*returns over time ")
st.write(portfolio_allocator_)


st.header('Cumsum Portfolio total')

port_cum_tot = pd.DataFrame(portfolio_allocator.sum(axis=1).cumsum())
port_cum_tot.index = sp_date.values
port_cum_tot.columns = ["NN"]

sp_show = st.checkbox("Show SP500 returns", False)
if sp_show == False:
    fig = px.line(port_cum_tot, x=port_cum_tot.index, y=port_cum_tot.columns, labels={
                     "value": "Returns (%)",
                     "index": "Date",
                     "variable": "Portfolio NN "
                 },title='Total Returns of NN Portfolio')
    st.write(fig)

else:
    df_back = pd.concat([sp_test,portfolio_allocator.sum(axis=1).cumsum()],axis=1)
    df_back.columns = ["SP500","Cartera NN"]
    df_back.index = portfolio_allocator_.index 

    fig = px.line(df_back, x=df_back.index, y=df_back.columns, labels={
                     "value": "Returns (%)",
                     "index": "Date",
                     "variable": "Companies "
                 },title='Benchmarks comparaison')
    st.write(fig)




sharpe_ratio = portfolio_allocator.sum(axis=1).cumsum().mean() / portfolio_allocator.sum(axis=1).cumsum().std()
sharpe_ratio_annualize = sharpe_ratio *(252**0.5)


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
col1111111, col2222222 = st.beta_columns(2)

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

a = pd.DataFrame(portfolio_allocator.sum(axis=1).cumsum())
b = pd.DataFrame(sp_test)
c = pd.concat([a,b], axis = 1).corr().iloc[0][1:][0]
col1111111.write("Corr")
col1111111.write(c , use_column_width=True)



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
col2222222.write("Corr")
col2222222.write(1, use_column_width=True)


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
df_back = pd.concat([sp_test,
                    random_df_returns.cumsum().sum(axis=1),
                    alloc_fix.sum(axis=1).cumsum(),
                    random_df_dirichlet_ret.sum(axis=1).cumsum(),
                    portfolio_allocator.sum(axis=1).cumsum()],axis=1)


df_back.columns = ["SP500","Fix Allocation","Dirichlet","Random Normal Distribution","Cartera NN"]
df_back.index = portfolio_allocator_.index 


fig = px.line(df_back, x=df_back.index, y=df_back.columns, labels={
                     "value": "Returns (%)",
                     "index": "Date",
                     "variable": "Companies "
                 },title='Benchmarks comparaison')
st.write(fig)

