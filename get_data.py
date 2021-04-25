import pandas as pd
import numpy as np
from ta import *
from bucket import *
import pandas_datareader.data as web
import os 
import time

from pandas_datareader import data,wb
import json, boto3
import matplotlib.dates as dates
from datetime import date , timedelta
import streamlit as st
from pathlib import Path

start = pd.to_datetime("2015-01-02")
end = pd.to_datetime(date.today())

sp = data.DataReader('SPY', 'yahoo',start,end)
#sp.to_csv("benchmark/sp500.csv")

url = "https://raw.githubusercontent.com/leosmigel/analyzingalpha/master/sp500-historical-components-and-changes/sp500_history.csv"
dff = pd.read_csv(url,index_col=0,parse_dates=[0])
dff.date = pd.to_datetime(dff.date)
filtered = dff[dff.date > pd.to_datetime("2015-01-01 00:00:00")]

removed = []
added = [] 
for i in range(len(filtered)):
    if filtered["variable"].iloc[i] == "removed_ticker":
        x = filtered["date"].iloc[i] , filtered["value"].iloc[i]
        removed.append(x)
    else:
        xy = filtered["date"].iloc[i], filtered["value"].iloc[i]
        added.append(xy)


removed_df = pd.DataFrame(removed)
added_df = pd.DataFrame(added)

repeated = []
for i in range(len(added_df)):
    for j in range(len(removed_df)):
        if added_df[1][i] == removed_df[1][j]:
            repeated.append(removed_df[1][j])

unique_tickers = np.unique(pd.concat([filtered.value,df]))

added_ticks = []
for i in range(len(added)):
    added_ticks.append(added[i][1])

removed_ticks = []
for i in range(len(removed)):
    removed_ticks.append(removed[i][1])


import pandas as pd
url = 'https://raw.githubusercontent.com/leosmigel/analyzingalpha/master/sp500-historical-components-and-changes/sp500_constituents.csv'
df = pd.read_csv(url,index_col=0,parse_dates=[0])

df = df["ticker"].reset_index(drop=True)

start = pd.to_datetime("2020-04-22")
end = pd.to_datetime(date.today())

good_dt = []
need_alphavantage = []
for i in unique_tickers:
    try:
        dat = data.DataReader(f'{i}', 'yahoo',start,end)
        good_dt.append(i)
        print("k",i)
    except:
        print(i)
        need_alphavantage.append(i)
        pass
    
pd.DataFrame(good_dt).to_csv("new_/good_df.csv")
pd.DataFrame(need_alphavantage).to_csv("new_/need_alphavantage.csv")

good_df = good_df.iloc[:,1:].values

alphavantage_not_taken = []
exept = []
for i in unique_tickers:
    if i in good_dt:
        try:
            if i in repeated:
                start = pd.to_datetime(dff[dff.value == i]["date"].reset_index(drop=True))[0]
                end = pd.to_datetime(dff[dff.value == i]["date"].reset_index(drop=True))[1]
                dat = data.DataReader(i, 'yahoo',start,end)
                lm = pd.concat([dat,sp],axis=1).fillna(0).iloc[:,:5]
                lm.to_csv(f"companies/{i}.csv")
                print("repeated",i)
            elif i in removed_ticks and i not in repeated:
                start = start = pd.to_datetime("2015-01-01")
                end = pd.to_datetime(dff[dff.value == i]["date"].reset_index(drop=True))[1]
                dat = data.DataReader(i, 'yahoo',start,end)
                lm = pd.concat([dat,sp],axis=1).fillna(0).iloc[:,:5]
                lm.to_csv(f"companies/{i}.csv")
                print("in removed",i)
            elif i in added_ticks and i not in repeated:
                start = pd.to_datetime(dff[dff.value == i]["date"].reset_index(drop=True))
                end = pd.to_datetime(date.today())
                dat = data.DataReader(i, 'yahoo',start,end)
                print(i)
                lm = pd.concat([dat,sp],axis=1).fillna(0).iloc[:,:5]
                lm.to_csv(f"companies/{i}.csv")
                print("in added",i)
    
            elif i not in repeated and i not in removed_ticks and i not in added_ticks:
                start = pd.to_datetime("2015-01-01")
                end = pd.to_datetime(date.today())
                dat = data.DataReader(i, 'yahoo',start,end)
                dat = pd.concat([dat,sp],axis=1).fillna(0).iloc[:,:5]
                dat.to_csv(f"companies/{i}.csv")
                print(i)
               
        except:
            exept.append(i)
            print("pass")
            pass
            
    else:
        print("AL")
        alphavantage_not_taken.append(i)
            


new_list = pd.concat([pd.DataFrame(exept),pd.DataFrame(alphavantage_not_taken)]).values



alphavantage_not = []
count = 0 
for i in new_list:
    i = i[0]
    print(count)
    try:
        if i in repeated:
            start = pd.to_datetime(dff[dff.value == i]["date"].reset_index(drop=True))[0]
            end = pd.to_datetime(dff[dff.value == i]["date"].reset_index(drop=True))[1]
            dat = web.DataReader(i, "av-daily", start=start, end=end,api_key=("7F9F15N5NKOPX907"))
            lm = pd.concat([dat,sp],axis=1).fillna(0).iloc[:,:5]
            lm.to_csv(f"companies/{i}.csv")
            print("repeated",i)
            count = count +1
            if count == 4:
                time.sleep(61)
                count=0
        elif i in removed_ticks and i not in repeated:
            start = start = pd.to_datetime("2015-01-01")
            end = pd.to_datetime(dff[dff.value == i]["date"].reset_index(drop=True))[1]
            dat = web.DataReader(i, "av-daily", start=start, end=end,api_key=("7F9F15N5NKOPX907"))
            lm = pd.concat([dat,sp],axis=1).fillna(0).iloc[:,:5]
            lm.to_csv(f"companies/{i}.csv")
            print("in removed",i)
            count = count +1
            if count == 4:
                time.sleep(61)
                count=0
        elif i in added_ticks and i not in repeated:
            start = pd.to_datetime(dff[dff.value == i]["date"].reset_index(drop=True))[0]
            end = pd.to_datetime(date.today())
            dat = web.DataReader(i, "av-daily", start=start, end=end,api_key=("7F9F15N5NKOPX907"))
            lm = pd.concat([dat,sp],axis=1).fillna(0).iloc[:,:5]
            lm.to_csv(f"companies/{i}.csv")
            print("in added",i)
            count = count +1
            if count == 4:
                time.sleep(61)
                count=0
        elif i not in repeated and i not in removed_ticks and i not in added_ticks:
            start = pd.to_datetime("2015-01-01")
            end = pd.to_datetime(date.today())
            dat = web.DataReader(i, "av-daily", start=start, end=end,api_key=("7F9F15N5NKOPX907"))
            lm = pd.concat([dat,sp],axis=1).fillna(0).iloc[:,:5]
            lm.to_csv(f"companies/{i}.csv")
            count = count +1
            
            if count == 4:
                time.sleep(61)
                count=0
                
        
    except:
        alphavantage_not.append(i)
        if count == 4:
                time.sleep(61)
                count=0
        print("passed",i)
        pass


count = 0 
not_taken = []
for i in alphavantage_not:
    try:
        we = web.DataReader(i, "av-daily",api_key=("7F9F15N5NKOPX907"))
        if we.index[0] > "2015-01-02":
            conc = pd.concat([we, sp],axis=1).fillna(0).iloc[:,:5]
            conc.to_csv(f"companies/{i}.csv")
            count = count +1
            print("conc",i)
            if count == 4:
                time.sleep(61)
                count=0
        else: 
            
            conc = we.loc["2015-01-02":]
            conc.to_csv(f"companies/{i}.csv")
            print("conc",i)
            count = count +1
            if count == 4:
                time.sleep(61)
                count=0
        
    except:
        not_taken.append(i)
        print("passed",i)
        pass

#json.dump_s3 = lambda obj, f: S3().Object(key=f).put(Body=json.dumps(obj, sort_keys=True, default=str).encode('UTF-8'))
#json.dump_s3_t = lambda obj, f: S3_t().Object(key=f).put(Body=json.dumps(obj, sort_keys=True, default=str).encode('UTF-8'))
