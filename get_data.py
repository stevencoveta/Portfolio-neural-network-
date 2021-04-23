import pandas as pd
import numpy as np
from ta import *
from bucket import *

from pandas_datareader import data,wb
import json, boto3
import matplotlib.dates as dates
from datetime import date , timedelta
import streamlit as st
from pathlib import Path

start = pd.to_datetime("2011-06-01")
end = pd.to_datetime(date.today())

sp = data.DataReader('SPY', 'yahoo',start,end)
sp.to_csv("benchmark/sp500.csv")

ticker_list = ['ANTM',
 'AON',
 'AOS',
 'APA',
 'APD',
 'APH',
 'APTV',
 'ARE',
 'ARNC',
 'ATO',
 'ATVI',
 'AVB',
 'AVGO',
 'AVY',
 'AWK',
 'AXP',
 'AZO',
 'BA',
 'BAC',
 'BAX',
 'BBY',
 'BDX',
 'BEN',
 'BIIB',
 'BK',
 'BKNG',
 'BLK',
 'BLL',
 'BMY',
 'BR',
 'BSX',
 'BWA']


#file1 = open('list_tickers.txt', 'r')
#Lines = file1.readlines()
 
#for line in ticker_list:
    #try:
        #print(line.strip())
        #dat = data.DataReader(f'{line}', 'yahoo',start,end)
        #print(dat.shape)
        #dat.to_csv(f"companies/{line}.csv")
    #except:
        #print(line)
        #pass
files = pd.read_csv("titles_companies.csv")

for i in files:
    print(i)





#json.dump_s3 = lambda obj, f: S3().Object(key=f).put(Body=json.dumps(obj, sort_keys=True, default=str).encode('UTF-8'))
#json.dump_s3_t = lambda obj, f: S3_t().Object(key=f).put(Body=json.dumps(obj, sort_keys=True, default=str).encode('UTF-8'))


#df_2D = pd.DataFrame(np.stack(close, axis = 1).reshape(len(df), -1))
#print(df_2D.shape)

#df_dict = df_2D.to_dict('records')
#df_2D.to_csv("close.csv")

#json.dump_s3(df_dict, "key")

#target_df = pd.DataFrame(target)
#target_df.to_csv("target.csv")
#json.dump_s3_t(target_df, "key")
#ohlcv_2D = pd.DataFrame(np.stack(ohlcv, axis = 1).reshape(len(df),-1))
#ohlcv_2D.to_csv("ohlcv.csv")

#print("data collected")