import boto3
import numpy as numpy
import pandas as pd 
from pathlib import Path
import io 
import os

def check_data():
    s3c = boto3.client('s3', aws_access_key_id='', aws_secret_access_key='')
    s3c._request_signer.sign = (lambda *args, **kwargs: None)
  
    if (Path.cwd() / 'companies').exists(): 
        print("all good")
    else:
        os.makedirs('companies')
        company = []
        with open('names.txt', 'r') as filehandle:
            company = [current_place.rstrip() for current_place in filehandle.readlines()]

        for i in company:
            print(i) 
            key = f"{i}.csv"
            obj = s3c.get_object(Bucket= "csv-companies" , Key = key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf8')
            df.columns = ["Date","open","high","low","close","volume"]
            df.to_csv(f'companies/{key}')
        
