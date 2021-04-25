import numpy as np 
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
import pandas as pd

def train_chunck(X_train, y_train):
    n_past = 30
    n_future = 1
    X_train1 = []
    y_train1 = []
    for i in range(n_past, len(X_train) - n_future +1 ):
            X_train1.append(X_train[i - n_past:i,0:X_train.shape[1]])
            y_train1.append(y_train[i:i + n_future])

    X_train ,y_train = np.stack(X_train1), np.stack(y_train1)
    return X_train ,y_train



def test_chunck(X_test, y_test):
    n_past = 30
    n_future = 1
    X_test1 = []
    y_test1 = []
    for i in range(n_past, len(X_test) - n_future +1 ):
            X_test1.append(X_test[i - n_past:i,0:X_test.shape[1]])
            y_test1.append(y_test[i:i + n_future])

    X_test, y_test = np.stack(X_test1), np.stack(y_test1)
    return X_test, y_test


def scaler_col(X):
    scalers = {}
    X = pd.DataFrame(X)
    for i in range(1,X.shape[1]):
        scalers[i] = StandardScaler()
        X.iloc[:,i:i+1] = scalers[i].fit_transform(X.iloc[:,i-1:i]) 
    return X