import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
import tensorflow as tf
from tensorflow import keras
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv3D, Flatten, Dropout, concatenate,BatchNormalization , LSTM , GRU, RNN, Conv2D, Conv1D, GlobalMaxPooling1D, TimeDistributed,Input,Bidirectional, ConvLSTM2D,GlobalAveragePooling1D,Attention
from keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from models import *
import streamlit as st 


def model_bilstm_get_prediction(X_train,y_train,X_test,y_test):
    
    callback = EarlyStopping(monitor='loss', patience=0,restore_best_weights=False)
    #random.seed(40)
    #Simple Bi-lstm
    model = Sequential()
    forward_layer = LSTM(64, activation='relu', return_sequences=True,kernel_regularizer=keras.regularizers.l2(0.03))
    backward_layer = LSTM(32, activation='relu', return_sequences=True, go_backwards=True,kernel_regularizer=keras.regularizers.l2(0.03))
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer,input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.load_weights("models_weight/model.h5")
    model.compile(optimizer = "adam", loss ="mae",metrics =["accuracy"])
    #history = model.fit(X_train, y_train, epochs=10,  batch_size=64, validation_data=(X_test, y_test), callbacks=[callback])
    #plt.plot(history.history["loss"])
    #plt.plot(history.history["val_loss"])

    #model.save_weights("models_weight/model.h5")
    y_pred = model.predict(X_test)
    return y_pred #, model.get_weights()



def autoencoder(X):
    input_dim = X.shape[1]  # 8
    encoding_dim = 20
    input_layer = Input(shape=(input_dim, ))
    encoder_layer_1 = Dense(X.shape[1], activation="tanh", activity_regularizer=regularizers.l1(0.001))(input_layer)
    encoder_layer_2 = Dense(6, activation="tanh")(encoder_layer_1)
    encoder_layer_3 = Dense(encoding_dim, activation="tanh")(encoder_layer_2)
    encoder = Model(inputs=input_layer, outputs=encoder_layer_3)
    encoder.summary()
    return encoder.predict(X)

def lstm(X_train,y_train,X_test,y_test):
    #LSTM model
    model = Sequential()
    model.add((LSTM(64, activation = "relu",input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences= False,kernel_regularizer=keras.regularizers.l2(0.12))))
    #model.add(LSTM(30, activation = "relu", return_sequences= True,kernel_regularizer=keras.regularizers.l2(0.12)))
    #model.add(Bidirectional(ConvLSTM2D(10)))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Flatten())
    #model.add(TimeDistributed(Flatten()))
    #model.add(Attention(32))
    model.add(Dense(10))
    model.add(Dense(1))
    #model.add(TimeDistributed(Dense(1)))
    #model.build(input_shape=(X_train.shape[1],X_train.shape[2],1))
    model.load_weights("models_1/model.h5")
    model.compile(optimizer = "adam", loss ="mae",metrics =["accuracy"])
    #history = model.fit(X_train, y_train, epochs=20,  batch_size=64, validation_data=(X_test, y_test))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])

    #model.save_weights("models_1/model.h5")
    y_pred = model.predict(X_test)
    return y_pred #, model.get_weights()