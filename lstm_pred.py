import time
#import warnings
import numpy as np
from numpy import newaxis
import tensorflow as tf
from tensorflow import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from data_in import data_load

data = 'data/daily_sp500/daily/table_nvda.csv'
input_arr = [50, 0.33]
# load the data
X_test, X_train, y_test, y_train, max = data_load(data, input_arr)

print(X_train.shape) #(17, 200, 4)
print(y_train.shape) #(3240,)

# Build the LSTM Model
model = Sequential()
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))

start = time.time()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])



print('compilation time: ', time.time()-start, 'seconds')

model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=1,
    validation_split=0.05)
