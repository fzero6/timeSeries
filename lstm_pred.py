import time
#import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from data_in import data_load

data = 'data/daily_sp500/daily/table_nvda.csv'

# load the data
X_test, X_train, y_test, y_train, max = data_load(data, 0.1)

# Build the LSTM Model
model = Sequential()

model.add(LSTM(
    400,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    400,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    len(y_train)))

model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time: ', time.time()-start, 'seconds')

model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=1,
    validation_split=0.05)
