import time
#import warnings
import numpy as np
from numpy import newaxis
import tensorflow as tf
from tensorflow import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
from support_functions import *

#modelName = '_32_300_05'    #model_batch_epochs_split

data = 'data/daily_sp500/daily/table_nvda.csv'
input_arr = [50, 0.33] ##### [window, %test, batch_size, epochs, val_split]
hyperParam = [512, 1, 0.05]    # [batch_size, epochs, validation_split]
# load the data
X_test, X_train, y_test, y_train, max = data_load(data, input_arr)

# Build the LSTM Model
model = Sequential()
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))

start = time.time()
model.compile(loss='mse', optimizer='adam')    # metrics=['accuracy']

print('compilation time: ', time.time()-start, 'seconds')

model.fit(
    X_train,
    y_train,
    batch_size=hyperParam[0],
    epochs=hyperParam[1],
    validation_split=hyperParam[2])

save_model(model, hyperParam)
print('model saved')


# plot the predictions
predictions = predict_sequences_multiple(model, X_test, 50, 50)
plot_results_multiple(predictions, y_test, 50)
