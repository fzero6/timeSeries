import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd

style.use('ggplot')

data_file = 'data/daily_sp500/daily/table_nvda.csv'

# fucntion to import stock data in the format OHLCV
def data_load(csv_file, test_size):
    length = 200 # data chunks

    df = pd.read_csv(csv_file, parse_dates=True,
                      index_col=0, names=['Date','NA' ,'Open', 'High', 'Low', 'Close', 'Volume'])
    df = df.drop('NA', 1)

    # reorganize dataFrame to OHLVC
    reorder = ['Open', 'High', 'Low', 'Volume', 'Close']
    df = df.reindex(columns=reorder)
    max = df.max()
    df = df/max
    max_vals = np.array(max)

    #convert data from Pandas DF to Numpy Array
    data = np.array(df)
    # need data set to have equal sections
    # find quotiant and slice data for even slices
    quot = len(data) % length
    data_new = data[quot:]

    x_data = data_new[:, :-1]
    y_data = data_new[:, -1]
    #print(y_data.shape)

    # create 3D array to stack data set into smaller chucnks
    # initialize variables for FOR loop
    samples = list()
    for i in range(0,len(x_data), length):
        sample = x_data[i: i + length]
        samples.append(sample)
    #print(len(samples))
    result = np.array(samples)

    #split the data for testing and training
    number = int((len(data_new) * test_size) / length)
    y_number = int(len(data_new) * test_size)
    #slice the arrays for training and testing
    X_test = result[:number, :, :]
    X_train = result[number:, :, :]
    y_test = y_data[:y_number]
    y_train = y_data[y_number:]
    #print('y_test = {}'.format(y_test))
    #print('y_train = {}'.format(y_train))

    return X_test, X_train, y_test, y_train, max_vals

X_test, X_train, y_test, y_train, max = data_load(data_file, 0.33)

'''
print(X_test.shape)
print(X_train.shape)
print(y_train.shape)
print(y_test.shape)
#print(max)
'''
'''
def normalize_data(data_window):
    # function normalizes the data for separately for each window
    normalized_data = []
    for window in data_window:
        for p in window:
            normalized_window = ((float(p) / float(window[0])) - 1)
            normalized_data.append(normalized_window)
'''
