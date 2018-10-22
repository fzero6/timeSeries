import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd

style.use('ggplot')
input_arr = [50, 0.33]  #[legnth, test_split_pct]
data_file = 'data/daily_sp500/daily/table_nvda.csv'

# fucntion to import stock data in the format OHLCV
def data_load(csv_file, input_arr):

    length = input_arr[0]
    test_size = input_arr[1]

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
    print('x_data =', x_data.shape)
    y_data = data_new[:, -1]
    print('y_data =', y_data.shape)
    #print(y_data.shape)

    # create 3D array to stack data set into smaller chucnks
    result = data_stack(x_data)
    result = np.array(result)
    print('result = ', result.shape)
    y_result = data_stack(y_data)
    y_result = np.array(y_result)
    print('y_result = ', y_result.shape)
    #split the data for testing and training
    #split = int((len(data_new) * test_size) / length)
    split = int(len(data_new) * test_size)

    #slice the arrays for training and testing
    x_test = result[:split, :, :]
    x_train = result[split:, :, :]
    y_test = y_data[:x_test.shape[0]]#y_data[:split]
    y_train = y_data[-x_train.shape[0]:]
    print('x_test = {}'.format(x_test.shape))
    print('y_test = {}'.format(y_test.shape))
    print('x_train = {}'.format(x_train.shape))
    print('y_train = {}'.format(y_train.shape))

    return x_test, x_train, y_test, y_train, max_vals

def data_stack(data,seq_len=50):
    result = []
    for i in range(len(data) - seq_len):
        result.append(data[i: i + seq_len])
    return result



def normalize(data_block):

    return data_block_norm


x_test, x_train, y_test, y_train, max = data_load(data_file, input_arr)


'''
def normalize_data(data_window):
    # function normalizes the data for separately for each window
    normalized_data = []
    for window in data_window:
        for p in window:
            normalized_window = ((float(p) / float(window[0])) - 1)
            normalized_data.append(normalized_window)
'''
