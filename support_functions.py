import os
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from numpy import newaxis
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

    # x_data = input values, y_inputs = closing prices to make predictions
    x_data = data_new[:, :-1]
    y_data = data_new[:, -1]

    # create 3D array to stack data set into smaller chucnks
    result = data_stack(x_data)
    result = np.array(result)

    #split the data for testing and training
    #split = int((len(data_new) * test_size) / length)
    split = int(len(data_new) * test_size)

    #slice the arrays for training and testing
    x_test = result[:split, :, :]
    x_train = result[split:, :, :]
    y_test = y_data[:x_test.shape[0]]#y_data[:split]
    y_train = y_data[-x_train.shape[0]:]

    return x_test, x_train, y_test, y_train, max_vals


def data_stack(data,seq_len=50):
    result = []
    for i in range(len(data) - seq_len):
        result.append(data[i: i + seq_len])
    return result


def normalize(data_block):
    #function normalizes the data for separately for each window
    normalized_data = []
    for window in data_window:
        for p in window:
            normalized_window = ((float(p) / float(window[0])) - 1)
            normalized_data.append(normalized_window)
    return data_block_norm


def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    print('yo')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def nameParse(hyperParam):
    modelName = ''
    for param in hyperParam:
        if isinstance(param, int):
            modelName = modelName + '_' + str(param)
        else:
            name = str(param).split('.')[1]
            modelName = modelName + '_' + str(name)
    return modelName


def save_model(model, hyperParam):
    path = os.getcwd() + '/trained_models/'
    # set model save name based on chosen hyperparameters
    modelName = nameParse(hyperParam)
    # save model architecture
    model_yaml = model.to_yaml()
    with open(path + 'model' + modelName + '.yaml', 'w') as yaml_file:
        yaml_file.write(model_yaml)
    #save model weights
    model.save_weights(path + 'wts' + modelName + '.h5')


x_test, x_train, y_test, y_train, max = data_load(data_file, input_arr)
