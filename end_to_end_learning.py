import tensorflow as tf
from keras.layers import Dense, Input, LSTM
from keras import Sequential

from ParamsHandler import ParamsHandler
from ModelHandler import ClassifiersFactory
from cv import get_data, train_test_split

import os
import gc
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, mean_absolute_error

warnings.filterwarnings("ignore")


def neural_network(timesteps, data_dim):
    n_output = 2
    model = Sequential()
    model.add(LSTM(128, input_shape=(1, timesteps, data_dim)))
    model.add(Dense(n_output, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def main():
    params = ParamsHandler.load_parameters('settings')

    # experiment variables
    dataset = params['dataset']
    mode = params['mode']
    classifiers = params['classifiers']
    subsets = params['subsets']  # modes
    seeds = params['seeds']
    nfolds = params['folds']

    # paths
    paths = params['paths'][os.name]
    input = paths['input']  # baseline_processed
    output = paths['output']  # eye_data_path
    diag = paths['plog']
    data = os.path.join(paths['data'], mode)
    results = os.path.join('results', mode)

    # multiprocessing
    mp_flag = params['multiprocessing']
    n_cores = params['n_cores'][os.name]

    # getting valid_pids from TasksTimestamps.csv and meta_data collected before - leaves us with 157 participants
    ttf = pd.read_csv(paths['ttf'])
    ttf_pids = ttf['StudyID'].unique()

    meta_data = pd.read_csv(os.path.join(data, 'meta_data.csv'))
    valid_pids = np.intersect1d(ttf_pids, meta_data['PID'])

    # get data
    pid_all_data = get_data(valid_pids)

    plog = pd.read_csv(diag)

    pid = 'EA-123'
    input_data = pid_all_data[pid]['input']
    diagnosis = np.array([[1, 0] if pid.startswith('E') else [0, 1]][0])

    x = input_data[:, :-2]
    x2 = np.reshape(x, (1, x.shape[0], x.shape[1]))
    net = neural_network(580, 6)
    history = net.fit(x2, diagnosis, epochs=10)
