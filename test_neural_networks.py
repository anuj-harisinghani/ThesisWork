"""

For testing Neural Networks


"""

import tensorflow as tf
from keras.layers import Dense, Input
from keras import Sequential

from ParamsHandler import ParamsHandler
from ModelHandler import ClassifiersFactory
from cv import get_data, train_test_split

import os
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, mean_absolute_error

warnings.filterwarnings("ignore")


def neural_network(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(18, input_shape=(n_inputs,), activation='relu'))
    model.add(Dense(8, activation='linear'))
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam')
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
    input = paths['input']      # baseline_processed
    output = paths['output']    # eye_data_path
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

    # experimentation
    seed = 0
    fold = 0
    strategy = 'all'
    clf = 'NeuralNetwork'
    x_train, y_train, x_test, y_test, a, b, c, d = train_test_split(pid_all_data, strategy, seed, fold)
    chain = neural_network(9, 6)
    chain.fit(x_train, y_train, epochs=20)
    preds = chain.predict(x_test)
    error = mean_absolute_error(y_true=y_test, y_pred=preds)
    print(error)


main()
