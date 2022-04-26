import keras.backend
import tensorflow as tf
from keras.layers import Dense, Input, LSTM, Dropout
from keras import Sequential

from ParamsHandler import ParamsHandler
from ModelHandler import ClassifiersFactory
from cv import get_data, train_test_split

import os
import random
import gc
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, mean_absolute_error

warnings.filterwarnings("ignore")

dev = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(dev, enable=True)


def neural_network(timesteps, data_dim):
    n_output = 2
    model = Sequential()
    model.add(LSTM(128, input_shape=(None, data_dim)))
    model.add(Dropout(0.5))
    # model.add(Dense(100, activation='relu'))
    model.add(Dense(n_output, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def subset_data(input_train, input_test, strategy):
    fold_train = fold_test = None
    if strategy == 'left':
        fold_train = np.array([input_train[i][:, :3] for i in range(len(input_train))])
        fold_test = np.array([input_test[j][:, :3] for j in range(len(input_test))])

    elif strategy == 'right':
        fold_train = np.array([input_train[i][:, 3:6] for i in range(len(input_train))])
        fold_test = np.array([input_test[j][:, 3:6] for j in range(len(input_test))])

    elif strategy == 'average':
        fold_train = np.array([np.mean((input_train[i][:, :3], input_train[i][:, 3:6]), axis=0) for i in range(len(input_train))])
        fold_test = np.array([np.mean((input_test[i][:, :3], input_test[i][:, 3:6]), axis=0) for i in range(len(input_test))])

    elif strategy == 'both_eyes':
        fold_train = np.array([input_train[i][:, :6] for i in range(len(input_train))])
        fold_test = np.array([input_test[j][:, :6] for j in range(len(input_test))])

    elif strategy == 'all':
        both_train, both_test = subset_data(input_train, input_test, 'both_eyes')
        avg_train, avg_test = subset_data(input_train, input_test, 'average')


    return fold_train, fold_test


def cross_validate(pid_all_data, seed, nfolds):
    strategy = 'all'
    valid_pids = list(pid_all_data.keys())
    random.Random(seed).shuffle(valid_pids)

    test_splits = np.array_split(valid_pids, nfolds)
    train_splits = [np.array(valid_pids)[~np.in1d(valid_pids, i)] for i in test_splits]


    for idx, fold in enumerate(range(nfolds)):
        input_train = np.array([pid_all_data[pid]['input'] for pid in train_splits[fold]])
        output_train = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in train_splits[fold]])
        labels_train = np.array([pid for pid in train_splits[fold]])

        input_test = np.array([pid_all_data[pid]['input'] for pid in test_splits[fold]])
        output_test = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in train_splits[fold]])
        labels_train = np.array([pid for pid in test_splits[fold]])

        data_dim_dict = {'left': 3, 'right': 3, 'both_eyes': 6, 'average': 3, 'all': 9}
        net = neural_network(None, data_dim_dict[strategy])


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



    # making splits
    nfolds = 10
    seed = 0
    fold = 0
    valid_pids = list(valid_pids)
    random.Random(seed).shuffle(valid_pids)

    test_splits = np.array_split(valid_pids, nfolds)
    train_splits = [np.array(valid_pids)[~np.in1d(valid_pids, i)] for i in test_splits]

    input_train = np.array([pid_all_data[pid]['input'] for pid in train_splits[fold]])
    output_train = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in train_splits[fold]])

    input_test = np.array([pid_all_data[pid]['input'] for pid in test_splits[fold]])
    output_test = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in train_splits[fold]])

    # pid = 'EA-123'
    # input_data = pid_all_data[pid]['input']
    # diagnosis = np.array([[1, 0] if pid.startswith('E') else [0, 1]][0])
    # y = np.reshape(diagnosis, (1, diagnosis.shape[0]))
    #
    # x = input_data
    # x2 = np.reshape(x, (1, x.shape[0], x.shape[1]))

    # net = neural_network(580, 8)
    # history = net.fit(x2, y, epochs=10)

    net = neural_network(None, 8)
    for idx in range(len(input_train)):
        x = input_train[idx]
        y = output_train[idx]
        # print(idx, len(x))

        x = np.reshape(x, (1, x.shape[0], x.shape[1]))
        y = np.reshape(y, (1, y.shape[0]))

        net.fit(x, y, epochs=20)

    correct = 0
    wrong = 0
    for idx2 in range(len(input_test)):
        x = input_train[idx2]
        y = output_train[idx2]
        # print(idx, len(x))

        x = np.reshape(x, (1, x.shape[0], x.shape[1]))
        y = np.reshape(y, (1, y.shape[0]))

        pred = net.predict(x)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        if all(pred == y):
            correct+=1
        else:
            wrong+=1




    x_test = input_test[0]
    x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))
    net.predict(x_test)


