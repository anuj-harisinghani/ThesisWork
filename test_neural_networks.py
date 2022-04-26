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
import gc
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, mean_absolute_error

warnings.filterwarnings("ignore")


def neural_network(n_inputs, n_outputs, n_layers, nodes1, nodes2, nodes3, loss, optimizer):
    model = Sequential()
    if n_layers == 3:
        model.add(Dense(nodes1, input_shape=(n_inputs,), activation='relu'))
        model.add(Dense(nodes2, activation='relu'))
        model.add(Dense(nodes3, activation='linear'))

    elif n_layers == 2:
        model.add(Dense(nodes1, input_shape=(n_inputs,), activation='relu'))
        model.add(Dense(nodes2, activation='linear'))

    elif n_layers == 1:
        model.add(Dense(nodes1, input_shape=(n_inputs,), activation='linear'))

    model.add(Dense(n_outputs))
    model.compile(loss=loss, optimizer=optimizer)

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
    val_set = True
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_split(pid_all_data, strategy, seed, fold, val_set)

    layers = [1, 2, 3]  # , 4, 5]
    n1 = [13]  # np.arange(12, 18)
    n2 = [9]  # np.arange(7, 12)
    n3 = [6]  # np.arange(3, 7)
    losses = ['mae']  # , 'mse']
    optimizers = ['adam']  # , 'sgd']


    errors = []

    n_layers = 1
    nodes2 = 0
    nodes3 = 0
    n1 = np.arange(4, 9)
    for nodes1 in n1:
        for loss in losses:
            for optimizer in optimizers:
                if loss == 'mse' and optimizer == 'sgd':
                    continue
                print(n_layers, nodes1, nodes2, nodes3, loss, optimizer)
                chain = neural_network(9, 4, n_layers, nodes1, nodes2, nodes3, loss, optimizer)
                chain.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
                preds = chain.predict(x_test)
                error = mean_absolute_error(y_test, preds)
                errors.append([error, n_layers, nodes1, nodes2, nodes3, loss, optimizer])
                chain.save(
                    os.path.join('models3', '{}_{}_{}_{}_{}'.format(n_layers, nodes1, nodes2, nodes3, loss, optimizer)))
                with open(os.path.join('models3', 'errors.txt'), 'a') as f:
                    f.write('{}_{}_{}_{}_{}\n'.format(error, n_layers, nodes1, nodes2, nodes3, loss, optimizer))
                del chain
                del preds
                gc.collect()

    n_layers = 2
    nodes3 = 0
    for nodes1 in n1:
        for loss in losses:
            for optimizer in optimizers:
                if loss == 'mse' and optimizer == 'sgd':
                    continue
                for nodes2 in n2:
                    print(n_layers, nodes1, nodes2, nodes3, loss, optimizer)
                    chain = neural_network(9, 4, n_layers, nodes1, nodes2, nodes3, loss, optimizer)
                    chain.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
                    preds = chain.predict(x_test)
                    error = mean_absolute_error(y_test, preds)
                    errors.append([error, n_layers, nodes1, nodes2, nodes3, loss, optimizer])
                    chain.save(
                        os.path.join('models2',
                                     '{}_{}_{}_{}_{}'.format(n_layers, nodes1, nodes2, nodes3, loss, optimizer)))
                    with open(os.path.join('models2', 'errors.txt'), 'a') as f:
                        f.write('{}_{}_{}_{}_{}\n'.format(error, n_layers, nodes1, nodes2, nodes3, loss, optimizer))
                    del chain
                    del preds
                    gc.collect()

    n_layers = 3
    for nodes1 in n1:
        for loss in losses:
            for optimizer in optimizers:
                if loss == 'mse' and optimizer == 'sgd':
                    continue
                for nodes2 in n2:
                    for nodes3 in n3:
                        print(n_layers, nodes1, nodes2, nodes3, loss, optimizer)
                        chain = neural_network(9, 4, n_layers, nodes1, nodes2, nodes3, loss, optimizer)
                        history = chain.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))
                        preds = chain.predict(x_test)
                        error = mean_absolute_error(y_test, preds)
                        errors.append([error, n_layers, nodes1, nodes2, nodes3, loss, optimizer])
                        chain.save(os.path.join('models4', '{}_{}_{}_{}_{}'.format(n_layers, nodes1, nodes2, nodes3, loss, optimizer)))
                        with open(os.path.join('models4', 'errors.txt'), 'a') as f:
                            f.write('{}_{}_{}_{}_{}\n'.format(error, n_layers, nodes1, nodes2, nodes3, loss, optimizer))
                        print(error, n_layers, nodes1, nodes2, nodes3, loss, optimizer)
                        del chain
                        del preds
                        gc.collect()

    # chain = neural_network(9, 4)
    # chain.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
    # preds = chain.predict(x_test)
    # error = mean_absolute_error(y_true=y_test, y_pred=preds)
    #
    # print(n_layers, nerror)


main()
