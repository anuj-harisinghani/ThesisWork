import keras.backend
import tensorflow as tf
from keras.layers import Dense, Input, LSTM, Dropout, Masking
from keras import Sequential
from sklearn.metrics import accuracy_score, mean_absolute_error, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix


from ParamsHandler import ParamsHandler
from ModelHandler import ClassifiersFactory
from cv import get_data, train_test_split

import os
import random
import gc
import numpy as np
import pandas as pd
import warnings
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
dev = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(dev, enable=True)


def neural_network(timesteps, data_dim):
    n_output = 2
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(timesteps, data_dim)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_output, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def neural_network2(timesteps, data_dim):
    n_output = 1
    model = Sequential()
    model.add(LSTM(128, input_shape=(None, data_dim), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_output, activation='sigmoid'))
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

        fold_train = np.array([np.hstack((both_train[i], avg_train[i])) for i in range(len(both_train))])
        fold_test = np.array([np.hstack((both_test[i], avg_test[i])) for i in range(len(both_test))])

    return fold_train, fold_test


def cross_validate(pid_all_data, strategy, seed, nfolds, n_epochs):
    valid_pids = list(pid_all_data.keys())
    random.Random(seed).shuffle(valid_pids)

    # making splits on PIDs based on number of folds - 10 folds gives 90:10 split to train:test
    test_splits = np.array_split(valid_pids, nfolds)
    train_splits = [np.array(valid_pids)[~np.in1d(valid_pids, i)] for i in test_splits]

    # going through all folds to create fold-specific train-test sets
    for fold in tqdm(range(nfolds), desc='seed: {} training'.format(seed)):
        # making train:test x, y, labels
        fold = 0
        input_train = np.array([pid_all_data[pid]['input'] for pid in train_splits[fold]])
        output_train = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in train_splits[fold]])
        labels_train = np.array([pid for pid in train_splits[fold]])

        input_test = np.array([pid_all_data[pid]['input'] for pid in test_splits[fold]])
        output_test = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in test_splits[fold]])
        labels_train = np.array([pid for pid in test_splits[fold]])

        # based on the subset strategy, getting the "input" data
        input_train, input_test = subset_data(input_train, input_test, strategy)

        data_dim_dict = {'left': 3, 'right': 3, 'both_eyes': 6, 'average': 3, 'all': 9}
        net = neural_network(None, data_dim_dict[strategy])
        # net2 = neural_network2(None, data_dim_dict[strategy])


        history = net.fit(input_train, output_train, epochs=100, steps_per_epoch=20)
        train_loss = history.history['loss']
        train_acc = history.history['accuracy']

        test_loss, test_acc = net.evaluate(input_test, output_test, steps=50)


        """
        training without padding - training each sequence one by one - varying sequence length
        """

        # # training
        # training_loss = []
        # training_loss2 = []
        # for train_index in tqdm(range(len(input_train)), desc='training fold {}'.format(fold)):
        #     x_train = input_train[train_index][np.newaxis, :, :]
        #     y_train = output_train[train_index][np.newaxis, :]
        #     y_train2 = y_train[:, 0]
        #
        #     history = net.fit(x_train, y_train, epochs=n_epochs, verbose=0)
        #     hist2 = net2.fit(x_train, y_train2, epochs=n_epochs, verbose=0)
        #
        #     training_loss.append(history.history['loss'])
        #     # training_loss2.append(hist2.history['loss'])
        #     training_loss2.append(hist2.history['loss'])
        #
        # # training loss is in terms of binary crossentropy
        # training_loss = np.array(training_loss)
        # training_loss2 = np.array(training_loss2)
        # # train_loss_per_epoch = np.mean(training_loss, axis=0)
        #
        #
        # # testing
        # preds = []
        # pred_probs = []
        #
        # test_losses = []
        # test_accs = []
        #
        # test_losses2 = []
        # test_accs2 = []
        # for test_index in tqdm(range(len(input_test)), desc='testing fold {}'.format(fold)):
        #     x_test = input_test[test_index][np.newaxis, :, :]
        #     y_test = output_test[test_index][np.newaxis, :]
        #     y_test2 = y_test[:, 0]
        #
        #     # pred_prob - probabilities predicted by softmax directly, pred - converted to binary instead of float probability
        #     pred_prob = net.predict(x_test)
        #     pred = pred_prob.round()
        #
        #     pred_probs.append(pred_prob[0])
        #     preds.append(pred[0])
        #
        #     loss, acc = net.evaluate(x_test, y_test, verbose=0)
        #     test_losses.append(loss)
        #     test_accs.append(acc)
        #
        #     loss2, acc2 = net2.evaluate(x_test, y_test2, verbose=0)
        #     test_losses2.append(loss2)
        #     test_accs2.append(acc2)
        #
        # # test_error = np.array(errors)
        # pred_probs = np.array(pred_probs)
        # preds = np.array(preds)
        # # test_accuracy = accuracy_score(output_test, preds)
        # # test_loss = log_loss(output_test, pred_probs)
        #
        # test_loss = np.mean(test_losses)
        # test_loss2 = np.mean(test_losses2)
        # test_acc = np.mean(test_accs)
        # test_acc2 = np.mean(test_accs2)
        #
        # # other basic metrics
        # roc_auc = roc_auc_score(output_test[:, 0], preds[:, 0])
        # f1 = f1_score(output_test[:, 0], preds[:, 0])
        # acc1 = accuracy_score(output_test[:, 0], preds[:, 0])
        # prec = precision_score(output_test[:, 0], preds[:, 0])
        # rec = recall_score(output_test[:, 0], preds[:, 0])
        # tn, fp, fn, tp = confusion_matrix(output_test[:, 0], preds[:, 0]).ravel()
        # spec = tn/(tn+fp)


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

    # for padding option - remove outliers that have either too many or too little data points, then pad the remaining
    pid_all_data = remove_outliers(pid_all_data)
    pid_all_data = align_and_pad(pid_all_data)

    strategy = 'all'
    n_epochs = 50
    seed = 0
    if mp_flag:
        pool = Pool(processes=n_cores)
        cv = [pool.apply_async(cross_validate, args=(pid_all_data, strategy, seed, nfolds, n_epochs))
              for seed in range(seeds)]
        _ = [p.get() for p in cv]



def remove_outliers(pid_all_data):
    pids = list(pid_all_data.keys())
    lens = np.array([[pid, len(pid_all_data[pid]['input'])] for pid in pids], dtype='object')
    md = pd.DataFrame(lens, columns=['PID', 'len'])

    lens = md.len
    l2 = lens.mean() - lens.std()
    u2 = lens.mean() + lens.std()

    new_lens = lens[(lens < u2) & (lens > l2)]
    outliers = lens[(lens > u2) | (lens < l2)]

    outlier_pids = md.iloc[outliers.index].PID
    kept_pids = md.iloc[new_lens.index].PID

    data = {pid: pid_all_data[pid] for pid in kept_pids}
    return data


def align_and_pad(data):

    # first gotta find the min and max of timestamp across all pids in the data
    min_ts = 0.0
    max_till_now = 0
    for pid in data.keys():
        pid_max_ts = data[pid]['timestamps'][-1, 0]
        if max_till_now < pid_max_ts:
            max_till_now = pid_max_ts
            max_pid = pid

    max_ts = max_till_now

    # now, create a range of all timestamps starting from min_ts to max_ts
    # using only OpenFace's timestamps, not RecordingTimestamp from Tobii
    timestamp_range = np.arange(min_ts, max_ts+0.1, 0.1)
    input_dims = 8
    output_dims = 6

    new_data = {pid: {'input': None, 'output': None} for pid in list(data.keys())}
    for pid in tqdm(data.keys(), 'padding PID data'):
        new_pid_input = np.zeros(shape=(len(timestamp_range), input_dims))
        new_pid_output = np.zeros(shape=(len(timestamp_range), output_dims))

        for ts_index, ts in enumerate(timestamp_range):
            if ts in data[pid]['timestamps'][:, 0]:
                index = np.where(data[pid]['timestamps'][:, 0] == ts)[0][0]
                new_pid_input[ts_index] = data[pid]['input'][index]
                new_pid_output[ts_index] = data[pid]['output'][index]

        new_data[pid]['input'] = new_pid_input
        new_data[pid]['output'] = new_pid_output

    return new_data
