import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Input, LSTM, Dropout, Masking, Bidirectional
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

from ParamsHandler import ParamsHandler
from ModelHandler import ClassifiersFactory
from ResultsHandler import ResultsHandler

import os
import random
import math
import numpy as np
import pandas as pd
import warnings
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt

# parameters for this file - warnings and tensorflow variables
warnings.filterwarnings("ignore")
dev = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(dev, enable=True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.compat.v1.disable_eager_execution()
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# some global variables
settings = ParamsHandler.load_parameters('settings')
paths = settings['paths'][os.name]
input_path = paths['input']
data_path = paths['data']
results_path = os.path.join('results', 'LSTM')
ttf = pd.read_csv(paths['ttf'])


def neural_network(timesteps, data_dim, mask_value=0.):
    n_output = 2
    model = Sequential()
    model.add(Masking(mask_value=mask_value, input_shape=(timesteps, data_dim)))
    model.add(LSTM(10))
    # model.add(Bidirectional(LSTM(10)))
    # model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(n_output, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    return model

#
# def neural_network(timesteps, data_dim, mask_value=0.):
#     n_output = 2
#     model = Sequential()
#     model.add(Masking(mask_value=mask_value, input_shape=(timesteps, data_dim)))
#     model.add(Bidirectional(LSTM(10)))
#     # model.add(LSTM(64))
#     model.add(Dropout(0.5))
#     model.add(Dense(4, activation='relu'))
#     model.add(Dense(n_output, activation='softmax'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     print(model.summary())
#     return model
#

#
# def neural_network2(batch_size, timesteps, data_dim, mask_value=0., stateful=True, verbose=0):
#     n_output = 2
#     model = Sequential()
#     model.add(Input(batch_input_shape=(batch_size, timesteps, data_dim)))
#     model.add(Masking(mask_value=mask_value))
#     model.add(LSTM(10, stateful=stateful, return_sequences=True))
#     model.add(LSTM(5))
#     # model.add(Dropout(0.5))
#     # model.add(Dense(4, activation='relu'))
#     model.add(Dense(n_output, activation='softmax'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     if verbose != 0:
#         print(model.summary())
#     return model
#

#
# def extract_task_data(pids, tasks):
#     # in combine_gaze_results.py, I was using PLog eye-tracking calibration flag, it is not necessary to have that
#     # condition here. Here I can use all the PIDs in the TaskTimestamps file.
#     # pids = list(ttf.StudyID.unique())
#     # tasks = ['PupilCalib', 'CookieTheft', 'Reading', 'Memory']
#
#     for pid in tqdm(pids, desc='extracting task data from OpenFace data without conditions'):
#         pid_save_path = os.path.join(data_path, 'LSTM', pid)
#         if not os.path.exists(pid_save_path):
#             os.mkdir(pid_save_path)
#
#         csv_filename = [i for i in os.listdir(input_path) if pid in i][0]
#         csv_file = pd.read_csv(os.path.join(input_path, csv_filename))
#         columns = list(csv_file.columns)[:13]
#         columns.remove('face_id')
#         selected_csv_file = csv_file[columns]
#
#         for task in tasks:
#             ttf_data = ttf[ttf.StudyID == pid]
#             if task not in ttf_data.Task.unique():
#                 print(pid, task)
#                 continue
#
#             # get timings of each task from TaskTimestamps.csv file
#             task_timings = ttf_data[ttf_data.Task == task]
#             start = task_timings.timestampIni_bip.iat[0]
#             end = task_timings.timestampEnd_bip.iat[0]
#
#             s = round(start / 1000, 1)
#             e = round(end / 1000, 1)
#
#             # within_task_range = np.arange(s, e+0.1, 0.1)
#             task_data = selected_csv_file[selected_csv_file.timestamp.between(s, e)]
#             task_data.to_csv(os.path.join(pid_save_path, task + '.csv'), index=False)
#


def get_data(pids, tasks):
    if type(tasks) == str:
        tasks = [tasks]

    data = {task: {pid: None for pid in pids} for task in tasks}
    for task in tqdm(tasks, 'getting task data'):
        for pid in pids:
            pid_save_path = os.path.join(data_path, 'LSTM', pid)
            file = pd.read_csv(os.path.join(pid_save_path, task + '.csv'))
            x_cols = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
                      'gaze_angle_x', 'gaze_angle_y']
            x = np.array(file[x_cols])
            data[task][pid] = x

    return data


def remove_outliers(data, pids, tasks, percentile_threshold=100, save_stats=False):
    task_stats = {task: {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None, 'total': None,
                         'count': None, '90%ile': None, '95%ile': None}
                  for task in tasks}

    new_task_stats = {task: {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None, 'total': None,
                             'count': None, '90%ile': None, '95%ile': None}
                      for task in tasks}

    new_pids = pd.DataFrame(columns=['PID', 'len', 'task'])
    for task in tasks:
        lens = np.array([[pid, len(data[task][pid])] for pid in pids], dtype='object')
        counts = lens[:, 1]
        md = pd.DataFrame(lens, columns=['PID', 'len'])
        md['task'] = task

        task_stats[task]['count'] = len(counts)
        task_stats[task]['mean'] = np.mean(counts)
        task_stats[task]['std'] = np.std(counts)
        task_stats[task]['median'] = np.median(counts)
        task_stats[task]['min'] = np.min(counts)
        task_stats[task]['max'] = np.max(counts)
        task_stats[task]['total'] = np.sum(counts)
        task_stats[task]['90%ile'] = np.percentile(counts, 90)
        task_stats[task]['95%ile'] = np.percentile(counts, 95)

        # for keeping all data points from PupilCalib, since PupilCalib's max length is not even close to 500
        # but if I'll use the median for each task, then it doesn't matter what sequence length I aim for (eg. 500)
        # if task == 'PupilCalib':
        #     new_task_stats[task] = task_stats[task]
        #     continue

        # for each task, choosing to remove PIDs with sequence lengths higher than 90%ile - always gives 90% number of PIDs
        l_0 = 0
        u_90 = np.percentile(counts, percentile_threshold)

        new_lens = md[(md.len < u_90) & (md.len > l_0)]
        outliers = md[(md.len > u_90) | (md.len < l_0)]
        new_counts = new_lens.len

        new_task_stats[task]['count'] = len(new_counts)
        new_task_stats[task]['mean'] = np.mean(new_counts)
        new_task_stats[task]['std'] = np.std(new_counts)
        new_task_stats[task]['median'] = np.median(new_counts)
        new_task_stats[task]['min'] = np.min(new_counts)
        new_task_stats[task]['max'] = np.max(new_counts)
        new_task_stats[task]['total'] = np.sum(new_counts)
        new_task_stats[task]['90%ile'] = np.percentile(new_counts, 90)
        new_task_stats[task]['95%ile'] = np.percentile(new_counts, 95)

        new_pids = new_pids.append(new_lens)

    if save_stats:
        stats = pd.DataFrame(task_stats).transpose()
        stats.to_csv(os.path.join('stats', 'LSTM', 'task_info', 'more_pids_task_stats.csv'))
        new_stats = pd.DataFrame(new_task_stats).transpose()
        new_stats.to_csv(os.path.join('stats', 'LSTM', 'task_info', 'outliers_removed_more_pids_task_stats.csv'))

    return new_pids


def pad_and_truncate(task_data, final_length, pad_val, pad_where, truncate_where, task_max_length=None):
    pids = list(task_data.keys())
    input_dims = 8

    truncated_data = {}
    if task_max_length:
        final_length = math.ceil(task_max_length / final_length) * final_length

    for pid in tqdm(pids, 'truncating and padding data'):
        pid_data = task_data[pid].transpose()
        truncated_data[pid] = pad_sequences(pid_data, maxlen=int(final_length), dtype=float, value=pad_val,
                                            truncating=truncate_where, padding=pad_where).transpose()

    return truncated_data


def subset_data(x_train, x_test, x_val, strategy):
    fold_train = fold_test = fold_val = None
    if strategy == 'left':
        fold_train = np.array([x_train[i][:, :3] for i in range(len(x_train))])
        fold_test = np.array([x_test[j][:, :3] for j in range(len(x_test))])
        fold_val = np.array([x_val[j][:, :3] for j in range(len(x_val))]) if x_val is not None else None

    elif strategy == 'right':
        fold_train = np.array([x_train[i][:, 3:6] for i in range(len(x_train))])
        fold_test = np.array([x_test[j][:, 3:6] for j in range(len(x_test))])
        fold_val = np.array([x_val[j][:, 3:6] for j in range(len(x_val))]) if x_val is not None else None

    elif strategy == 'average':
        fold_train = np.array([np.mean((x_train[i][:, :3], x_train[i][:, 3:6]), axis=0) for i in range(len(x_train))])
        fold_test = np.array([np.mean((x_test[i][:, :3], x_test[i][:, 3:6]), axis=0) for i in range(len(x_test))])
        fold_val = np.array([np.mean((x_val[i][:, :3], x_val[i][:, 3:6]), axis=0) for i in range(len(x_val))]) if x_val is not None else None

    elif strategy == 'both_eyes':
        fold_train = np.array([x_train[i][:, :6] for i in range(len(x_train))])
        fold_test = np.array([x_test[j][:, :6] for j in range(len(x_test))])
        fold_val = np.array([x_val[j][:, :6] for j in range(len(x_val))]) if x_val is not None else None

    elif strategy == 'all':
        both_train, both_test, both_val = subset_data(x_train, x_test, x_val, 'both_eyes')
        avg_train, avg_test, avg_val = subset_data(x_train, x_test, x_val, 'average')
        fold_train = np.array([np.hstack((both_train[i], avg_train[i])) for i in range(len(both_train))])
        fold_test = np.array([np.hstack((both_test[i], avg_test[i])) for i in range(len(both_test))])
        fold_val = np.array([np.hstack((both_val[i], avg_val[i])) for i in range(len(both_val))]) if x_val is not None else None

    return fold_train, fold_test, fold_val


def cross_validate(data, strategy, seed, stateful=False, task_median_length=None):
    tf.random.set_seed(seed)

    params = ParamsHandler.load_parameters('LSTM_params')
    nfolds = params['folds']
    val_set = params['val_set']
    pad_val = params['pad_val']
    epochs, steps_per_epoch, batch_size = params['n_epochs'], params['steps_per_epoch'], params['batch_size']

    pids = list(data.keys())
    random.Random(seed).shuffle(pids)

    if val_set:
        nfolds = nfolds // 2
        test_splits_c = np.array_split(pids, nfolds)  # currently nfolds = 10, this divides into 5 parts
        train_splits = [np.array(pids)[~np.in1d(pids, i)] for i in test_splits_c]  # makes 80:20 splits
        val_splits = [np.array_split(i, 2)[0] for i in test_splits_c]
        test_splits = test_splits_c[[~np.in1d(test_splits_c[i], val_splits[i]) for i in range(len(test_splits_c))]]

    else:
        test_splits = np.array_split(pids, nfolds)
        train_splits = [np.array(pids)[~np.in1d(pids, i)] for i in test_splits]
        val_splits = None

    metrics = {'roc': [], 'acc': [], 'f1': [], 'prec': [], 'recall': [], 'spec': [],
               'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
               'n_train_hc': [], 'n_train_e': [], 'n_test_hc': [], 'n_test_e': []}


    # going through all folds to create fold-specific train-test sets
    for fold in tqdm(range(nfolds), desc='seed: {} training'.format(seed)):
        # making train:test x, y, labels
        # fold = 0
        validation_data = None
        x_val = y_val = None
        x_train = np.array([data[pid] for pid in train_splits[fold]])
        y_train = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in train_splits[fold]])
        labels_train = np.array([pid for pid in train_splits[fold]])

        x_test = np.array([data[pid] for pid in test_splits[fold]])
        y_test = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in test_splits[fold]])
        labels_test = np.array([pid for pid in test_splits[fold]])

        n_train_hc, n_train_e = np.bincount(y_train[:, 0])
        n_test_hc, n_test_e = np.bincount(y_test[:, 0])

        metrics['n_train_hc'].append(n_train_hc)
        metrics['n_train_e'].append(n_train_e)
        metrics['n_test_hc'].append(n_test_hc)
        metrics['n_test_e'].append(n_test_e)

        if val_set:
            x_val = np.array([data[pid] for pid in val_splits[fold]])
            y_val = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in val_splits[fold]])
            labels_val = np.array([pid for pid in val_splits[fold]])

        x_train, x_test, x_val = subset_data(x_train, x_test, x_val, strategy)
        validation_data = (x_val, y_val) if x_val is not None else None

        data_dim_dict = {'left': 3, 'right': 3, 'both_eyes': 6, 'average': 3, 'all': 9}


        # divide into batches if stateful==True
        # if stateful:
        #     divisor = x_train.shape[1] // task_median_length
        #     x_train = np.array(np.split(x_train, divisor, axis=1))
        #     # x_train = some.reshape((some.shape[1], some.shape[0], some.shape[2], some.shape[3]))
        #     # y_train = y_train[:, np.newaxis, :]
        #     y_train = y_train[np.newaxis, :, :]
        #     # y_train = np.repeat(y_train, divisor, axis=1)
        #     y_train = np.repeat(y_train, divisor, axis=0)
        #
        #     net = neural_network2(batch_size=x_train.shape[1],
        #                           timesteps=x_train.shape[2],
        #                           data_dim=data_dim_dict[strategy],
        #                           mask_value=pad_val,
        #                           stateful=True)
        #
        #     net.reset_states()
        #     history = net.fit(x_train[0], y_train[0], validation_data=validation_data,
        #                           epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1, shuffle=False)

        # neural network
        # else:
        net = neural_network(x_train.shape[1], data_dim_dict[strategy], mask_value=pad_val)

        history = net.fit(x_train, y_train, validation_data=validation_data,
                          epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=0)
        train_loss = history.history['loss']
        train_acc = history.history['accuracy']

        # plt.plot(train_loss)
        # plt.plot(train_acc)

        test_loss, test_acc = net.evaluate(x_test, y_test, verbose=0)
        pred_probs = net.predict(x_test)
        preds = pred_probs.round()

        # calculating metrics
        roc = roc_auc_score(y_test[:, 0], preds[:, 0])
        f1 = f1_score(y_test[:, 0], preds[:, 0])
        acc = accuracy_score(y_test[:, 0], preds[:, 0])
        prec = precision_score(y_test[:, 0], preds[:, 0])
        rec = recall_score(y_test[:, 0], preds[:, 0])
        tn, fp, fn, tp = confusion_matrix(y_test[:, 0], preds[:, 0]).ravel()
        spec = tn / (tn + fp)

        # saving metrics
        metrics['roc'].append(roc)
        metrics['acc'].append(acc)
        metrics['f1'].append(f1)
        metrics['prec'].append(prec)
        metrics['recall'].append(rec)
        metrics['spec'].append(spec)

        metrics['train_acc'].append(train_acc)
        metrics['train_loss'].append(train_loss)
        metrics['test_acc'].append(test_acc)
        metrics['test_loss'].append(test_loss)

    return metrics


def main():
    lstm_params = ParamsHandler.load_parameters('LSTM_params')

    # experiment variables
    seeds = lstm_params['seeds']
    nfolds = lstm_params['folds']
    subsets = lstm_params['subsets']
    mp_flag = lstm_params['multiprocessing']
    n_cores = lstm_params['n_cores'][os.name]
    pad_val = lstm_params['pad_val']
    pad_where = lstm_params['pad_where']
    truncate_where = lstm_params['truncate_where']

    tasks = ['PupilCalib', 'CookieTheft', 'Reading', 'Memory']
    pids = list(ttf.StudyID.unique())
    pids_to_remove = ['HH-076']  # HH-076 being removed because the task timings are off compared to the video length

    # run this once
    # extract_task_data(pids)

    pids.remove(pids_to_remove[0])

    data = get_data(pids, tasks)
    new_pids = remove_outliers(data, pids, tasks, percentile_threshold=100, save_stats=False)

    stateful = False
    task_meta_data = {task: {'PIDs': None, 'median sequence length': None} for task in tasks}

    # tasks = ['Memory']
    for task in tasks:
        # task = 'PupilCalib'
        task_info = new_pids[new_pids.task == task]
        task_meta_data[task]['PIDs'] = task_pids = list(task_info.PID)
        task_meta_data[task]['median sequence length'] = task_median_length = task_info.len.median()
        task_meta_data[task]['max sequence length'] = task_max_length = task_info.len.max()

        task_data = get_data(task_pids, task)[task]
        # task_median_length = 500
        truncated_data = pad_and_truncate(task_data, task_median_length, pad_val, pad_where, truncate_where)

        if stateful:
            task_median_length = 100
            truncated_data = pad_and_truncate(task_data, task_median_length, pad_val, pad_where, truncate_where, task_max_length)

        # data = truncated_data
        # seed = 0
        # folds = 10
        strategy = 'all'
        # metrics = cross_validate(truncated_data, strategy='all', seed=0)

        saved_metrics = []
        for seed in range(seeds):
            metrics = cross_validate(truncated_data, strategy, seed)
            saved_metrics.append(metrics)

        output_foldername = 'first_run_upto_median_ts_LSTM_10_Dense_4_50_50'
        save_results(task, saved_metrics, output_foldername)
        ResultsHandler.compile_results('LSTM', output_foldername)


def save_results(task, saved_metrics, output_foldername, seed=None):
    output_folder = os.path.join(results_path, output_foldername)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    feature_set_names = {'PupilCalib': 'ET_Basic', 'CookieTheft': 'Eye', 'Reading': 'Eye_Reading', 'Memory': 'Audio'}
    metrics = ['acc', 'roc', 'fms', 'precision', 'recall', 'specificity']
    metric_names = {'acc': 'acc', 'roc': 'roc', 'fms': 'f1', 'precision': 'prec', 'recall': 'recall', 'specificity': 'spec'}

    for i, seed_metrics in enumerate(saved_metrics):
        dfs = []
        for metric in metrics:
            metric_name = metric_names[metric]
            metric_data = seed_metrics[metric_name]
            data = pd.DataFrame(metric_data, columns=['1'])
            data['metric'] = metric
            data['model'] = 'LSTM_median'
            data['method'] = 'end_to_end'
            dfs += [data]

        df = pd.concat(dfs, axis=0, ignore_index=True)

        seed_path = os.path.join(output_folder, str(i))
        if not os.path.exists(seed_path):
            os.mkdir(seed_path)

        df.to_csv(os.path.join(seed_path, 'results_new_features_{}.csv'.format(feature_set_names[task])), index=False)
        print('results saved for {}'.format(task))


main()


