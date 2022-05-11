import keras
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Input, LSTM, Dropout, Masking
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

from ParamsHandler import ParamsHandler
from ModelHandler import ClassifiersFactory

import os
import random
import numpy as np
import pandas as pd
import warnings
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
dev = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(dev, enable=True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def neural_network(timesteps, data_dim, mask_value=0.):
    n_output = 2
    model = Sequential()
    model.add(Masking(mask_value=mask_value, input_shape=(timesteps, data_dim)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_output, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
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


def get_data(pids):
    params = ParamsHandler.load_parameters('settings')
    mode = params['mode']
    paths = params['paths'][os.name]
    data_path = os.path.join(paths['data'], mode)

    pid_all_data = {pid: {'input': None, 'timestamps': None} for pid in pids}

    for pid in pids:
        pid_path = os.path.join(data_path, pid)
        pid_input = pd.read_csv(os.path.join(pid_path, 'within_tasks_input_unmasked.csv'))

        x = np.array(pid_input)
        pid_all_data[pid]['input'] = x[:, 1:-1]

        timestamps = np.array(pid_input[['timestamp', 'RecordingTimestamp']])
        ts_from_start = timestamps - timestamps[0]

        pid_all_data[pid]['timestamps'] = timestamps
        pid_all_data[pid]['ts_from_start'] = ts_from_start
        pid_all_data[pid]['task'] = pid_input['task']

    return pid_all_data


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


def align_and_pad(data, pad_where='end', pad_value=0.):
    input_dims = 8
    new_data = {pid: {'input': None, 'output': None} for pid in list(data.keys())}

    # for 0 padding at the end
    lens = np.array([[pid, len(vals['input'])] for pid, vals in data.items()], dtype='object')
    max_lens = lens[:, 1].max()

    for pid in tqdm(data.keys(), 'padding PID data with {} at the {}'.format(pad_value, pad_where)):
        new_pid_input = np.full(shape=(max_lens, input_dims), fill_value=pad_value)
        x = data[pid]['input']

        if pad_where == 'end':
            new_pid_input[:x.shape[0], :x.shape[1]] = x
        elif pad_where == 'start':
            new_pid_input[-x.shape[0]:, -x.shape[1]:] = x

        new_data[pid]['input'] = new_pid_input

    # for adding 0 values when aligning according to actual timestamp values
    # else:
    #     # first gotta find the min and max of timestamp across all pids in the data
    #     min_ts = 0.0
    #     max_till_now = 0
    #     for pid in data.keys():
    #         pid_max_ts = data[pid]['timestamps'][-1, 0]
    #         if max_till_now < pid_max_ts:
    #             max_till_now = pid_max_ts
    #             max_pid = pid
    #
    #     max_ts = max_till_now
    #
    #     # now, create a range of all timestamps starting from min_ts to max_ts
    #     # using only OpenFace's timestamps, not RecordingTimestamp from Tobii
    #     timestamp_range = np.arange(min_ts, max_ts + 0.1, 0.1)
    #
    #     for pid in tqdm(data.keys(), 'padding PID data'):
    #         new_pid_input = np.zeros(shape=(len(timestamp_range), input_dims))
    #         # new_pid_output = np.zeros(shape=(len(timestamp_range), output_dims))
    #
    #         for ts_index, ts in enumerate(timestamp_range):
    #             if ts in data[pid]['timestamps'][:, 0]:
    #                 index = np.where(data[pid]['timestamps'][:, 0] == ts)[0][0]
    #                 new_pid_input[ts_index] = data[pid]['input'][index]
    #                 # new_pid_output[ts_index] = data[pid]['output'][index]
    #
    #         new_data[pid]['input'] = new_pid_input
    #         # new_data[pid]['output'] = new_pid_output

    return new_data


def subset_data(input_train, input_test, input_val, strategy):
    fold_train = fold_test = fold_val = None
    if strategy == 'left':
        fold_train = np.array([input_train[i][:, :3] for i in range(len(input_train))])
        fold_test = np.array([input_test[j][:, :3] for j in range(len(input_test))])
        fold_val = np.array([input_val[j][:, :3] for j in range(len(input_val))]) if input_val is not None else None

    elif strategy == 'right':
        fold_train = np.array([input_train[i][:, 3:6] for i in range(len(input_train))])
        fold_test = np.array([input_test[j][:, 3:6] for j in range(len(input_test))])
        fold_val = np.array([input_val[j][:, 3:6] for j in range(len(input_val))]) if input_val is not None else None

    elif strategy == 'average':
        fold_train = np.array([np.mean((input_train[i][:, :3], input_train[i][:, 3:6]), axis=0) for i in range(len(input_train))])
        fold_test = np.array([np.mean((input_test[i][:, :3], input_test[i][:, 3:6]), axis=0) for i in range(len(input_test))])
        fold_val = np.array([np.mean((input_val[i][:, :3], input_val[i][:, 3:6]), axis=0) for i in range(len(input_val))]) if input_val is not None else None

    elif strategy == 'both_eyes':
        fold_train = np.array([input_train[i][:, :6] for i in range(len(input_train))])
        fold_test = np.array([input_test[j][:, :6] for j in range(len(input_test))])
        fold_val = np.array([input_val[j][:, :6] for j in range(len(input_val))]) if input_val is not None else None

    elif strategy == 'all':
        both_train, both_test, both_val = subset_data(input_train, input_test, input_val, 'both_eyes')
        avg_train, avg_test, avg_val = subset_data(input_train, input_test, input_val, 'average')
        fold_train = np.array([np.hstack((both_train[i], avg_train[i])) for i in range(len(both_train))])
        fold_test = np.array([np.hstack((both_test[i], avg_test[i])) for i in range(len(both_test))])
        fold_val = np.array([np.hstack((both_val[i], avg_val[i])) for i in range(len(both_val))]) if input_val is not None else None

    return fold_train, fold_test, fold_val


def cross_validate(data: dict, strategy: str, seed: int, params: dict):
    tf.random.set_seed(seed)
    nfolds = params['folds']
    val_set = params['val_set']
    pad_val = params['pad_val']
    epochs, steps_per_epoch, batch_size = params['n_epochs'], params['steps_per_epoch'], params['batch_size']

    # shuffling pids and making splits on PIDs based on number of folds - 10 folds gives 90:10 split to train:test
    pids = list(data.keys())
    random.Random(seed).shuffle(pids)

    if val_set:
        nfolds = nfolds//2
        test_splits_c = np.array_split(pids, nfolds)  # currently nfolds = 10, this divides into 5 parts
        train_splits = [np.array(pids)[~np.in1d(pids, i)] for i in test_splits_c]  # makes 80:20 splits
        val_splits = [np.array_split(i, 2)[0] for i in test_splits_c]
        test_splits = test_splits_c[[~np.in1d(test_splits_c[i], val_splits[i]) for i in range(len(test_splits_c))]]

    else:
        test_splits = np.array_split(pids, nfolds)
        train_splits = [np.array(pids)[~np.in1d(pids, i)] for i in test_splits]
        val_splits = None


    # test_splits = np.array_split(pids, nfolds)
    # train_splits = [np.array(pids)[~np.in1d(pids, i)] for i in test_splits]

    metrics = {'roc': [], 'acc': [], 'f1': [], 'prec': [], 'recall': [], 'spec': [],
               'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
               'n_train_hc': [], 'n_train_e': [], 'n_test_hc': [], 'n_test_e': []}

    # going through all folds to create fold-specific train-test sets
    for fold in tqdm(range(nfolds), desc='seed: {} training'.format(seed)):
        # making train:test x, y, labels
        # fold = 0
        validation_data = None
        input_val = output_val = None
        input_train = np.array([data[pid]['input'] for pid in train_splits[fold]])
        output_train = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in train_splits[fold]])
        labels_train = np.array([pid for pid in train_splits[fold]])

        input_test = np.array([data[pid]['input'] for pid in test_splits[fold]])
        output_test = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in test_splits[fold]])
        labels_test = np.array([pid for pid in test_splits[fold]])

        n_train_hc, n_train_e = np.bincount(output_train[:, 0])
        n_test_hc, n_test_e = np.bincount(output_test[:, 0])
        metrics['n_train_hc'].append(n_train_hc)
        metrics['n_train_e'].append(n_train_e)
        metrics['n_test_hc'].append(n_test_hc)
        metrics['n_test_e'].append(n_test_e)

        if val_set:
            input_val = np.array([data[pid]['input'] for pid in val_splits[fold]])
            output_val = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in val_splits[fold]])
            labels_val = np.array([pid for pid in val_splits[fold]])

        # based on the subset strategy, getting the "input" data
        input_train, input_test, input_val = subset_data(input_train, input_test, input_val, strategy)
        validation_data = (input_val, output_val) if input_val is not None else None

        data_dim_dict = {'left': 3, 'right': 3, 'both_eyes': 6, 'average': 3, 'all': 9}
        net = neural_network(None, data_dim_dict[strategy], mask_value=pad_val)
        # net2 = neural_network2(None, data_dim_dict[strategy])

        history = net.fit(input_train, output_train, validation_data=validation_data,
                          epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
        train_loss = history.history['loss']
        train_acc = history.history['accuracy']

        # plt.plot(train_loss)
        # plt.plot(train_acc)

        test_loss, test_acc = net.evaluate(input_test, output_test, verbose=0)
        pred_probs = net.predict(input_test)
        preds = pred_probs.round()

        # calculating metrics
        roc = roc_auc_score(output_test[:, 0], preds[:, 0])
        f1 = f1_score(output_test[:, 0], preds[:, 0])
        acc = accuracy_score(output_test[:, 0], preds[:, 0])
        prec = precision_score(output_test[:, 0], preds[:, 0])
        rec = recall_score(output_test[:, 0], preds[:, 0])
        tn, fp, fn, tp = confusion_matrix(output_test[:, 0], preds[:, 0]).ravel()
        spec = tn/(tn+fp)

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
    # save_results(metrics, strategy, net)


def main():
    params = ParamsHandler.load_parameters('settings')
    lstm_params = ParamsHandler.load_parameters('LSTM_params')

    # experiment variables
    seeds = lstm_params['seeds']
    subsets = lstm_params['subsets']
    mp_flag = lstm_params['multiprocessing']
    n_cores = lstm_params['n_cores'][os.name]
    pad_val = lstm_params['pad_val']
    pad_where = lstm_params['pad_where']

    # getting valid_pids from TasksTimestamps.csv and meta_data collected before - leaves us with 157 participants
    paths = params['paths'][os.name]
    mode = params['mode']
    data = os.path.join(paths['data'], mode)
    ttf = pd.read_csv(paths['ttf'])
    ttf_pids = ttf['StudyID'].unique()

    meta_data = pd.read_csv(os.path.join(data, 'meta_data_unmasked.csv'))
    valid_pids = np.intersect1d(ttf_pids, meta_data['PID'])

    # get data
    pid_all_data = get_data(valid_pids)
    tasks_timings_analysis(pid_all_data, ttf)
    # for padding option - remove outliers that have either too many or too little data points, then pad the remaining
    # pid_all_data = remove_outliers(pid_all_data)
    # pid_all_data = align_and_pad(pid_all_data, align_at_end=False)
    data = align_and_pad(pid_all_data, pad_where=pad_where, pad_value=pad_val)

    # run for all strategies one by one
    strategy = 'all'
    # seed = 0
    # fold = 0
    # params = lstm_params

    metrics = cross_validate(data, strategy, 0, lstm_params)
    if mp_flag:
        pool = Pool(processes=n_cores)
        cv = [pool.apply_async(cross_validate, args=(data, strategy, seed, lstm_params))
              for seed in range(seeds)]
        _ = [p.get() for p in cv]





def save_results(metrics: dict, strategy: str, net):
    lstm_params = ParamsHandler.load_parameters('LSTM_params')
    n_epochs = lstm_params['n_epochs']
    steps = lstm_params['steps_per_epoch']
    output_folder = lstm_params['output_folder']

    seeds = lstm_params['seeds']
    folds = lstm_params['folds']

    if output_folder == '':
        folder_names = os.listdir(os.path.join('results', 'LSTM'))
        output_folder = '{}_{}e_{}s'.format(strategy, n_epochs, steps)

        same_named_folders = [i for i in folder_names if output_folder in i]
        if not same_named_folders:
            latest_num = '1'
        else:
            latest_folder = sorted([i for i in folder_names if output_folder in i])[-1]
            latest_num = str(int(latest_folder.replace(output_folder, '')) + 1)

        output_folder += '_' + latest_num

    save_path = os.path.join('results', 'LSTM', output_folder)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # this for just one seed

    for seed in range(seeds):
        fold_path = os.path.join(save_path, str(seed))
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        summary_keys = [i for i in metrics.keys() if not i.startswith('train')]
        summary_metrics = {key: metrics[key] for key in summary_keys}
        summary_metrics = pd.DataFrame(summary_metrics, columns=summary_keys)

        summary_metrics['n_train_diff'] = summary_metrics['n_train_hc'] - summary_metrics['n_train_e']
        summary_metrics['n_test_diff'] = summary_metrics['n_test_hc'] - summary_metrics['n_test_e']

        summary_metrics.to_csv(os.path.join(fold_path, 'summary_metrics.csv'))

        # training loss
        plt.rcParams['figure.autolayout'] = True
        plt.figure(figsize=(16, 12), dpi=120)
        plt.plot(metrics['train_loss'][seed])
        plt.legend(['training loss'])
        plt.xlabel('epochs')
        plt.ylabel('log loss (binary crossentropy)')
        plt.title('training loss over {} epochs {} steps per epoch: version {}'.format(n_epochs, steps, latest_num))
        plt.savefig(os.path.join(fold_path, 'training loss.png'))

        plt.figure(figsize=(16, 12), dpi=120)
        plt.plot(metrics['train_acc'][seed], color='orange')
        plt.legend(['training accuracy'])
        plt.xlabel('epochs')
        plt.ylabel('accuracy %')
        plt.title('training accuracy over {} epochs {} steps per epoch: version {}'.format(n_epochs, steps, latest_num))
        plt.savefig(os.path.join(fold_path, 'training accuracy.png'))


def tasks_timings_analysis(pid_all_data, ttf):
    pids = pid_all_data.keys()
    tasks = ttf['Task'].unique()

    task_pid_all_data = {task: None for task in tasks}

    # getting summary stats for tasks, number of samples, mean, etc.
    task_stats = {task: {'mean': None, 'std': None, 'min': None, 'max': None, 'total': None, 'count': None} for task in tasks}

    task_counts = {task: [] for task in tasks}
    for pid in pids:
        data = pid_all_data[pid]['task']
        vals = data.value_counts()
        for task in vals.index:
            task_counts[task].append(vals[task])

    for task in tasks:
        counts = task_counts[task]
        task_stats[task]['count'] = len(counts)
        task_stats[task]['mean'] = np.mean(counts)
        task_stats[task]['std'] = np.std(counts)
        task_stats[task]['min'] = np.min(counts)
        task_stats[task]['max'] = np.max(counts)
        task_stats[task]['total'] = np.sum(counts)

    stats = pd.DataFrame(task_stats).transpose()
    stats.to_csv(os.path.join('stats', 'LSTM', 'task_stats.csv'))


    for task in tasks:
        for pid in pids:
            ts = pid_all_data[pid]['timestamps'][:, 1]
