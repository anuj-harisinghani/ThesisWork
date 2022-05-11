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

# parameters for this file - warnings and tensorflow variables
warnings.filterwarnings("ignore")
dev = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(dev, enable=True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# some global variables
settings = ParamsHandler.load_parameters('settings')
paths = settings['paths'][os.name]
input_path = paths['input']
data_path = paths['data']
ttf = pd.read_csv(paths['ttf'])


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


def extract_task_data(pids, tasks):
    # in combine_gaze_results.py, I was using PLog eye-tracking calibration flag, it is not necessary to have that
    # condition here. Here I can use all the PIDs in the TaskTimestamps file.
    # pids = list(ttf.StudyID.unique())
    # tasks = ['PupilCalib', 'CookieTheft', 'Reading', 'Memory']

    for pid in tqdm(pids, desc='extracting task data from OpenFace data without conditions'):
        pid_save_path = os.path.join(data_path, 'LSTM', pid)
        if not os.path.exists(pid_save_path):
            os.mkdir(pid_save_path)

        csv_filename = [i for i in os.listdir(input_path) if pid in i][0]
        csv_file = pd.read_csv(os.path.join(input_path, csv_filename))
        columns = list(csv_file.columns)[:13]
        columns.remove('face_id')
        selected_csv_file = csv_file[columns]

        for task in tasks:
            ttf_data = ttf[ttf.StudyID == pid]
            if task not in ttf_data.Task.unique():
                print(pid, task)
                continue

            # get timings of each task from TaskTimestamps.csv file
            task_timings = ttf_data[ttf_data.Task == task]
            start = task_timings.timestampIni_bip.iat[0]
            end = task_timings.timestampEnd_bip.iat[0]

            s = round(start / 1000, 1)
            e = round(end / 1000, 1)

            # within_task_range = np.arange(s, e+0.1, 0.1)
            task_data = selected_csv_file[selected_csv_file.timestamp.between(s, e)]
            task_data.to_csv(os.path.join(pid_save_path, task + '.csv'), index=False)


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


def remove_outliers(data, pids, tasks, save_stats=False):
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
        u_90 = np.percentile(counts, 90)

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



def main():
    lstm_params = ParamsHandler.load_parameters('LSTM_params')

    # experiment variables
    seeds = lstm_params['seeds']
    subsets = lstm_params['subsets']
    mp_flag = lstm_params['multiprocessing']
    n_cores = lstm_params['n_cores'][os.name]
    pad_val = lstm_params['pad_val']
    pad_where = lstm_params['pad_where']

    tasks = ['PupilCalib', 'CookieTheft', 'Reading', 'Memory']
    pids = list(ttf.StudyID.unique())
    pids_to_remove = ['HH-076']  # HH-076 being removed because the task timings are off compared to the video length

    # run this once
    # extract_task_data(pids)

    pids.remove(pids_to_remove[0])

    data = get_data(pids, tasks)
    new_pids = remove_outliers(data, pids, tasks)


    task_meta_data = {task: {'PIDs': None, 'median sequence length': None} for task in tasks}
    for task in tasks:
        task_info = new_pids[new_pids.task == task]
        task_meta_data[task]['PIDs'] = task_pids = list(task_info.PID)
        task_meta_data[task]['median sequence length'] = task_median_length = task_info.len.median()

        task_data = get_data(task_pids, task)[task]




