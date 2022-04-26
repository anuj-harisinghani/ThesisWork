from ParamsHandler import ParamsHandler
from ModelHandler import ClassifiersFactory
from FixationGenerator import fixation_detection
from average_results import average_results, plot_all_averaged_models, average_errors

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import random
from multiprocessing import Pool

from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import accuracy_score, mean_absolute_error

warnings.filterwarnings("ignore")


# function to extract data
def get_data(valid_pids):
    params = ParamsHandler.load_parameters('settings')
    mode = params['mode']
    paths = params['paths'][os.name]
    data = os.path.join(paths['data'], mode)

    # get input and output columns from features
    features = ParamsHandler.load_parameters('features')
    op_cols = features['output'][mode]

    pid_all_data = {pid: {'input': None, 'output': None, 'timestamps': None} for pid in valid_pids}

    for pid in valid_pids:
        pid_path = os.path.join(data, pid)
        pid_input = pd.read_csv(os.path.join(pid_path, 'within_tasks_input.csv'))
        pid_output = pd.read_csv(os.path.join(pid_path, 'within_tasks_output.csv'))

        x = np.array(pid_input)
        y = np.array(pid_output)

        pid_all_data[pid]['input'] = x[:, 1:-1]
        pid_all_data[pid]['output'] = y[:, 1:len(op_cols)]

        timestamps = np.array(pid_input[['timestamp', 'RecordingTimestamp']])
        ts_from_start = timestamps - timestamps[0]

        pid_all_data[pid]['timestamps'] = timestamps
        pid_all_data[pid]['ts_from_start'] = ts_from_start

    return pid_all_data


def train_test_split(pid_all_data, strategy, seed, fold, val_set):
    # creating splits of PIDs, to get lists of PIDs that are gonna be in train and test sets
    params = ParamsHandler.load_parameters('settings')
    nfolds = params['folds']
    mode = params['mode']
    results = os.path.join('results', mode)
    valid_pids = list(pid_all_data.keys())

    random.Random(seed).shuffle(valid_pids)

    # adding validation set calculation
    if val_set:
        test_splits_c = np.array_split(valid_pids, nfolds//2)  # currently nfolds = 10, this divides into 5 parts
        train_splits = [np.setdiff1d(valid_pids, i) for i in test_splits_c]  # makes 80:20 splits
        val_splits = [np.array_split(i, 2)[0] for i in test_splits_c]
        test_splits = [np.setdiff1d(test_splits_c[i], val_splits[i]) for i in range(len(test_splits_c))]

    else:
        test_splits = np.array_split(valid_pids, nfolds)
        train_splits = [np.setdiff1d(valid_pids, i) for i in test_splits]
        val_splits = None

    # fold_path = os.path.join(results, str(seed))
    # pd.DataFrame(train_splits).to_csv(os.path.join(fold_path, 'train_pids.csv'))
    # pd.DataFrame(test_splits, columns=['PID']).to_csv(os.path.join(fold_path, 'test_pids.csv'))

    # get the input and output train-test sets
    input_train = np.vstack([pid_all_data[pid]['input'] for pid in train_splits[fold]])
    output_train = np.vstack([pid_all_data[pid]['output'] for pid in train_splits[fold]])

    input_test = np.vstack([pid_all_data[pid]['input'] for pid in test_splits[fold]])
    output_test = np.vstack([pid_all_data[pid]['output'] for pid in test_splits[fold]])

    input_val = np.vstack([pid_all_data[pid]['input'] for pid in val_splits[fold]])
    output_val = np.vstack([pid_all_data[pid]['output'] for pid in val_splits[fold]])

    # ts_train = np.vstack([pid_all_data[pid]['timestamps'] for pid in train_splits[fold]])
    # ts_test = np.vstack([pid_all_data[pid]['timestamps'] for pid in test_splits[fold]])
    #
    # ts_from_start_train = np.vstack([pid_all_data[pid]['ts_from_start'] for pid in train_splits[fold]])
    # ts_from_start_test = np.vstack([pid_all_data[pid]['ts_from_start'] for pid in test_splits[fold]])

    x_train = y_train = x_test = y_test = x_val = y_val = None

    # now take out the pieces that are not required - subsets
    if strategy == 'left':
        x_train = input_train[:, :3]
        y_train = output_train[:, :2]
        x_test = input_test[:, :3]
        y_test = output_test[:, :2]
        x_val = input_val[:, :3]
        y_val = output_val[:, :2]

    elif strategy == 'right':
        x_train = input_train[:, 3:6]
        y_train = output_train[:, 2:4]
        x_test = input_test[:, 3:6]
        y_test = output_test[:, 2:4]
        x_val = input_val[:, 3:6]
        y_val = output_val[:, 2:4]

    elif strategy == 'both_eyes':
        x_train = input_train[:, :6]
        y_train = output_train[:, :4]
        x_test = input_test[:, :6]
        y_test = output_test[:, :4]
        x_val = input_val[:, :6]
        y_val = output_val[:, :4]

    elif strategy == 'avg':
        x_train = (input_train[:, :3] + input_train[:, 3:6]) / 2
        y_train = (output_train[:, :2] + output_train[:, 2:4]) / 2
        x_test = (input_test[:, :3] + input_test[:, 3:6]) / 2
        y_test = (output_test[:, :2] + output_test[:, 2:4]) / 2
        x_val = (input_val[:, :3] + input_val[:, 3:6]) / 2
        y_val = (output_val[:, :2] + output_val[:, 2:4]) / 2

    elif strategy == 'all':
        x_train = np.hstack((input_train[:, :3], input_train[:, 3:6], (input_train[:, :3] + input_train[:, 3:6]) / 2))
        # y_train = np.hstack(
        #     (output_train[:, :2], output_train[:, 2:4], (output_train[:, :2] + output_train[:, 2:4]) / 2))
        y_train = np.hstack((output_train[:, :2], output_train[:, 2:4]))
        x_test = np.hstack((input_test[:, :3], input_test[:, 3:6], (input_test[:, :3] + input_test[:, 3:6]) / 2))
        # y_test = np.hstack((output_test[:, :2], output_test[:, 2:4], (output_test[:, :2] + output_test[:, 2:4]) / 2))
        y_test = np.hstack((output_test[:, :2], output_test[:, 2:4]))
        x_val = np.hstack((input_val[:, :3], input_val[:, 3:6], (input_val[:, :3] + input_val[:, 3:6]) / 2))
        y_val = np.hstack((output_val[:, :2], output_val[:, 2:4]))

    return x_train, y_train, x_test, y_test, x_val, y_val


# try multi processing
def try_multi(pid_all_data, subsets, classifiers, seed):
    params = ParamsHandler.load_parameters('settings')
    nfolds = params['folds']
    mode = params['mode']
    val_set = True

    fold_results = []
    fold_models = {s: [] for s in subsets}
    for strategy in subsets:
        for clf in classifiers:
            for fold in range(nfolds):
                x_train, y_train, x_test, y_test, \
                ts_train, ts_test, ts_from_start_train, ts_from_start_test = \
                    train_test_split(pid_all_data, strategy, seed, fold, val_set)

                if clf != 'NeuralNetwork':
                    model = ClassifiersFactory().get_model(clf)
                    chain = RegressorChain(base_estimator=model)
                    chain = chain.fit(x_train, y_train)

                else:
                    chain = ClassifiersFactory(9, 4).get_model(clf)
                    chain.fit(x_train, y_train)

                preds = chain.predict(x_test)
                error = mean_absolute_error(y_true=y_test, y_pred=preds)

                fold_results.append([error, clf, strategy])
                if clf == 'LinearReg':
                    fold_models[strategy].append(chain)

    result_file = pd.DataFrame(fold_results, columns=['mae', 'clf', 'strategy'])
    result_file.to_csv(os.path.join('results', mode, str(seed), 'errors.csv'))

    avg_folds = result_file.groupby(['clf', 'strategy']).mean().reset_index()
    avg_folds.to_csv(os.path.join('results', mode, str(seed), 'avg_errors.csv'))

    return fold_models


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

    # make output_folders
    if not os.path.exists(results):
        os.mkdir(results)

    # get data
    pid_all_data = get_data(valid_pids)

    output_clfs = [os.path.join(results, str(fold)) for fold in range(nfolds)]
    for oc in output_clfs:
        if not os.path.exists(oc):
            os.mkdir(oc)

    # multiprocessing
    if mp_flag:
        pool = Pool(processes=n_cores)
        cv = [pool.apply_async(try_multi, args=(pid_all_data, subsets, classifiers, seed))
              for seed in range(seeds)]
        _ = [p.get() for p in cv]

    else:
        models = []
        for seed in tqdm(range(seeds)):
            models.append(try_multi(pid_all_data, subsets, classifiers, seed))

    # call average errors
    average_errors(mode)

    # model_save_path = os.path.join('models', mode)
    return models

# main()
