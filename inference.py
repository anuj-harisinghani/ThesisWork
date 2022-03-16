from ParamsHandler import ParamsHandler
from ModelHandler import ClassifiersFactory
from FixationGenerator import fixation_detection
from average_results import average_results, plot_all_averaged_models, average_errors
import cv

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


def run_inference(pid_all_data, models, strat, seeds, nfolds):

    pids = pid_all_data.keys()

    params = ParamsHandler.load_parameters('settings')
    mode = params['mode']
    paths = params['paths'][os.name]
    pred_coord_path = os.path.join(paths['pred_coords'], mode)

    if not os.path.exists(pred_coord_path):
        os.mkdir(pred_coord_path)

    # saving the predicted raw coordinates with their timestamps into a folder for each PID
    for pid in tqdm(pids, desc='saving avg_predictions for pids'):
        pid_path = os.path.join(pred_coord_path, pid)
        if not os.path.exists(pid_path):
            os.mkdir(pid_path)

        # only for 'all' right now
        input_data = pid_all_data[pid]['input']
        input_subset = np.hstack((input_data[:, :3], input_data[:, 3:6], (input_data[:, :3] + input_data[:, 3:6]) / 2))

        # output_data = pid_all_data[pid]['output']
        # y = np.hstack((output_data[:, :2], output_data[:, 2:4], (output_data[:, :2] + output_data[:, 2:4]) / 2))

        timestamps = pid_all_data[pid]['timestamps']

        # make predictions with the given data for the subset strategy
        preds = []
        for seed in range(seeds):
            for fold in range(nfolds):
                preds.append(models[strat][seed][fold].predict(input_subset))

        preds = np.array(preds)
        average_preds = np.mean(preds, axis=0)

        avg_pred_with_timestamps = np.hstack((timestamps, average_preds))
        avg_pred_cols = ['OpenFace_TS', 'RecordingTimestamp', 'left_x', 'left_y', 'right_x', 'right_y', 'avg_x', 'avg_y']

        avg_pred_file = pd.DataFrame(avg_pred_with_timestamps, columns=avg_pred_cols)
        avg_pred_file.to_csv(os.path.join(pid_path, 'avg_predictions.csv'), index=False)


def generate_fixations(pids):
    params = ParamsHandler.load_parameters('settings')
    mode = params['mode']
    paths = params['paths'][os.name]
    pred_coord_path = os.path.join(paths['pred_coords'], mode)

    for pid in tqdm(pids, desc='generating fixations for pids'):
        avg_pred_filepath = os.path.join(pred_coord_path, pid, 'avg_predictions.csv')
        avg_preds = pd.read_csv(avg_pred_filepath)







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
    pid_all_data = cv.get_data(valid_pids)

    # set a subset, classifier
    classifiers = ['LinearReg']
    models = {s: [] for s in subsets}

    # train models and get them
    for seed in tqdm(range(seeds)):
        fold_models = cv.try_multi(pid_all_data, subsets, classifiers, seed)
        [models[s].append(fold_models[s]) for s in subsets]

    for s in subsets:
        models[s] = np.array(models[s])

    strat = 'all'
    run_inference(pid_all_data, models, strat, seeds, nfolds)



