from ParamsHandler import ParamsHandler
from ModelHandler import ClassifiersFactory
from FixationGenerator import fixation_detection, saccade_detection
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


# getting some parameters
params = ParamsHandler.load_parameters('settings')
mode = params['mode']
paths = params['paths'][os.name]
EMDAT_save_path = paths['EMDAT_save_path']
EMDAT_output_path = paths['EMDAT_output_path']
pred_coord_path = os.path.join(paths['pred_coords'], mode)
et_pipeline_pred_coord_path = paths['et_processing_pipeline_raw_data_path']


def create_segment_file(ttf, valid_pids):
    """
    This function creates tab-separated segment files (.seg) for each PID. These seg files go to EMDAT to be used for
    creating the required features.
    :param ttf:
    :param valid_pids:
    :return:
    """

    # params = ParamsHandler.load_parameters('settings')
    # mode = params['mode']
    # paths = params['paths'][os.name]
    # pred_coord_path = os.path.join(paths['pred_coords'], mode)

    for pid in tqdm(valid_pids, desc='creating segment files'):
        pid_path = os.path.join(pred_coord_path, pid)

        ttf_data = ttf[ttf['StudyID'] == pid]
        task_sc = ttf_data[['Task', 'Task', 'timestampIni', 'timestampEnd']]
        # all_sc = pd.DataFrame(
        #     [[pid + '_allsc', 'all_sc', ttf_data['timestampIni'].iloc[0], ttf_data['timestampEnd'].iloc[-1]]],
        #     columns=task_sc.columns)
        all_sc = pd.DataFrame(columns=task_sc.columns)

        seg = all_sc.append(task_sc, ignore_index=True)
        seg.to_csv(os.path.join(pid_path, pid + '.seg'), sep='\t', header=False, index=False)

        # EMDAT_save_path = r"C:\Users\Anuj\PycharmProjects\EMDAT-et-features-generation\src\data\TobiiV3"  # TobiiV2 for old
        # EMDAT_save_path = '/home/anuj/et-processing-pipeline/data/Preprocessing/Segments'
        seg.to_csv(os.path.join(EMDAT_save_path, pid + '.seg'), sep='\t', header=False, index=False)


def run_inference(pid_all_data, models, strat, seeds, nfolds):
    """
    Function to run inference on the data, using the 10x10 trained models and the chosen strat. This will produce gaze
    predictions for each pid in the list with the data.
    :param pid_all_data:
    :param models:
    :param strat:
    :param seeds:
    :param nfolds:
    :return:
    """
    pids = pid_all_data.keys()

    # params = ParamsHandler.load_parameters('settings')
    # mode = params['mode']
    # paths = params['paths'][os.name]
    # pred_coord_path = os.path.join(paths['pred_coords'], mode)

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
        avg_pred_cols = ['OpenFace_TS', 'RecordingTimestamp', 'left_x', 'left_y', 'right_x', 'right_y', 'avg_x',
                         'avg_y']

        avg_pred_file = pd.DataFrame(avg_pred_with_timestamps, columns=avg_pred_cols)
        avg_pred_file.to_csv(os.path.join(pid_path, 'avg_predictions.csv'), index=False)

    # copy avg_predictions to EMDAT folder
    # for pid in pids:
    #     pid_path = os.path.join(pred_coord_path, pid)
    #     file = pd.read_csv(os.path.join(pid_path, 'avg_predictions.csv'))
    #     EMDAT_save_path = r"C:\Users\Anuj\PycharmProjects\EMDAT-et-features-generation\src\data\TobiiV2"
    #     file.to_csv(os.path.join(EMDAT_save_path, pid + '-All-Data.tsv'), sep='\t')
    #
    # # copy left eye fixation files to EMDAT folder
    # for pid in pids:
    #     pid_path = os.path.join(pred_coord_path, pid)
    #     file = pd.read_csv(os.path.join(pid_path, 'left_fixation.csv'))
    #     EMDAT_save_path = r"C:\Users\Anuj\PycharmProjects\EMDAT-et-features-generation\src\data\TobiiV2"
    #     file.to_csv(os.path.join(EMDAT_save_path, pid + '-Fixation-Data.tsv'), sep='\t')
    #
    # # copy left eye saccade files to EMDAT folder
    # for pid in pids:
    #     pid_path = os.path.join(pred_coord_path, pid)
    #     file = pd.read_csv(os.path.join(pid_path, 'left_saccade.csv'))
    #     EMDAT_save_path = r"C:\Users\Anuj\PycharmProjects\EMDAT-et-features-generation\src\data\TobiiV2"
    #     file.to_csv(os.path.join(EMDAT_save_path, pid + '-Saccade-Data.tsv'), sep='\t')


def generate_fixations(pids):
    """
    Function to create fixations and saccades for each PID using the predicted gaze coordinates.
    Using the functions defined in FixationGenerator.py
    :param pids:
    :return:
    """
    # params = ParamsHandler.load_parameters('settings')
    # mode = params['mode']
    # paths = params['paths'][os.name]
    # pred_coord_path = os.path.join(paths['pred_coords'], mode)

    for pid in tqdm(pids, desc='generating fixations for pids'):
        pid_path = os.path.join(pred_coord_path, pid)
        avg_pred_filepath = os.path.join(pid_path, 'avg_predictions.csv')
        avg_preds = pd.read_csv(avg_pred_filepath)

        # for time, choose RecordingTimestamp since it is in milliseconds which is what the fixation generation algo needs
        time = avg_preds.RecordingTimestamp

        # leftx = avg_preds.left_x
        # lefty = avg_preds.left_y

        # rightx = avg_preds.right_x
        # righty = avg_preds.right_y
        #
        avgx = avg_preds.avg_x
        avgy = avg_preds.avg_y

        # fixations
        fix_cols = ['starttime', 'endtime', 'duration', 'endx', 'endy']

        # left_sfix, left_efix = fixation_detection(leftx, lefty, time)
        # lefix = pd.DataFrame(left_efix, columns=fix_cols)
        # lefix.to_csv(os.path.join(pid_path, 'left_fixation.csv'))

        # right_sfix, right_efix = fixation_detection(rightx, righty, time)
        # # right_cols = ['starttime_r', 'endtime_r', 'duration_r', 'endx_r', 'endy_r']
        # refix = pd.DataFrame(right_efix, columns=fix_cols)
        # refix.to_csv(os.path.join(pid_path, 'right_fixation.csv'))

        avg_sfix, avg_efix = fixation_detection(avgx, avgy, time)
        avgefix = pd.DataFrame(avg_efix, columns=fix_cols)
        avgefix.to_csv(os.path.join(pid_path, 'avg_fixation.csv'))

        # saccades
        # sacc_cols = ['starttime', 'endtime', 'duration', 'startx', 'starty', 'endx', 'endy']
        #
        # avg_ssac, avg_esac = saccade_detection(avgx, avgy, time)
        # avgsacc = pd.DataFrame(avg_esac, columns=sacc_cols)
        # avgsacc.to_csv(os.path.join(pid_path, 'avg_saccade.csv'))

        # adding details about fixation and saccade into the avg_preds file to make it All-Data.tsv for TobiiV3 in EMDAT
        # new_cols = ['ValidityLeft', 'ValidityRight', 'FixationIndex', 'SaccadeIndex', 'GazeEventType', 'GazeEventDuration',
        #             'FixationPointX', 'FixationPointY']

        new_cols = ['ParticipantName', 'FixationIndex', 'SaccadeIndex', 'GazeEventType', 'GazeEventDuration',
                    'FixationPointX', 'FixationPointY', 'ValidityLeft', 'ValidityRight',
                    'UnclassifiedIndex']

        gaze_events = pd.DataFrame(columns=new_cols, index=avg_preds.index)
        gaze_events.ValidityLeft = gaze_events.ValidityRight = 0
        gaze_events.ParticipantName = pid

        # Fixations ----------------------------------------------------------------------------------------------------
        for row in avgefix.index:
            fix_start, fix_end, fix_dur = avgefix.loc[row].starttime, avgefix.loc[row].endtime, avgefix.loc[row].duration
            endx, endy = avgefix.loc[row].endx, avgefix.loc[row].endy
            fix_range = avg_preds[(avg_preds.RecordingTimestamp >= fix_start) & (avg_preds.RecordingTimestamp <= fix_end)]
            fix_range_index = fix_range.index

            # edit values
            gaze_events.FixationIndex[fix_range_index] = row+1
            gaze_events.GazeEventType[fix_range_index] = 'Fixation'
            gaze_events.GazeEventDuration[fix_range_index] = int(fix_dur)
            gaze_events.FixationPointX[fix_range_index] = endx
            gaze_events.FixationPointY[fix_range_index] = endy

        # Saccades -----------------------------------------------------------------------------------------------------

        sacc_count = 1
        nan_gaze_events = gaze_events[gaze_events.GazeEventType.isna()]
        nan_gaze_events_index = nan_gaze_events.index
        gaze_events.GazeEventType[nan_gaze_events_index] = 'Saccade'

        sacc_count = 1
        for row in nan_gaze_events_index:
            gaze_events.SaccadeIndex[row] = sacc_count
            if row + 1 not in nan_gaze_events_index:
                sacc_count += 1

        for u in range(1, sacc_count):
            found_uncl = gaze_events[gaze_events.SaccadeIndex == u]
            found_uncl_index = found_uncl.index

            found_uncl_timestamps = avg_preds.RecordingTimestamp[found_uncl_index]
            new_uncl_dur = int(found_uncl_timestamps.iloc[-1] - found_uncl_timestamps.iloc[0])

            gaze_events.GazeEventDuration[found_uncl_index] = new_uncl_dur

        """
        old method of adding saccades:
        I was generating saccades from the given algorithm and then checking for overlaps
        however, that's wrong. Above, I'll be taking fixations and for every portion of time between the fixation, I'll
        consider that as a single saccade. That seems to be the way to go, until a better algorithm is found.
        """
        # adding SaccadeIndex to places where saccades happen
        # There are some spots where saccades and fixations overlap
        # Underlying code checks for overlaps and only labels continous rows as saccades
        # adding 'Saccade' to GazeEventType

        # for row in avgsacc.index:
        #     sacc_start, sacc_end = avgsacc.starttime[row], avgsacc.endtime[row]
        #     sacc_range = avg_preds[(avg_preds.RecordingTimestamp >= sacc_start) & (avg_preds.RecordingTimestamp <= sacc_end)]
        #     sacc_range_index = sacc_range.index
        #
        #     # check if the suggested saccade range already has a fixation in it (overlapping time)
        #     if gaze_events.FixationIndex[sacc_range_index].count() > 0:
        #         # print('Saccade range overlapping with some Fixation')
        #         new_sacc_range_index = sacc_range_index[gaze_events.FixationIndex[sacc_range.index].isna()]
        #         gaze_events.GazeEventType[new_sacc_range_index] = 'Saccade'
        #
        #         for i in new_sacc_range_index:
        #             gaze_events.SaccadeIndex[i] = sacc_count
        #             if i+1 not in new_sacc_range_index:
        #                 # print('saccade breaks at ', i)
        #                 sacc_count += 1
        #
        # # after labelling whichever row has a saccade in it, now based on the SaccadeIndex, put in duration
        # # duration will simply be 100* number of rows for that saccade
        # for s in range(1, sacc_count):
        #     found_sacc = gaze_events[gaze_events.SaccadeIndex == s]
        #     found_sacc_index = found_sacc.index
        #
        #     found_sacc_timestamps = avg_preds.RecordingTimestamp[found_sacc.index]
        #     new_sacc_dur = int(found_sacc_timestamps.iloc[-1] - found_sacc_timestamps.iloc[0])
        #
        #     gaze_events.GazeEventDuration[found_sacc_index] = new_sacc_dur

        # Unclassified -------------------------------------------------------------------------------------------------
        # for rows with neither Fixation or Saccade GazeEventType, put Unclassified and its duration

        """
        Adding unclassified points. No need for this since Saccade is not being generated from the algorithm
        """
        # nan_gaze_events = gaze_events[gaze_events.GazeEventType.isna()]
        # nan_gaze_events_index = nan_gaze_events.index
        # gaze_events.GazeEventType[nan_gaze_events_index] = 'Unclassified'
        #
        # unclassified_count = 1
        # for row in nan_gaze_events_index:
        #     gaze_events.UnclassifiedIndex[row] = unclassified_count
        #     if row+1 not in nan_gaze_events_index:
        #         # print('unclassified breaks at ', row)
        #         unclassified_count += 1
        #
        # for u in range(1, unclassified_count):
        #     found_uncl = gaze_events[gaze_events.UnclassifiedIndex == u]
        #     found_uncl_index = found_uncl.index
        #
        #     found_uncl_timestamps = avg_preds.RecordingTimestamp[found_uncl.index]
        #     new_uncl_dur = int(found_uncl_timestamps.iloc[-1] - found_uncl_timestamps.iloc[0])
        #
        #     gaze_events.GazeEventDuration[found_uncl_index] = new_uncl_dur


        gaze_file = pd.concat((avg_preds, gaze_events), axis=1)
        # EMDAT_save_path = r"C:\Users\Anuj\PycharmProjects\EMDAT-et-features-generation\src\data\TobiiV3"

        # saving gaze file as "All-Data" to be used by EMDAT to create Fixation, Saccade, Path, InfoUnit feature sets
        gaze_file.to_csv(os.path.join(EMDAT_save_path, pid + '-All-Data.tsv'), sep='\t')

        # saving gaze file to ET-Preprocessing-Pipeline to be used by ReadingScript.py and ReadingFeatures.R
        # to create Fraser_reading_features using Reading AOIs
        gaze_file.to_csv(os.path.join(et_pipeline_pred_coord_path, 'Gaze_' + pid + '.tsv'), sep='\t')
        gaze_file.to_csv(os.path.join(pid_path, pid+'-All-Data.csv'))


def create_dataset_files():
    """
    Run this function after EMDAT has created its feature file from the fixations and all-data files created above
    Takes EMDAT feature files and puts them to multimodal-ml-framework
    :return:
    """
    EMDAT_output = os.path.join(EMDAT_output_path, 'output_features_fixed_fixations.tsv')

    file = pd.read_csv(EMDAT_output, sep='\t')

    fraser_reading_file = pd.read_csv(os.path.join(EMDAT_output_path, 'Fraser_reading_features.csv'))
    rename = {'PatientName': 'interview'}
    fraser_reading_file = fraser_reading_file.rename(rename, axis=1)
    fraser_reading_file['task'] = 'Reading'

    # canary_path = r"C:\Users\Anuj\PycharmProjects\multimodal-ml-framework\datasets\canary"
    canary_path = os.path.join(paths['ML_Framework_dataset'], 'canary')
    fix_features = pd.read_csv(os.path.join(canary_path, 'eye_fixation.csv')).columns[1:]
    path_features = pd.read_csv(os.path.join(canary_path, 'eye_path.csv')).columns[1:]
    sacc_features = pd.read_csv(os.path.join(canary_path, 'eye_saccade.csv')).columns[1:]
    pupil_features = pd.read_csv(os.path.join(canary_path, 'eye_pupil.csv')).columns[1:]

    aoi_infounit_features = pd.read_csv(os.path.join(canary_path, 'eye_aoi_infounit.csv')).columns[1:]
    aoi_halves_features = pd.read_csv(os.path.join(canary_path, 'eye_aoi_halves.csv')).columns[1:]
    aoi_quadrant_features = pd.read_csv(os.path.join(canary_path, 'eye_aoi_quadrant.csv')).columns[1:]

    eye_fraser_features = pd.read_csv(os.path.join(canary_path, 'eye_fraser_reading.csv')).columns[1:]

    rename_cols = {'Part_id': 'interview', 'Sc_id': 'task'}
    file = file.rename(rename_cols, axis=1)

    # generating webcam feature files and dividing features among these files
    fix_file = file[fix_features]
    path_file = file[path_features]
    sacc_file = file[sacc_features]
    pupil_file = file[pupil_features]

    aoi_infounit_file = file[aoi_infounit_features]
    aoi_halves_file = file[aoi_halves_features]
    aoi_quadrant_file = file[aoi_quadrant_features]
    
    eye_fraser_file = fraser_reading_file[eye_fraser_features]
    

    save_path = os.path.join(paths['ML_Framework_dataset'], 'webcam')
    fix_file.to_csv(os.path.join(save_path, 'eye_fixation.csv'))
    path_file.to_csv(os.path.join(save_path, 'eye_path.csv'))
    sacc_file.to_csv(os.path.join(save_path, 'eye_saccade.csv'))
    pupil_file.to_csv(os.path.join(save_path, 'eye_pupil.csv'))

    aoi_infounit_file.to_csv(os.path.join(save_path, 'eye_aoi_infounit.csv'))
    aoi_infounit_file.to_csv(os.path.join(save_path, 'eye_infounit.csv'))
    aoi_halves_file.to_csv(os.path.join(save_path, 'eye_aoi_halves.csv'))
    aoi_quadrant_file.to_csv(os.path.join(save_path, 'eye_aoi_quadrant.csv'))

    eye_fraser_file.to_csv(os.path.join(save_path, 'eye_fraser_reading.csv'))


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
    # pid_all_data = cv.get_data(valid_pids)

    # generate fixations and save them to EMDAT and predictions folders
    generate_fixations(valid_pids)
    create_segment_file(ttf, valid_pids)

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
    pids = pid_all_data.keys()

    generate_fixations(pids)
