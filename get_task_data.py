import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# import moviepy.editor
import matplotlib.pyplot as plt
from ParamsHandler import ParamsHandler


params = ParamsHandler.load_parameters('settings')

# experiment variables
dataset = params['dataset']
tasks = params['tasks']
mode = params['mode']
classifiers = params['classifiers']
subsets = params['subsets']  # modes
seeds = params['seeds']
n_folds = params['folds']

# paths
paths = params['paths'][os.name]
input = paths['input']      # baseline_processed
output = paths['output']    # eye_data_path
diag = paths['plog']
data = os.path.join(paths['data'], mode)
results = os.path.join('results', mode)

# multiprocessing
mp_flag = params['multiprocessing']
n_cores = params['n_cores']


# meta-data
meta_data = pd.read_csv(os.path.join(data, 'meta_data.csv'))
valid_pids = list(meta_data['PID'])

# tasks_timestamps file
ttf = pd.read_csv(os.path.join(data, 'TasksTimestamps.csv'))
ttf_cols = list(ttf.columns)
og_cols = ttf_cols[2:4]
bip_cols = ttf_cols[6:8]

for pid in valid_pids:
    # eye_data_pid_filename = 'Gaze_' + pid + '.tsv'
    # eye_data_pid_file = os.path.join(eye_data_path, eye_data_pid_filename)
    # if not os.path.exists(eye_data_pid_file):
    #     print(pid, ' not found in original eye-data')
    #     continue
    # 
    # eye_data_pid = pd.read_csv(eye_data_pid_file, delimiter='\t')
    # 
    # eye_data_columns_of_interest = [
    #     'ParticipantName',
    #     'RecordingDuration',
    #     'RecordingTimestamp',
    #     'LocalTimeStamp',
    #     'EyeTrackerTimestamp',
    # 
    #     'KeyPressEventIndex',
    #     'KeyPressEvent',
    #     'MouseEventIndex',
    #     'MouseEvent',
    #     'StudioEventIndex',
    #     'StudioEvent',
    # 
    #     'FixationIndex',
    #     'SaccadeIndex',
    #     'GazeEventType',
    #     'GazeEventDuration',
    #     'GazePointIndex',
    # ]
    # tsv = eye_data_pid[eye_data_columns_of_interest]
    
    pid_timings = ttf[ttf['StudyID'] == pid]
    og_start = pid_timings['timestampIni'].iloc[0]  # RecordingTimestamp where PupilCalib started
    # bip_start = pid_timings['timeIni_bip'].iloc[0]
    
    input_data = pd.read_csv(os.path.join(data, pid, 'masked_input.csv'))
    start_index = np.argmin(abs(input_data['RecordingTimestamp'] - og_start))  # the starting index in input file where PupilCalib starts
    
    
    