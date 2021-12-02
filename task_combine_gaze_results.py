import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# import moviepy.editor
import matplotlib.pyplot as plt


dataset = 'Baseline'
result_path = os.path.join('results')
data_saving_path = None
diagnosis_file_path = None
baseline_processed = None
eye_data_path = None
n_jobs = None

task_timestamps_file = None
task_save_path = None
tasks = ['PupilCalib', 'CookieTheft', 'Reading', 'Memory']
task_save_folder_paths = None

if os.name == 'nt':
    baseline_processed = r"C:/Users/Anuj/Desktop/Canary/Baseline/OpenFace-eye-gaze"
    eye_data_path = r'C:/Users/Anuj/Desktop/Canary/Baseline/eye_movement'
    diagnosis_file_path = r'C:/Users/Anuj/Desktop/Canary/canary-nlp/datasets/csv_tables/participant_log.csv'
    data_saving_path = r"C:/Users/Anuj/Desktop/Canary/Baseline/extracted_data4/"
    n_jobs = 6

    task_timestamps_file = r"C:/Users/Anuj/Desktop/Canary/Baseline/TasksTimestamps.csv"
    task_save_path = r"C:/Users/Anuj/Desktop/Canary/Baseline/task_data"
    task_save_folder_paths = {task: os.path.join(task_save_path, task) for task in tasks}

elif os.name == 'posix':
    processed_files_path = '/home/anuj/OpenFace2/OpenFace/build/processed/'
    baseline_processed = os.path.join(processed_files_path, 'Baseline', '')
    eye_data_path = '/home/anuj/Documents/CANARY_Baseline/eye_movement/'
    diagnosis_file_path = '/home/anuj/multimodal-ml-framework/datasets/canary/participant_log.csv'
    data_saving_path = '/home/anuj/Documents/CANARY_Baseline/extracted_data4'
    task_timestamps_file = '/home/anuj/Documents/CANARY_Baseline/TasksTimestamps.csv'
    n_jobs = -1

if not os.path.exists(data_saving_path):
    os.mkdir(data_saving_path)

# tasks
for task in tasks:
    if not os.path.exists(task_save_folder_paths[task]):
        os.mkdir(task_save_folder_paths[task])


# PIDs = os.listdir(baseline_processed)
input_data = []
output_data = []

# getting the PIDs which had valid eye-tracking calibration, based on what we use in multimodal-ml-framework
# this gives us 163 PIDs with valid eye-tracking, and that are processed by OpenFace
# so we'll focus on these ones first
pids_diag = pd.read_csv(diagnosis_file_path)
eye_tracking_calibration_flag = 2
valid_pids = list(pids_diag[pids_diag['Eye-Tracking Calibration?'] >= eye_tracking_calibration_flag]['interview'])

# custom stuff:
incompatible_pids = ['EO-028', 'HI-045', 'EA-046', 'EL-114', 'ET-171']
valid_pids = np.setdiff1d(valid_pids, incompatible_pids)

# metadata
meta_data = []
meta_data_cols = ['PID', 'Total TS error', 'Mean TS error', 'Num masked data points', 'Min TS', 'Max TS']

# tasks metadata
task_time_stats = []
task_time_stats_cols = ['task',
                        'mean start timestamp', 'mean end timestamp',
                        'max start timestamp', 'max end timestamp',
                        'min start timestamp', 'min end timestamp',
                        'std start timestamp', 'std end timestamp']


ttf = pd.read_csv(task_timestamps_file)
ttf_cols = list(ttf.columns)
og_cols = ttf_cols[2:4]
bip_cols = ttf_cols[6:8]


stats = ['mean', 'max', 'min', 'std']
big_stats_og = []
big_stats_bip = []

for task in tasks:
    stats_og = []
    stats_bip = []

    task_filter = ttf[ttf['Task'] == task]
    valid_pid_filter = task_filter[task_filter['StudyID'].isin(valid_pids)]

    # og timestamp values and their stats
    st_og_col = []
    for col in og_cols:
        print(task, col)
        data = valid_pid_filter[col]
        mean = data.mean()
        maxi = data.max()
        mini = data.min()
        std = data.std()

        st_og_col.extend([mean, maxi, mini, std])

    for i in range(len(st_og_col) // 2):
        stats_og.append(st_og_col[i])
        stats_og.append(st_og_col[i + 4])

    task_stats_og = [task]
    task_stats_og.extend(stats_og)
    big_stats_og.append(task_stats_og)

    # og timestamp values and their stats
    st_bip_col = []
    for col in bip_cols:
        print(task, col)
        data = valid_pid_filter[col]
        mean = data.mean()
        maxi = data.max()
        mini = data.min()
        std = data.std()

        st_bip_col.extend([mean, maxi, mini, std])

    for i in range(len(st_bip_col) // 2):
        stats_bip.append(st_bip_col[i])
        stats_bip.append(st_bip_col[i + 4])

    task_stats_bip = [task]
    task_stats_bip.extend(stats_bip)

    big_stats_bip.append(task_stats_bip)

big_stats_og = np.array(big_stats_og)
big_stats_bip = np.array(big_stats_bip)

pd.DataFrame(big_stats_og, columns=task_time_stats_cols).to_csv(os.path.join(task_save_path, 'timestamp_stats_normal.csv'))
pd.DataFrame(big_stats_bip, columns=task_time_stats_cols).to_csv(os.path.join(task_save_path, 'timestamp_stats_bip.csv'))
