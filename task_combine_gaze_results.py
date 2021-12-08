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

# valid_pids from meta_data outliers
meta_data = pd.read_csv(os.path.join(data_saving_path, 'meta_data_outliers.csv'))
meta_mask = meta_data['outlier?'] == False
valid_pids = list(meta_data[meta_mask]['PID'])


# tasks metadata
task_time_stats = []
task_time_stats_cols = ['task',
                        'mean start timestamp', 'mean end timestamp',
                        'max start timestamp', 'max end timestamp',
                        'min start timestamp', 'min end timestamp',
                        'std start timestamp', 'std end timestamp']

task_duration_stats_cols = ['task', 'mean duration seconds', 'max duration seconds', 'min duration seconds', 'std duration seconds']
sampling_rate = 1000  # this is not sampling rate. The values in the TasksTimestamps.csv file are in milliseconds.

ttf = pd.read_csv(task_timestamps_file)
ttf_cols = list(ttf.columns)
og_cols = ttf_cols[2:4]
bip_cols = ttf_cols[6:8]


stats = ['mean', 'max', 'min', 'std']
big_stats_og = []
big_stats_bip = []
big_stats_duration = []

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
        mean = data.mean() / sampling_rate
        maxi = data.max() / sampling_rate
        mini = data.min() / sampling_rate
        std = data.std() / sampling_rate

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
        mean = data.mean() / sampling_rate
        maxi = data.max() / sampling_rate
        mini = data.min() / sampling_rate
        std = data.std() / sampling_rate

        st_bip_col.extend([mean, maxi, mini, std])

    for i in range(len(st_bip_col) // 2):
        stats_bip.append(st_bip_col[i])
        stats_bip.append(st_bip_col[i + 4])

    task_stats_bip = [task]
    task_stats_bip.extend(stats_bip)

    big_stats_bip.append(task_stats_bip)

    # across pids data, to plot graph
    across_pid_og_start = valid_pid_filter['timestampIni']
    across_pid_og_end = valid_pid_filter['timestampEnd']
    across_pid_bip_start = valid_pid_filter['timestampIni_bip']
    across_pid_bip_end = valid_pid_filter['timestampEnd_bip']

    across_pid_duration = across_pid_og_end.subtract(across_pid_og_start, axis=0)
    across_pid_duration_seconds = across_pid_duration/sampling_rate

    # duration stats
    mean = across_pid_duration_seconds.mean()
    maxi = across_pid_duration_seconds.max()
    mini = across_pid_duration_seconds.min()
    std = across_pid_duration_seconds.std()
    big_stats_duration.append([task, mean, maxi, mini, std])

    ylabel = 'num participants'

    # # OG Start
    # label = task + ' starting timestamp'
    #
    # plt.xlabel(label)
    # plt.ylabel(ylabel)
    # plt.hist(across_pid_og_start, bins=50)  # or plt.hist(lens, bins=50) for older one
    #
    # # OG End
    # label = task + ' end timestamp'
    # plt.xlabel(label)
    # plt.ylabel(ylabel)
    # plt.hist(across_pid_og_end, bins=50)  # or plt.hist(lens, bins=50) for older one
    #
    # # BIP Start
    # label = task + ' starting timestamp BIP'
    # plt.xlabel(label)
    # plt.ylabel(ylabel)
    # plt.hist(across_pid_bip_start, bins=50)  # or plt.hist(lens, bins=50) for older one
    #
    # # BIP End
    # label = task + ' end timestamp BIP'
    # plt.xlabel(label)
    # plt.ylabel(ylabel)
    # plt.hist(across_pid_bip_end, bins=50)  # or plt.hist(lens, bins=50) for older one
    #
    # # Duration = same for OG and BIP
    # label = task + ' duration'
    # plt.xlabel(label)
    # plt.ylabel(ylabel)
    # plt.hist(across_pid_duration, bins=50)  # or plt.hist(lens, bins=50) for older one

    # Duration in seconds
    plt.clf()
    label = task + ' duration in seconds'
    plt.xlabel(label)
    plt.ylabel(ylabel)
    plt.hist(across_pid_duration_seconds, bins=50)  # or plt.hist(lens, bins=50) for older one
    plt.savefig(os.path.join(task_save_path, label+'.png'))

big_stats_og = np.array(big_stats_og)
big_stats_bip = np.array(big_stats_bip)
big_stats_duration = np.array(big_stats_duration)

pd.DataFrame(big_stats_og, columns=task_time_stats_cols).to_csv(os.path.join(task_save_path, 'timestamp_stats_normal.csv'))
pd.DataFrame(big_stats_bip, columns=task_time_stats_cols).to_csv(os.path.join(task_save_path, 'timestamp_stats_bip.csv'))
pd.DataFrame(big_stats_duration, columns=task_duration_stats_cols).to_csv(os.path.join(task_save_path, 'duration_stats_seconds.csv'))




