import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# import moviepy.editor
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

dataset = 'Baseline'
result_path = os.path.join('results')
data_saving_path = None
diagnosis_file_path = None
baseline_processed = None
eye_data_path = None
n_jobs = None
tts_path = None

if os.name == 'nt':
    baseline_processed = r"C:/Users/Anuj/Desktop/Canary/Baseline/OpenFace-eye-gaze"
    eye_data_path = r'C:/Users/Anuj/Desktop/Canary/Baseline/eye_movement'
    diagnosis_file_path = r'C:/Users/Anuj/Desktop/Canary/canary-nlp/datasets/csv_tables/participant_log.csv'
    # data_saving_path = r"C:/Users/Anuj/Desktop/Canary/Baseline/extracted_data/mm"
    data_saving_path = r"C:/Users/Anuj/Desktop/Canary/Baseline/extracted_data/pixel"
    tts_path = r"C:/Users/Anuj/Desktop/Canary/Baseline/TasksTimestamps.csv"
    n_jobs = 6

elif os.name == 'posix':
    processed_files_path = '/home/anuj/OpenFace2/OpenFace/build/processed/'
    baseline_processed = os.path.join(processed_files_path, 'Baseline', '')
    eye_data_path = '/home/anuj/Documents/CANARY_Baseline/eye_movement/'
    diagnosis_file_path = '/home/anuj/multimodal-ml-framework/datasets/canary/participant_log.csv'
    # data_saving_path = '/home/anuj/Documents/CANARY_Baseline/extracted_data/mm'
    data_saving_path = '/home/anuj/Documents/CANARY_Baseline/extracted_data/pixel'
    tts_path = '/home/anuj/Documents/CANARY_Baseline/TasksTimestamps.csv'
    n_jobs = -1

if not os.path.exists(data_saving_path):
    os.mkdir(data_saving_path)

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


# taskstimestamps.csv
ttf = pd.read_csv(tts_path)
ttf_cols = list(ttf.columns)
og_cols = ttf_cols[2:4]
bip_cols = ttf_cols[6:8]


for pid in tqdm(valid_pids):
    if os.name == 'posix':
        pid_folder_path = os.path.join(baseline_processed, pid)
        csv_filename = [i for i in os.listdir(pid_folder_path) if i.endswith('.csv')][0]
        csv_file = pd.read_csv(os.path.join(pid_folder_path, csv_filename))

    elif os.name == 'nt':
        csv_filename = [i for i in os.listdir(baseline_processed) if pid in i][0]
        csv_file = pd.read_csv(os.path.join(baseline_processed, csv_filename))

    # extracting gaze relevant columns from the csv_files
    columns = list(csv_file.columns)[:13]

    '''
    this returns columns
    ['frame',
     'face_id',
     'timestamp',
     'confidence',
     'success',
     'gaze_0_x',
     'gaze_0_y',
     'gaze_0_z',
     'gaze_1_x',
     'gaze_1_y',
     'gaze_1_z',
     'gaze_angle_x',
     'gaze_angle_y']
    '''

    extracted_data = csv_file[columns]
    # input_data.append(extracted_data)

    # getting the original Tobii eye-data using the PID
    eye_data_pid_filename = 'Gaze_' + pid + '.tsv'
    eye_data_pid_file = os.path.join(eye_data_path, eye_data_pid_filename)
    if not os.path.exists(eye_data_pid_file):
        print(pid, ' not found in original eye-data')
        continue

    eye_data_pid = pd.read_csv(eye_data_pid_file, delimiter='\t')
    eye_data_columns_of_interest = [
        'ParticipantName',
        'RecordingDuration',
        'RecordingTimestamp',
        'LocalTimeStamp',
        'EyeTrackerTimestamp',

        'KeyPressEventIndex',
        'KeyPressEvent',
        'MouseEventIndex',
        'MouseEvent',
        'StudioEventIndex',
        'StudioEvent',

        'FixationIndex',
        'SaccadeIndex',
        'GazeEventType',
        'GazeEventDuration',
        'GazePointIndex',

        'GazePointLeftX (ADCSpx)',
        'GazePointLeftY (ADCSpx)',
        'GazePointRightX (ADCSpx)',
        'GazePointRightY (ADCSpx)',
        'GazePointX (ADCSpx)',
        'GazePointY (ADCSpx)',

        'GazePointX (MCSpx)',
        'GazePointY (MCSpx)',

        'GazePointLeftX (ADCSmm)',
        'GazePointLeftY (ADCSmm)',
        'GazePointRightX (ADCSmm)',
        'GazePointRightY (ADCSmm)',

        'StrictAverageGazePointX (ADCSmm)',
        'StrictAverageGazePointY (ADCSmm)',

        'EyePosLeftX (ADCSmm)',
        'EyePosLeftY (ADCSmm)',
        'EyePosLeftZ (ADCSmm)',
        'EyePosRightX (ADCSmm)',
        'EyePosRightY (ADCSmm)',
        'EyePosRightZ (ADCSmm)',

        'DistanceLeft',
        'DistanceRight'
    ]  # experimenting with these columns, more columns will be added if needed
    eye_data_selected = eye_data_pid[eye_data_columns_of_interest]

    # to get the index of studio event start and end (usually the studio event is Screen Recording start and end)
    # 'EyeTrackerTimestamp' field is empty at start_index and end_index. Take start+1 and end-1
    '''
    right now using any studioevent as the start and end time, need to check if there's only 2 studio events, and if 
    the time between them corresponds to the screen recording or does it correspond to the webcam video.
    '''

    start_index, end_index = np.where(eye_data_selected['StudioEvent'].isna() != True)[0]
    sampling_rate = 120
    gaze_len_seconds = ((end_index - start_index) / sampling_rate)

    '''
    right now sampling rate is set to a constant value, although it can be verified by collecting the difference in 
    time between the recorded frames, taking a mean of those and dividing 1000ms by that mean. It would be very close
    to 120Hz, so keeping this value here.
    '''

    # getting the duration of the video to verify if the time between studio events is the same as video length
    # video_pid_file = os.path.join(videos_path, 'VIDEO_' + pid + '.asf')
    # if not os.path.exists(video_pid_file):
    #     video_pid_file = os.path.join(videos_path, 'Video_' + pid + '.asf')
    #
    # video = moviepy.editor.VideoFileClip(video_pid_file)
    # video_duration_seconds = video.duration

    # check if there's any PID which has a high variance in eye-gaze data length and video length
    # if so, skip those PIDs
    # if abs(video_duration_seconds - gaze_len_seconds) > 5:
    #     print(pid, ' has very wildly varying gaze and video lengths, skipping...')
    #     continue

    '''
    OpenFace's output is for every 1/10 of a second, while Tobii data is 1/120 of a second.
    To match them up, every 12th data point in Tobii will correspond to 1/10 of a second.
    Note: there's no gaze data at start_index and end_index, so need to use start+1 and end-1
    there's two ways to get the Tobii eye-tracking data matching in time with the OpenFace output:
    1. take only every 12th data point in Tobii data, starting from the start_index+1 to end_index-1
    2. take the average of gaze coordinates for every 12 data points in Tobii data, starting from start_index+1
    '''

    # check if there's any PID which has a high variance in eye-gaze data length and OpenFace's 3D gaze vector length
    #
    # if abs((end_index-start_index+1)/12 - extracted_data.shape[0]) > 5:
    #      print(pid, 'has a high difference in Tobii data points and OpenFace's output data points')

    '''
    For method 1: 

    Note: every PID has some sort of difference between the number of data points from Tobii gaze data, and the number
    of data points from OpenFace's output data. In that case, one option would be to use the lesser amount of data
    (either there's less data points in Tobii, or there's less data points in OpenFace output). 

    Note2: There are gaze points available before and after the "screenRecording" markers in Tobii gaze data. I'm 
    choosing to ignore those points outside, since they're not part of the video (that OpenFace uses to make output).
    '''

    # method 1: data points from OpenFace and Tobii closest to each other in timestamp (from beginning)
    '''
    Note: I found that using every 12th data point is not the best idea, since it's not certain that every 12th data
    point corresponds exactly with OpenFace's data points. What I ended up doing is calculating the index of every data
    point where the difference in timestamps is the least. 
    Basically: the difference of timestamp between every data point and the start index from Tobii should be the same as
    the difference in timestamp between every data point and start index of OpenFace's output. 
    Example: for a certain data point in Tobii's data, the difference from that data point to the start index is 
    1.997 seconds. Now the data point in OpenFace that has the timestamp 1.2 seconds from the start will be the closest
    here. What I've explained here is in reverse: I first get the timestamp from OpenFace's output and see which data
    point in Tobii's data is the closest in timestamp.


    Code to do that:
    # ts_diffs = []
    # ts_start = eye_data_selected['RecordingTimestamp'][start_index+1]
    # all_tobii_ts_from_start = np.array(eye_data_selected['RecordingTimestamp']) - ts_start
    # for idx in range(extracted_data.shape[0]):
    #     if extracted_data['confidence'][idx] < 0.85:
    #         continue
    #     openface_ts_now = extracted_data['timestamp'][idx]
    #     tobii_idx = np.argmin(abs(all_tobii_ts_from_start/1000-openface_ts_now))
    #     ts_now = eye_data_selected['RecordingTimestamp'][tobii_idx]
    #     tobii_ts_from_start = (ts_now - ts_start) / 1000
    #     
    #     ts_diff = tobii_ts_from_start - openface_ts_now
    #     ts_diffs.append([idx, tobii_ts_from_start, openface_ts_now, ts_diff, abs(ts_diff)])
    # ts_diffs2 = np.array(ts_diffs, dtype='object')

    This gives a combined sum error of 6.6 seconds across all data points (mean error = 0.00239 seconds). If choosing 
    every 12th second, we would have gotten a total error of 257.10 seconds (mean error = 0.09339 seconds).
    '''
    pid_input = []
    pid_output = []

    openface_input_cols = ['timestamp']
    tobii_ADCSpx_cols = ['RecordingTimestamp']

    openface_input_cols.extend([i for i in extracted_data.columns if i.startswith('gaze')])
    tobii_ADCSpx_cols.extend([i for i in eye_data_selected.columns if i.endswith('ADCSpx)')])
    # tobii_ADCSpx_cols.extend([i for i in eye_data_selected.columns if i.endswith('ADCSmm)')])

    ts_start = eye_data_selected['RecordingTimestamp'][start_index + 1]
    all_tobii_ts_from_start = np.array(eye_data_selected['RecordingTimestamp']) - ts_start
    ts_diffs = []

    for idx in range(extracted_data.shape[0]):
        # skip the idx if the confidence in OpenFace's output is lower than 85%
        # other method to get the best ones would be based on success: if success is 0, skip
        # NOTE: the only places where success is 0 is when the confidence is very very low (like less than 0.1)
        # so confidence being < 0.85 is the only filter that matters
        if extracted_data['confidence'][idx] < 0.85 or extracted_data['success'][idx] != 1:
            continue

        openface_ts_now = extracted_data['timestamp'][idx]
        tobii_index = np.argmin(abs(all_tobii_ts_from_start / 1000 - openface_ts_now))

        ts_now = eye_data_selected['RecordingTimestamp'][tobii_index]
        tobii_ts_from_start = (ts_now - ts_start) / 1000
        ts_diff = tobii_ts_from_start - openface_ts_now
        ts_diffs.append([idx, tobii_ts_from_start, openface_ts_now, ts_diff, abs(ts_diff)])
        # currently using ADCSpx as the gaze coordinates, but will suffer if resolution is different
        # ADCSmm would suffer if the screen size is different.

        openface_input = extracted_data.iloc[idx][openface_input_cols]
        tobii_output = eye_data_selected.iloc[tobii_index][tobii_ADCSpx_cols]
        # tobii_output['RecordingTimestamp'] = openface_input['timestamp']

        pid_input.append(np.array(openface_input))
        pid_output.append(np.array(tobii_output))

    ts_diffs2 = np.array(ts_diffs, dtype='object')
    error_sum = np.sum(ts_diffs2[:, 4])
    error_mean = np.mean(ts_diffs2[:, 4])
    # print(pid, ' total time difference error: ', error_sum, ' and average error: ', error_mean)

    pid_input = np.array(pid_input)
    pid_output = np.array(pid_output)

    # adding 'RecordingTimestamp' from pid_output to pid_input
    pid_input = np.append(pid_input, pid_output[:, 0].reshape(len(pid_output), 1), axis=1)
    openface_input_cols.append('RecordingTimestamp')

    # making the timestamp values the same in input and output. To be used later when getting data from
    # different windows of timings
    pid_output[:, 0] = pid_input[:, 0]
    tobii_ADCSpx_cols[0] = 'timestamp'

    '''
    after getting till here, pid_output may have rows where one or more values would be NaN. Again, two options here:
    1. Remove all rows where there's even a single NaN, and keep the rest. Remove the corresponding rows from input.
    2. Remove some of the columns from tobii_ADCSpx. (sometimes there's no values for left eye, sometimes for right eye)

    Here, I'm choosing to go for option 1, then saving the resulting input and outputs. Also gonna save the original.
    '''

    mask = ~pd.isna(pd.DataFrame(pid_output)).any(axis=1)
    nan_removed_input = pid_input[mask]
    nan_removed_output = pid_output[mask]

    min_ts = np.min(nan_removed_input[:, 0])
    max_ts = np.max(nan_removed_input[:, 0])
    # print(pid, error_sum, error_mean, nan_removed_input.shape[0])

    data = [pid, error_sum, error_mean, nan_removed_input.shape[0], min_ts, max_ts]
    meta_data.append(data)

    pid_saving_path = os.path.join(data_saving_path, pid)
    if not os.path.exists(pid_saving_path):
        os.mkdir(pid_saving_path)

    # if os.path.exists(os.path.join(pid_saving_path, 'masked_input.csv')) and \
    #         os.path.exists(os.path.join(pid_saving_path, 'masked_output.csv')):
    #     continue

    pid_input_df = pd.DataFrame(nan_removed_input, columns=openface_input_cols)
    pid_input_df.to_csv(os.path.join(pid_saving_path, 'masked_input.csv'))
    pid_output_df = pd.DataFrame(nan_removed_output, columns=tobii_ADCSpx_cols)
    pid_output_df.to_csv(os.path.join(pid_saving_path, 'masked_output.csv'))

    # getting data that starts after PupilCalib starts
    pid_timings = ttf[ttf['StudyID'] == pid]
    og_start = pid_timings['timestampIni'].iloc[0]
    start_index_from_pupil = np.argmin(abs(pid_input_df['RecordingTimestamp'] - og_start))

    # if os.path.exists(os.path.join(pid_saving_path, 'from_pupil_input.csv')) and \
    #         os.path.exists(os.path.join(pid_saving_path, 'from_pupil_output.csv')):
    #     continue

    pid_input_from_pupil = pid_input_df.iloc[start_index_from_pupil:, ]
    pid_input_from_pupil.to_csv(os.path.join(pid_saving_path, 'from_pupil_input.csv'))
    pid_output_from_pupil = pid_output_df.iloc[start_index_from_pupil:, ]
    pid_output_from_pupil.to_csv(os.path.join(pid_saving_path, 'from_pupil_output.csv'))




# md = pd.concat(meta_data)
meta_data_pd = pd.DataFrame(np.array(meta_data), columns=meta_data_cols)
meta_data_pd.to_csv(os.path.join(data_saving_path, 'meta_data.csv'))

lens = meta_data_pd['Num masked data points']
lens = lens.astype(int)
# outlier removal on 2 standard deviations
l2 = lens.mean() - 2 * lens.std()
u2 = lens.mean() + 2 * lens.std()

new_lens = lens[(lens < u2) & (lens > l2)]
outliers = lens[(lens > u2) | (lens < l2)]

# import matplotlib.pyplot as plt
# plt.xlabel('num data points')
# plt.ylabel('num participants')
# # h1 = plt.hist(lens, bins=50, range=[0, max(lens)])
# plt.hist(lens[(lens < u2) & (lens > l2)], bins=50, range=[0, max(lens)])  # or plt.hist(lens, bins=50) for older one

md = meta_data_pd
outlier_pids = md.iloc[list(outliers.index)]['PID']
outlier_or_not = np.zeros_like(lens, dtype=bool)

for d in outliers.index:
    outlier_or_not[d] = True

meta_data = np.array(md.iloc[:, 1:])
md2 = np.append(meta_data, outlier_or_not.reshape(len(outlier_or_not), 1), axis=1)
new_cols = list(md.columns)[1:]
new_cols.append('outlier?')

pd.DataFrame(md2, columns=new_cols).to_csv(os.path.join(data_saving_path, 'meta_data_outliers.csv'))

'''
# code to check if the StudioEvent lengths match with the video length (or are close to each other):

indices = []
video_lens = []
se_lens = []
for pid in valid_pids:
    eye_data_pid_filename = 'Gaze_' + pid + '.tsv'
    eye_data_pid_file = os.path.join(eye_data_path, eye_data_pid_filename)
    if not os.path.exists(eye_data_pid_file):
        print(pid, ' not found in original eye-data')
        continue
    eye_data_pid = pd.read_csv(eye_data_pid_file, delimiter='\t')
    eye_data_columns_of_interest = [
        'ParticipantName',
        'RecordingDuration',
        'RecordingTimestamp',
        'LocalTimeStamp',
        'EyeTrackerTimestamp',
        'KeyPressEventIndex',
        'KeyPressEvent',
        'StudioEventIndex',
        'StudioEvent',
        'FixationIndex',
        'SaccadeIndex',
        'GazeEventType',
        'GazeEventDuration',
        'GazePointIndex',
        'GazePointLeftX (ADCSpx)',
        'GazePointLeftY (ADCSpx)',
        'GazePointRightX (ADCSpx)',
        'GazePointRightY (ADCSpx)',
        'GazePointX (ADCSpx)',
        'GazePointY (ADCSpx)',
        'GazePointX (MCSpx)',
        'GazePointY (MCSpx)',
        'GazePointLeftX (ADCSmm)',
        'GazePointLeftY (ADCSmm)',
        'GazePointRightX (ADCSmm)',
        'GazePointRightY (ADCSmm)',
        'StrictAverageGazePointX (ADCSmm)',
        'StrictAverageGazePointY (ADCSmm)',
        'EyePosLeftX (ADCSmm)',
        'EyePosLeftY (ADCSmm)',
        'EyePosLeftZ (ADCSmm)',
        'EyePosRightX (ADCSmm)',
        'EyePosRightY (ADCSmm)',
        'EyePosRightZ (ADCSmm)'
    ]  # experimenting with these columns, more columns will be added if needed
    eye_data_selected = eye_data_pid[eye_data_columns_of_interest]
    start_index, end_index = np.where(eye_data_selected['StudioEvent'].isna() != True)[0]
    se_indices.append([start_index, end_index])
    sampling_rate = 120

    video_pid_file = os.path.join(videos_path, 'VIDEO_' + pid + '.asf')
    if not os.path.exists(video_pid_file):
        video_pid_file = os.path.join(videos_path, 'Video_' + pid + '.asf')
    video = moviepy.editor.VideoFileClip(video_pid_file)
    video_duration_seconds = video.duration

    se_len = ((end_index - start_index) / sampling_rate) / 60
    video_lens.append(video_duration_seconds)
    se_lens.append(se_len)

comparisons = np.array([[i, valid_pids[i], video_lens[i]/60, se_lens[i], 
                        (video_lens[i]/60 - se_lens[i])*60, (video_lens[i]/60 - se_lens[i])*60 < 1] \
                        for i in range(len(video_lens))], dtype='object')

outliers = comparisons[np.where(comparisons[:,4] > 5)]

# this would give PIDs with very wildly varying video and gaze timings.
# in Baseline, the PIDs with this anomaly are: EO-028, HI-045, EA-046, EL-114, ET-171

# investigation for each PID:
# EO-028: eye-gaze file very short, doesn't contain much data. The webcam video and screen recording are long (5 min+)
# HI-045: same as above
# EA-046: same as above
# EL-114: same as above
# ET-171: same as above

okay_pids = np.setdiff1d(valid_pids, outliers[:,1])

'''

'''
# For metadata check, and removal of outliers:
md = pd.read_csv(os.path.join(data_saving_path, 'meta_data.csv'))
lens = md['Num masked data points']

# outlier removal on 2 standard deviations
l2 = lens.mean() - 2 * lens.std()
u2 = lens.mean() + 2 * lens.std()

new_lens = lens[(lens < u2) & (lens > l2)]
outliers = lens[(lens > u2) | (lens < l2)]

plt.xlabel('num data points')
plt.ylabel('num participants')
# h1 = plt.hist(lens, bins=50, range=[0, max(lens)])
plt.hist(lens[(lens < u2) & (lens > l2)], bins=50, range=[0, max(lens)])  # or plt.hist(lens, bins=50) for older one

outlier_pids = md.iloc[list(outliers.index)]['PID']
outlier_or_not = np.zeros_like(lens, dtype=bool)

for d in outliers.index:
    outlier_or_not[d] = True

meta_data = np.array(md.iloc[:, 1:])
md2 = np.append(meta_data, outlier_or_not.reshape(len(outlier_or_not), 1), axis=1)
new_cols = list(md.columns)[1:]
new_cols.append('outlier?')

pd.DataFrame(md2, columns=new_cols).to_csv(os.path.join(data_saving_path, 'meta_data_outliers.csv'))
'''