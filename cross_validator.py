from ModelHandler import ClassifiersFactory

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

warnings.filterwarnings("ignore")

dataset = 'Baseline'
graph_path = os.path.join('graphs')
error_path = os.path.join('errors')
data_saving_path = None
n_jobs = None

if os.name == 'nt':
    baseline_processed = 'C:/Users/Anuj/Desktop/Canary/Baseline/OpenFace-eye-gaze'
    eye_data_path = 'C:/Users/Anuj/Desktop/Canary/Baseline/eye_movement'
    diagnosis_file_path = 'C:/Users/Anuj/Desktop/Canary/canary-nlp/datasets/csv_tables/participant_log.csv'
    data_saving_path = 'C:/Users/Anuj/Desktop/Canary/Baseline/extracted_data4/'
    n_jobs = 2

elif os.name == 'posix':
    processed_files_path = '/home/anuj/OpenFace2/OpenFace/build/processed/'
    baseline_processed = os.path.join(processed_files_path, 'Baseline', '')
    eye_data_path = '/home/anuj/Documents/CANARY_Baseline/eye_movement/'
    diagnosis_file_path = '/home/anuj/multimodal-ml-framework/datasets/canary/participant_log.csv'
    data_saving_path = '/home/anuj/Documents/CANARY_Baseline/extracted_data4'
    n_jobs = -1

# get valid pids from meta_data based on outlier or not
meta_data = pd.read_csv(os.path.join(data_saving_path, 'meta_data_outliers.csv'))
meta_mask = meta_data['outlier?'] == False
valid_pids = list(meta_data[meta_mask]['PID'])
max_timestamp = max(meta_data[meta_mask]['Max TS'])
min_timestamp = min(meta_data[meta_mask]['Min TS'])
n_timesteps = int((max_timestamp - min_timestamp) * 10)

# for pid in valid_pids:
#     pid_path = os.path.join(data_saving_path, pid)
#     # Unnamed column crept in when making these files, remove them by using [:, 1:]
#     pid_input = pd.read_csv(os.path.join(pid_path, 'masked_input.csv'))
#     pid_output = pd.read_csv(os.path.join(pid_path, 'masked_output.csv'))
#     ip_cols = list(pid_input.columns)  # 8 columns
#     op_cols = list(pid_output.columns)  # 6 columns
#
#     # for taking all 8 columns in input, and 2 columns in output
#     x = np.array(pid_input)[:, 1:]
#     y = np.array(pid_output)[:, -2:]
#
#     # for taking only gaze angle (x and y) as input, and gaze coordinate (x and y) as output
#     # x = np.array(pid_input)[:, -2:]
#     # y = np.array(pid_output)[:, -2:]
#
#     # model = DecisionTreeRegressor()
#     # model = KNeighborsRegressor()
#     # model = LinearSVR()
#     # model = LogisticRegression()
#     model = RandomForestRegressor()
#     model = MultiOutputRegressor(model)
#     cv = RepeatedKFold(n_splits=100, n_repeats=10, random_state=1)
#     n_scores = np.absolute(cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1))
#     np.mean(n_scores)

ip_cols = [
    'timestamp',
    'gaze_0_x',
    'gaze_0_y',
    'gaze_0_z',
    'gaze_1_x',
    'gaze_1_y',
    'gaze_1_z',
    'gaze_angle_x',
    'gaze_angle_y']  # 9 columns

op_cols = [
    'timestamp',
    'GazePointLeftX (ADCSpx)',
    'GazePointLeftY (ADCSpx)',
    'GazePointRightX (ADCSpx)',
    'GazePointRightY (ADCSpx)',
    'GazePointX (ADCSpx)',
    'GazePointY (ADCSpx)']  # 7 columns

input_data = []
output_data = []
all_data = []

nip = len(ip_cols) - 1
nop = len(op_cols) - 1

for pid in valid_pids:
    pid_path = os.path.join(data_saving_path, pid)
    # Unnamed column crept in when making these files, remove them by using [:, 1:]
    pid_input = pd.read_csv(os.path.join(pid_path, 'masked_input.csv'))
    pid_output = pd.read_csv(os.path.join(pid_path, 'masked_output.csv'))

    # for taking all columns except Unnamed column
    x = np.array(pid_input)[:, 1:]
    y = np.array(pid_output)[:, 1:]

    t_x = np.zeros(shape=(n_timesteps, nip))
    t_y = np.zeros(shape=(n_timesteps, nop))
    for t in tqdm(range(n_timesteps), desc=pid):
        if t / 10 in x[:, 0]:  # t/10 will also be in y[:,0], and most probably in the same location
            ind = np.argwhere(x[:, 0] == t / 10)[0][0]
            t_x[t] = x[ind][1:]
            t_y[t] = y[ind][1:]

    input_data.append(t_x)
    output_data.append(t_y)
    all_data.append(np.append(t_x, t_y, axis=1))

input_data = np.array(input_data)
output_data = np.array(output_data)
all_data = np.array(all_data)

'''
after this above code, the dataset should be in a 3 dimensional array
n_pids * n_timesteps * num_cols

choosing one timestep (one timestep = 0.1s) means taking a 2D array of shape (n_pids * num_cols)
taking more than one timestep would mean taking a 3D array, but they can be reshaped/concatenated
reshape using: 
array.reshape(-1, input_data.shape[2])
this preserves the ip_cols, but stuff everything else together in the first dimension.

----------------------------------------------------------------------------------------------------------------
all_data is a variable that has both input and outputs side by side in a 3D array of shape:
n_pids * n_timesteps * (nip + nop)

will make it easier to stack across timesteps (using np.vstack) and then splitting off the ip_cols and op_cols

----------------------------------------------------------------------------------------------------------------
for calculating num data points in all windows and plotting graph

for window_size in tqdm(range(1, window_iter), desc='running through all windows'):
    windows.append(window_size)
    # window_size = 2  # hyperparameter, defines number of seconds to take in a window starting from 0

    window_data = np.vstack(all_data[:, 0:window_size*10, :])
    full_data_points = [i for i in range(len(window_data)) if window_data[i].any() != False]
    window_data = window_data[full_data_points]

    lens.append(len(window_data))

plt.title('Data points with increasing time windows')
plt.xlabel('time window lengths')
plt.ylabel('number of data points')
plt.plot(windows, lens)
'''

# processing the data to fit in 'data' variable
# each window size has an 'x' 'y', and each 'x' 'y' has left, right, avg, all datasets
classifiers = ['RandomForest']
window_iter = 20
modes = ['left', 'right', 'avg_vector', 'avg_angle', 'all']

data = {i: None for i in range(1, window_iter)}
lens = []
for window_size in tqdm(range(1, window_iter), desc='data processing'):
    mode_data = {'x': {i: None for i in modes}, 'y': {i: None for i in modes}}

    window_data = np.vstack(all_data[:, 0:window_size * 10, :])
    full_data_points = [i for i in range(len(window_data)) if window_data[i].any() != False]
    window_data = window_data[full_data_points]

    lens.append(len(window_data))
    print('data points that are not all zeros:', len(window_data))
    mode_data['x']['all'] = window_x_all = window_data[:, :nip]  # all x
    mode_data['x']['left'] = window_x_left = window_x_all[:, :3]  # left x
    mode_data['x']['right'] = window_x_right = window_x_all[:, 3:-2]  # right x
    mode_data['x']['avg_angle'] = window_x_all[:, -2:]  # avg angle as reported by OpenFace
    mode_data['x']['avg_vector'] = (window_x_left + window_x_right)/2  # manually averaged gaze vectors

    mode_data['y']['all'] = window_y_all = window_data[:, nip:]  # all y
    mode_data['y']['left'] = window_y_all[:, :2]  # left y
    mode_data['y']['right'] = window_y_all[:, 2:4]  # right y
    mode_data['y']['avg_angle'] = mode_data['y']['avg_vector'] = window_y_all[:, -2:]  # avg y for angles and vectors

    data[window_size] = mode_data


# making classifications on each mode one by one, on the classifiers that are mentioned, across windows
for mode in modes:
    mean_errors = []
    windows = []

    for clf in classifiers:
        for window_size in tqdm(range(1, window_iter), desc=mode):
            windows.append(window_size)

            window_data = np.vstack(all_data[:, 0:window_size * 10, :])
            full_data_points = [i for i in range(len(window_data)) if window_data[i].any() != False]
            window_data = window_data[full_data_points]

            window_x = data[window_size]['x'][mode]
            window_y = data[window_size]['y'][mode]

            model = ClassifiersFactory().get_model(clf)
            chain = RegressorChain(base_estimator=model)
            cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
            n_errors = np.absolute(cross_val_score(chain, window_x, window_y, scoring='neg_mean_absolute_error',
                                                   cv=cv, n_jobs=n_jobs))
            mean_errors.append(np.mean(n_errors))

        plt.clf()
        plt.title('{} {} {}'.format(clf, window_iter, mode))
        plt.xlabel('window size')
        plt.ylabel('mean error')
        plt.plot(windows, mean_errors)
        plt.savefig(os.path.join(graph_path, '{}_{}_{}.png'.format(clf, window_iter, mode)))
        plt.close()

        error_filename = os.path.join(error_path, '{}_{}_{}.csv'.format(clf, window_iter, mode))
        pd.DataFrame(mean_errors, columns=['mean absolute error'], index=windows).to_csv(error_filename)
