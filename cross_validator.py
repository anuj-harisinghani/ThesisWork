from ModelHandler import ClassifiersFactory
from average_results import average_results

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import random
from multiprocessing import Pool

from sklearn.model_selection import RepeatedKFold, GroupKFold, cross_val_score, train_test_split, cross_validate
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import accuracy_score, mean_absolute_error

warnings.filterwarnings("ignore")

dataset = 'Baseline'
result_path = os.path.join('results')
data_saving_path = None
n_jobs = None

if os.name == 'nt':
    baseline_processed = 'C:/Users/Anuj/Desktop/Canary/Baseline/OpenFace-eye-gaze'
    eye_data_path = 'C:/Users/Anuj/Desktop/Canary/Baseline/eye_movement'
    diagnosis_file_path = 'C:/Users/Anuj/Desktop/Canary/canary-nlp/datasets/csv_tables/participant_log.csv'
    data_saving_path = 'C:/Users/Anuj/Desktop/Canary/Baseline/extracted_data4/'
    n_jobs = 6

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

# creating splits of PIDs, to get lists of PIDs that are gonna be in train and test sets
random_seed = 0
nfolds = 10
random.Random(random_seed).shuffle(valid_pids)
test_splits = np.array_split(valid_pids, nfolds)
train_splits = [np.setdiff1d(valid_pids, i) for i in test_splits]

pid_all_data = {}

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
    pid_all_data[pid] = np.append(t_x, t_y, axis=1)

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
classifiers = ['SVM', 'KNN', 'DecisionTree', 'AdaBoost', 'LogReg', 'Bagging', 'Dummy', 'LinearReg']
window_iter = 20
# modes = ['left', 'right', 'both_eyes', 'avg_vector', 'avg_angle', 'all']  # don't use avg_angle, it's not
modes = ['left', 'right', 'both_eyes', 'avg_vector', 'all']

output_clfs = [os.path.join(result_path, clf) for clf in classifiers]
for oc in output_clfs:
    if not os.path.exists(oc):
        os.mkdir(oc)


def try_multi(idx, mode, clf):
    data = {'train': {i: None for i in range(1, window_iter)},
            'test': {i: None for i in range(1, window_iter)},
            'all': {i: None for i in range(1, window_iter)}}

    train_data = np.array([pid_all_data[p] for p in train_splits[idx]])
    test_data = np.array([pid_all_data[p] for p in test_splits[idx]])

    for window_size in tqdm(range(1, window_iter), desc=str(idx) + ' data processing'):
        # all
        all_mode_data = {'x': {i: None for i in modes}, 'y': {i: None for i in modes}}

        all_window_data = np.vstack(all_data[:, 0:window_size * 10, :])
        all_full_data_points = [i for i in range(len(all_window_data)) if all_window_data[i].any() != False]
        all_window_data = all_window_data[all_full_data_points]

        # lens.append(len(window_data))
        all_mode_data['x']['all'] = all_window_x_all = all_window_data[:, :nip]  # all x
        all_mode_data['x']['both_eyes'] = all_window_data[:, :6]
        all_mode_data['x']['left'] = all_window_x_left = all_window_x_all[:, :3]  # left x
        all_mode_data['x']['right'] = all_window_x_right = all_window_x_all[:, 3:-2]  # right x
        all_mode_data['x']['avg_angle'] = all_window_x_all[:, -2:]  # avg angle as reported by OpenFace
        all_mode_data['x']['avg_vector'] = (all_window_x_left + all_window_x_right) / 2  # manually averaged

        all_mode_data['y']['all'] = all_window_y_all = all_window_data[:, nip:]  # all y
        all_mode_data['y']['both_eyes'] = all_window_data[:, nip:-2]
        all_mode_data['y']['left'] = all_window_y_left = all_window_y_all[:, :2]  # left y
        all_mode_data['y']['right'] = all_window_y_right = all_window_y_all[:, 2:4]  # right y
        all_mode_data['y']['avg_angle'] = all_window_y_all[:, -2:]  # avg y for angles and vectors
        all_mode_data['y']['avg_vector'] = (all_window_y_left + all_window_y_right) / 2

        data['all'][window_size] = all_mode_data

        # train
        train_mode_data = {'x': {i: None for i in modes}, 'y': {i: None for i in modes}}

        train_window_data = np.vstack(train_data[:, 0:window_size * 10, :])
        train_full_data_points = [i for i in range(len(train_window_data)) if train_window_data[i].any() != False]
        train_window_data = train_window_data[train_full_data_points]

        # lens.append(len(window_data))
        # print('train data points:', len(train_window_data))
        train_mode_data['x']['all'] = train_window_x_all = train_window_data[:, :nip]  # all x
        train_mode_data['x']['both_eyes'] = train_window_data[:, :6]
        train_mode_data['x']['left'] = train_window_x_left = train_window_x_all[:, :3]  # left x
        train_mode_data['x']['right'] = train_window_x_right = train_window_x_all[:, 3:-2]  # right x
        train_mode_data['x']['avg_angle'] = train_window_x_all[:, -2:]  # avg angle as reported by OpenFace
        train_mode_data['x']['avg_vector'] = (train_window_x_left + train_window_x_right) / 2  # manually averaged

        train_mode_data['y']['all'] = train_window_y_all = train_window_data[:, nip:]  # all y
        train_mode_data['y']['both_eyes'] = train_window_data[:, nip:-2]
        train_mode_data['y']['left'] = train_window_y_left = train_window_y_all[:, :2]  # left y
        train_mode_data['y']['right'] = train_window_y_right = train_window_y_all[:, 2:4]  # right y
        train_mode_data['y']['avg_angle'] = train_window_y_all[:, -2:]  # avg y for angles and vectors
        train_mode_data['y']['avg_vector'] = (train_window_y_left + train_window_y_right) / 2

        data['train'][window_size] = train_mode_data

        # test
        test_mode_data = {'x': {i: None for i in modes}, 'y': {i: None for i in modes}}

        test_window_data = np.vstack(test_data[:, 0:window_size * 10, :])
        test_full_data_points = [i for i in range(len(test_window_data)) if test_window_data[i].any() != False]
        test_window_data = test_window_data[test_full_data_points]

        # print('test data points:', len(test_window_data))
        test_mode_data['x']['all'] = test_window_x_all = test_window_data[:, :nip]  # all x
        test_mode_data['x']['both_eyes'] = test_window_data[:, :6]
        test_mode_data['x']['left'] = test_window_x_left = test_window_x_all[:, :3]  # left x
        test_mode_data['x']['right'] = test_window_x_right = test_window_x_all[:, 3:-2]  # right x
        test_mode_data['x']['avg_angle'] = test_window_x_all[:, -2:]  # avg angle as reported by OpenFace
        test_mode_data['x']['avg_vector'] = (test_window_x_left + test_window_x_right) / 2  # manually averaged

        test_mode_data['y']['all'] = test_window_y_all = test_window_data[:, nip:]  # all y
        test_mode_data['y']['both_eyes'] = test_window_data[:, nip:-2]
        test_mode_data['y']['left'] = test_window_y_left = test_window_y_all[:, :2]  # left y
        test_mode_data['y']['right'] = test_window_y_right = test_window_y_all[:, 2:4]  # right y
        test_mode_data['y']['avg_angle'] = test_window_y_all[:, -2:]  # avg y for angles and vectors
        test_mode_data['y']['avg_vector'] = (test_window_y_left + test_window_y_right) / 2

        data['test'][window_size] = test_mode_data

    # making classifications on each mode one by one, on the classifiers that are mentioned, across windows
    # for mode in modes:

    train_mean_errors = []
    windows = []

    # for clf in classifiers:
    output_folder = os.path.join(result_path, clf)
    fold_path = os.path.join(output_folder, str(idx))
    if not os.path.exists(fold_path):
        os.mkdir(fold_path)

    pd.DataFrame(train_splits[idx], columns=['PID']).to_csv(os.path.join(fold_path, 'train_pids.csv'))
    pd.DataFrame(test_splits[idx], columns=['PID']).to_csv(os.path.join(fold_path, 'test_pids.csv'))

    for window_size in tqdm(range(1, window_iter), desc=str(idx) + ' ' + clf + ' ' + mode):
        windows.append(window_size)

        train_window_x = data['train'][window_size]['x'][mode]
        train_window_y = data['train'][window_size]['y'][mode]

        test_window_x = data['test'][window_size]['x'][mode]
        test_window_y = data['test'][window_size]['y'][mode]

        # window_x = data['all'][window_size]['x'][mode]
        # window_y = data['all'][window_size]['y'][mode]

        model = ClassifiersFactory().get_model(clf)
        chain = RegressorChain(base_estimator=model)
        # cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
        # train_errors = np.absolute(cross_val_score(chain, train_window_x, train_window_y,
        #                                            scoring='neg_mean_absolute_error',
        #                                            cv=cv, n_jobs=n_jobs))
        # train_errors = cross_validate(chain, train_window_x, train_window_y,
        #                               scoring='neg_mean_absolute_error',
        #                               cv=cv, n_jobs=-1,
        #                               return_estimator=True, return_train_score=True)

        chain = chain.fit(train_window_x, train_window_y)
        window_preds = chain.predict(test_window_x)
        error = mean_absolute_error(y_true=test_window_y, y_pred=window_preds)
        train_mean_errors.append(np.mean(error))

    plt.clf()
    plt.title('{} {} {}'.format(clf, window_iter, mode))
    plt.xlabel('window size')
    plt.ylabel('mean error')
    plt.plot(windows, train_mean_errors)
    print('saving plot {}_{}_{}.png'.format(clf, window_iter, mode))
    plt.savefig(os.path.join(fold_path, '{}_{}_{}.png'.format(clf, window_iter, mode)))
    plt.close()

    error_filename = os.path.join(fold_path, '{}_{}_{}.csv'.format(clf, window_iter, mode))
    pd.DataFrame(train_mean_errors, columns=['mean absolute error'], index=windows).to_csv(error_filename)

    return 0


for clf in classifiers:
    for m in modes:
        cpu_count = os.cpu_count()
        pool = Pool(processes=cpu_count)
        cv = [pool.apply_async(try_multi, args=(seed, m, clf)) for seed in range(nfolds)]
        op = [p.get() for p in cv]
        average_results(classifiers[0], window_iter, m)
