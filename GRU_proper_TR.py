import pandas as pd
import numpy as np
import os
import sys
import torch
import math
import random
import shutil
import matplotlib.pyplot as plt
import itertools
# import seaborn as sns
import pickle

from typing import Callable
from torch import nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, \
    precision_score, recall_score, confusion_matrix, roc_curve
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from ParamsHandler import ParamsHandler
from ResultsHandler import ResultsHandler

# Torch Device
if not torch.cuda.is_available():
    print("WARNING: running on CPU since GPU is not available")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(0)

import warnings

warnings.filterwarnings("ignore")

# torch params
torch_params = ParamsHandler.load_parameters('torch_params')
TORCH_PARAMS = torch_params
CLUSTER = torch_params['cluster']
SEEDS = np.arange(torch_params['seeds'])
NFOLDS = torch_params['folds']
TASKS = torch_params['tasks']
DATASET = torch_params['dataset']
STRATEGY = torch_params['strategy']
OUTPUT_FOLDERNAME = torch_params['output_foldername']
DATA_DIM_DICT = torch_params['data_dim_dict'][DATASET]

# Data pre-processing params
OUTLIER_THRESHOLD = torch_params['outlier_threshold']
PAD_WHERE = torch_params['pad_where']
TRUNCATE_WHERE = torch_params['truncate_where']
FINAL_LENGTH = torch_params['final_length']
PAD_VAL = float(torch_params['pad_val'])
BATCH_SIZE = torch_params['batch_size']

# Training Params
VAL_SET = torch_params['val_set']
EPOCHS = torch_params['epochs']
LEARNING_RATE = torch_params['learning_rate']
CHUNK_LEN = torch_params['chunk_len']
CHUNK_PROCESSING = torch_params['multi_chunk']
VERBOSE = torch_params['verbose']

# Neural Network Params
NN_TYPE = torch_params['network_type']
INPUT_SIZE = DATA_DIM_DICT[STRATEGY]
# INPUT_SIZE = torch_params['input_size']
OUTPUT_SIZE = torch_params['output_size']
HIDDEN_SIZE = torch_params['hidden_size']
BIDIRECTIONAL = torch_params['bidirectional']
NUM_LAYERS = torch_params['num_layers']
DROPOUT = torch_params['dropout']
FOLDS_TO_PROCESS = None

# Handle paths for reading data and saving information
model_save_path = os.path.join(os.getcwd(), 'models', 'chunk_' + DATASET, OUTPUT_FOLDERNAME)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path, exist_ok=True)

results_path = os.path.join(os.getcwd(), 'results', 'chunk_' + DATASET)
data_path = os.path.join(os.getcwd(), 'data')

# Handling participant IDs
ttf = pd.read_csv(os.path.join(data_path, 'TasksTimestamps.csv'))
# PIDS = list(ttf.StudyID.unique())
canary_162_pids = pd.read_csv(os.path.join(data_path, 'canary_162_pids.csv'))
PIDS = list(canary_162_pids.interview)

if DATASET == 'webcam':
    pids_to_remove = ['HH-076', 'EO-139', 'HA-197']  # HH-076 being removed because the task timings are off compared to the video length
else:
    pids_to_remove = ['EO-028', 'HI-045', 'EA-046', 'EL-114', 'ET-171']
PIDS = [pid for pid in PIDS if pid not in pids_to_remove]

'''
canary 162 pids are the superset PIDS from the canary fusion framework. of these, 126 are ET+lang and the rest are lang only
canary 126 is ET only (taken from Pupil Calib) from canary

Pids in the 162 set that have issues here with videos - 
1. HH-076 - video length and eye-tracking file timings (which affect the task timings) don't match
2. EO-139 - don't have task timings in Oswald's script, there is raw eye-tracking data but this PID also doesn't exist in the 126 set
3. HA-197 -  same as EO-139
4. HA-108 - only removed in CookieTheft due to the extremely long sequence

CT - 158, R - 159 PIDs after removing the PIDs above
'''


def get_data(pids: list = PIDS, tasks: list or str = TASKS) -> dict:
    """
    function get_data --> reads the CSV files containing relevant data, based on the PIDs specified and the tasks
    uses the constant DATASET to choose which dataset to pull data from. Possible datasets - Webcam, Tobii
    :param pids: list of participants to get the data for
    :param tasks: the tasks to get data for. Could be a single task (if tasks is string) or a list of tasks
    :return: dict "data" -> contains all the input data. First level is task, second level is each PID
    """

    # check if a single task (string) or a list of tasks is passed in
    if type(tasks) == str:
        tasks = [tasks]

    # initialize data
    data = {task: {} for task in tasks}

    if DATASET == 'webcam':
        x_cols = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z',
                  'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
                  'gaze_avg_x', 'gaze_avg_y', 'gaze_avg_z']

        if STRATEGY == 'left':
            x_cols = x_cols[:3]
        elif STRATEGY == 'right':
            x_cols = x_cols[3:6]
        elif STRATEGY == 'avg':
            x_cols = x_cols[6:9]
        elif STRATEGY == 'both_eyes':
            x_cols = x_cols[:6]

        for task in tqdm(tasks, 'getting task data'):
            for pid in pids:
                if task == 'CookieTheft' and pid == 'HA-108':
                    continue
                pid_save_path = os.path.join(data_path, 'webcam_all', pid)
                file = pd.read_csv(os.path.join(pid_save_path, task + '.csv'))
                x = np.array(file[x_cols])
                data[task][pid] = x

    else:  # when DATASET == 'tobii'
        x_cols = ['GazePointLeftX (ADCSpx)', 'GazePointLeftY (ADCSpx)',
                  'GazePointRightX (ADCSpx)', 'GazePointRightY (ADCSpx)',
                  'GazePointX (ADCSpx)', 'GazePointY (ADCSpx)',
                  'PupilLeft', 'PupilRight',
                  'ValidityLeft', 'ValidityRight']

        if STRATEGY == 'left':
            x_cols = x_cols[:2]
        elif STRATEGY == 'right':
            x_cols = x_cols[2:4]
        elif STRATEGY == 'avg':
            x_cols = x_cols[4:6]
        elif STRATEGY == 'both_eyes':
            x_cols = x_cols[:4]
        elif STRATEGY == 'all':
            x_cols = x_cols[:6]

        for task in tqdm(tasks, 'getting tobii task data'):
            for pid in pids:
                pid_save_path = os.path.join(data_path, 'Tobii_all', pid)
                file = pd.read_csv(os.path.join(pid_save_path, task + '.csv'))
                x = np.array(file[x_cols])
                data[task][pid] = x

    return data


def remove_outliers(data, percentile_threshold=OUTLIER_THRESHOLD, save_stats=False) -> pd.DataFrame:
    """
    function remove_outliers --> given the data, the pids and tasks, remove outliers which are above the specified
    percentile threshold
    :param data: data gotten from get_data function
    :param percentile_threshold: the threshold above which you want to remove participants
    :param save_stats: boolean flag for if you want to save statistics before and after outlier removal
    :return: pd.DataFrame containing info about each PID, like their sequence length and which task
    """

    # task_stats saves statistics of sequence length distribution BEFORE removing outliers
    task_stats = {task: {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None, 'total': None,
                         'count': None, '90%ile': None, '95%ile': None}
                  for task in TASKS}

    # new_task_stats saves statistics of sequence length distribution AFTER removing outliers
    new_task_stats = {task: {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None, 'total': None,
                             'count': None, '90%ile': None, '95%ile': None}
                      for task in TASKS}

    task_lengths = {}
    new_task_lengths = {}
    removed_task_percent = {}

    # initializing the DataFrame
    new_pids = pd.DataFrame(columns=['PID', 'len', 'task'])
    for task in TASKS:
        # lens = np.array([[pid, len(data[task][pid])] for pid in PIDS], dtype='object')
        lens = np.array([[pid, len(data[task][pid])] for pid in data[task].keys()],
                        dtype='object')  # removing HA-108 from CT
        counts = lens[:, 1]
        task_lengths[task] = counts
        lens_df = pd.DataFrame(lens, columns=['PID', 'len'])
        lens_df['task'] = task

        # Statistics before removing outliers
        task_stats[task]['count'] = len(counts)
        task_stats[task]['mean'] = np.mean(counts)
        task_stats[task]['std'] = np.std(counts)
        task_stats[task]['median'] = np.median(counts)
        task_stats[task]['min'] = np.min(counts)
        task_stats[task]['max'] = np.max(counts)
        task_stats[task]['total'] = np.sum(counts)
        task_stats[task]['90%ile'] = np.percentile(counts, 90)
        task_stats[task]['95%ile'] = np.percentile(counts, 95)

        # set lower and upper limits
        l_0 = 0
        u_90 = np.percentile(counts, percentile_threshold)

        # split participants based on the limits --> within the limits are acceptable, out of the limits are outliers
        new_lens = lens_df[(lens_df.len <= u_90) & (lens_df.len >= l_0)]

        # only for 90%ile plotting
        # ninety_perc_counts = lens_df[(lens_df.len <= np.percentile(counts, 90)) & (lens_df.len >= l_0)].len

        for ind in lens_df[lens_df.len > u_90].index:
            lens_df['len'][ind] = round(u_90)
        new_counts = lens_df.len
        new_task_lengths[task] = new_counts

        # Statistics after removing outliers
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

    # Execute this if you want to save the statistics, by default its false
    if save_stats:
        stats = pd.DataFrame(task_stats).transpose()
        stats.to_csv(os.path.join('stats', DATASET, 'task_info_162', 'R_without_14k_more_pids_task_stats_2.csv'))
        new_stats = pd.DataFrame(new_task_stats).transpose()
        new_stats.to_csv(
            os.path.join('stats', DATASET, 'task_info_162', 'R_without_14k_outliers_removed_more_pids_task_stats_2.csv'))

        task_renames = {'CookieTheft': 'Picture Description', 'Reading': 'Reading'}
        for task in TASKS:
            plt.rc('font', size=15)

            plt.figure()
            # plt.title(task_renames[task] + ' sequence length distribution')
            plt.xlabel('length of sequence')
            plt.ylabel('number of participants')
            # hist1 = sns.histplot(task_lengths[task], bins=50)
            # hist1.axvline(round(task_stats[task]['90%ile'], 2), color='black', linestyle='--', alpha=0.6, lw=2)
            if task == 'CookieTheft':
                plt.xticks(np.arange(0, 4000, 500))
                plt.yticks(np.arange(0, 20, 2))
            else:
                plt.xticks(np.arange(400, 2250, 250))
                plt.yticks(np.arange(0, 26, 2))
            plt.savefig(os.path.join('stats', DATASET, 'task_info_162', 'Participant_dist_2_without_14k_' + task + '.png'))

            # plot for after truncation, what's left for each participant
            plt.figure()
            # hist2 = sns.histplot(new_task_lengths[task], bins=50)
            # plt.title(task_renames[task] + ' sequence length distribution - 90%ile truncated')
            plt.xlabel('length of sequence')
            plt.ylabel('number of participants')
            if task == 'CookieTheft':
                plt.xticks(np.arange(0, 1500, 200))
                plt.yticks(np.arange(0, 20, 2))

            if task == 'Reading':
                plt.xticks(np.arange(400, 1050, 100))
                plt.yticks(np.arange(0, 20, 2))
            # plt.hist(new_task_lengths[task], bins=100)
            plt.savefig(os.path.join('stats', DATASET, 'task_info_162', 'outlier_removed_seq_dist_2_' + task + '.png'))

            # plot for after truncation, what's removed from each participant
            rem = pd.concat((lens_df.PID, pd.Series(task_lengths[task]) - new_task_lengths[task]), axis=1)
            rem = rem.rename(columns={0: 'rem'})
            rem['percent_removed'] = (rem.rem*100)/(rem.rem + round(u_90))
            removed_task_percent[task] = rem[rem['percent_removed'] > 0].percent_removed

            # removed_task_percent_Reading = np.array([i*100/(i+round(u_90)) for i in removed_task_lengths[task]])

            plt.figure()
            # hist3 = sns.histplot(removed_task_percent[task], bins=50)
            # hist3 = sns.histplot(removed_task_percent_Reading, bins=len(removed_task_percent_Reading))
            # plt.title('Percentage of sequence length reduced in ' + task_renames[task])
            plt.xlabel('percent length removed')
            plt.ylabel('number of participants')
            plt.yticks(np.arange(0, 3))
            if task == 'CookieTheft':
                plt.xticks(np.arange(0, 70, 5))
            else:
                plt.xticks(np.arange(0, 55, 5))

            # hist3.axvline(round(task_stats[task]['90%ile'], 2), color='black', linestyle='--', alpha=0.6, lw=2)
            plt.savefig(os.path.join('stats', DATASET, 'task_info_162',
                                     'percent_portion_of_sequences_removed_without_14k_' + task + '.png'))

    return new_pids


class Preprocess:
    def __init__(self, max_sequence_length, pad_side, truncation_side, chunk_compatible=CHUNK_PROCESSING):
        self.max_sequence_length = max_sequence_length
        self.truncation_side = truncation_side
        self.pad_side = pad_side
        self.min_seq_length_factor = 100
        if chunk_compatible:
            self.max_sequence_length = math.ceil(
                self.max_sequence_length / self.min_seq_length_factor) * self.min_seq_length_factor

    def truncate(self, sequence: np.array) -> np.array:
        if self.truncation_side == 'post':
            return sequence[:self.max_sequence_length, :]
        else:
            return sequence[(sequence.shape[0] - self.max_sequence_length):, :]

    def pad_sequence(self, sequence: np.array) -> np.array:
        # make if then else block
        padding = np.full((self.max_sequence_length - len(sequence), sequence.shape[1]), PAD_VAL)
        if self.pad_side == 'post':
            return np.append(sequence, padding, axis=0)
        elif self.pad_side == 'pre':
            return np.append(padding, sequence, axis=0)
        elif self.pad_side == 'None':
            return sequence

    def transform(self, sequence: np.array) -> np.array:
        # scaling data values to [0,1]
        scaler = MinMaxScaler()
        sequence = scaler.fit_transform(sequence)

        # this happens when sequence is longer than the required length
        sequence = self.truncate(sequence)

        # this happens when the sequence is shorter than the required length
        if len(sequence) < self.max_sequence_length:
            sequence = self.pad_sequence(sequence)

        return sequence

    def pad_and_truncate(self, task_data: dict) -> dict:
        # task_data is just data[task], data comes from get_data
        pids = list(task_data.keys())
        truncated_data = {}
        for pid in pids:
            pid_data = task_data[pid]
            truncated_data[pid] = self.transform(pid_data)

        return truncated_data


def make_batches(x, y, batch_size=None, shuffle=False):
    num_batches = x.shape[0] // batch_size if batch_size is not None else 1
    x_batched = None
    y_batched = None
    if shuffle:
        pass

    else:
        x_batched = np.array(np.split(x, num_batches, axis=0))
        y_batched = np.array(np.split(y, num_batches, axis=0))

    return x_batched, y_batched, num_batches


"""# Neural Network Stuff """


def compute_batch_accuracy(o: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes the accuracy of the predictions over the items in a single batch
    :param o: the logit output of datum in the batch
    :param y: the correct class index of each datum
    :return the percentage of correct predictions as a value in [0,1]
    """
    preds = np.argmax(o.detach().cpu().numpy(), axis=1)
    y_true = np.argmax(y.detach().cpu().numpy(), axis=1)
    rights = np.count_nonzero(preds == y_true)
    wrongs = len(preds == y_true) - rights
    accuracy = rights / (rights + wrongs)
    return accuracy


def compute_metrics(y_true: np.array, y_pred: np.array) -> dict:
    """
    Computes the metrics for the given predictions and labels
    :param y_true: the ground-truth labels
    :param y_pred: the predictions of the model
    :return: the following metrics in a dict:
        * Sensitivity (TP rate) / Specificity (FP rate) / Combined
        * Accuracy / F1 / AUC
    """
    # sensitivity = recall_score(y_true, y_pred, pos_label=1)
    # specificity = recall_score(y_true, y_pred, pos_label=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    try:
        roc_auc_score(y_true, y_pred)
    except ValueError:
        print('y_true', np.bincount(np.array(y_true, dtype=int)))
        return {}

    return {
        "roc": roc_auc_score(y_true, y_pred),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "spec": tn / (tn + fp),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
        # "combined": (sensitivity + specificity) / 2,
    }


def compute_optimal_roc_threshold(y_true: np.array, y_1_scores: np.array) -> float:
    """
    Computes the optimal ROC threshold
    :param y_true: the ground truth
    :param y_1_scores: the predicted scores for the positive class
    :return: the optimal ROC threshold (defined for more than one sample, else 0.5)
    """
    fp_rates, tp_rates, thresholds = roc_curve(y_true, y_1_scores)
    best_threshold, dist = 0.5, 100

    for i, threshold in enumerate(thresholds):
        current_dist = np.sqrt((np.power(1 - tp_rates[i], 2)) + (np.power(fp_rates[i], 2)))
        if current_dist <= dist:
            best_threshold, dist = threshold, current_dist

    return best_threshold


def pprint_metrics(metrics: dict):
    for metric, value in metrics.items():
        print(("\t - {} " + "".join(["."] * (15 - len(metric))) + " : {}").format(metric, value))


class GRU2(nn.Module):

    def __init__(self, num_layers, hidden_size, dropout):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=INPUT_SIZE,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          bidirectional=BIDIRECTIONAL,
                          dropout=dropout,
                          batch_first=True)

        self.fc = nn.Linear(2 * self.hidden_size if BIDIRECTIONAL else self.hidden_size, OUTPUT_SIZE)

    def init_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(2 * self.num_layers if BIDIRECTIONAL else self.num_layers, batch_size, self.hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        @param x: input sequence of shape [batch_size, sequence_length, num_features]
        @param h: hidden state of shape [num_layers (2x if bidirectional), batch_size, hidden_size]
        """
        o, h = self.gru(x, h)
        # return self.fc(o[:, -1, :]), h    # check this! what are you returning here (a single value)
        return self.fc(o[-1, :]), h  # this returns predictions for each timestep of the input sequence


def weighted_average_early_loss(loss):
    # print(loss.shape)
    len_T = loss.shape[0]
    early_loss = 0.0
    for i in range(1, len_T):
        # weight can be ((len_T-i) / len_T) if you want to put more weight on the losses closer to the final timestep
        # otherwise, if you want to put equal weight on intermediate losses then just do (1/len_T) which Lipton recommends
        weight = (1/len_T)  # ((len_T-i) / len_T)  # or (1/len_T)
        early_loss += weight*loss[i]
    return early_loss


def TR_custom_loss(alpha):
    def inner_loss(y_pred, y_true):
        crit = nn.CrossEntropyLoss(reduce=False)
        loss = crit(y_pred, y_true)  # this will be of shape (batch_size, timesteps) = (1, seq_len) for single sequence of length seq_len
        # print('crit_loss', loss)
        return alpha*weighted_average_early_loss(loss[:-1]) + (1-alpha)*loss[-1]
    return inner_loss


def train2(network: torch.nn.Module, epochs: int, lr: float,
           x_train: np.array, y_train: np.array,
           criterion: Callable,
           task: str, seed: int, fold: int) -> tuple:
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    best_val_loss = 1000.0
    early_stopping_counter = 0

    fold_val_metrics = None

    loss_value, batch_accuracy = None, None
    batch_loss, batch_acc = None, None
    best_network = None

    x_train = [torch.from_numpy(seq_x).float().to(DEVICE) for seq_x in x_train]
    y_train = torch.from_numpy(y_train).float().to(DEVICE)
    num_seq = len(x_train)

    for epoch in tqdm(range(epochs), desc='training nested fold {}'.format(fold), disable=False):
        network = network.train()
        train_loss, train_accuracy = [], []

        for batch_num in tqdm(range(num_seq), desc='epoch: ' + str(epoch)):
            # print('batch_num = ', batch_num)
            # Zero out all the gradients
            optimizer.zero_grad()
            # Initialize the hidden state of the RNN and move it to device
            h = network.init_state(1).to(DEVICE)

            batch_loss, batch_acc = [], []

            x = x_train[batch_num][None, :, :]
            seq_len = x.shape[1]
            # y_batch = y_train[batch_num][None, :]
            y = y_train[batch_num].repeat(seq_len, 1)

            # num_divisions = x_batch.shape[1] // CHUNK_LEN  # CHUNK_LEN
            # if x_batch.shape[1] % CHUNK_LEN != 0:
            #     num_divisions += 1

            '''
            # print(x_batch.shape, num_divisions)
            for division in range(num_divisions):
                # Move training inputs and labels to device
                x = x_batch[:, division * CHUNK_LEN: (division + 1) * CHUNK_LEN, :]
                y = y_batch
                # x = torch.from_numpy(x).float().to(DEVICE)
                # y = torch.from_numpy(y_batch).float().to(DEVICE)  # [0,1]

                x.requires_grad = True

                # Predict
                o, h = network(x, h)

                # Compute the error
                loss = criterion(o, y)
                # check documentation of how y being two values - should just be 1 value that represents the actual class [0 or 1]

                # Backpropagate
                loss.backward(retain_graph=True)
                loss_value = loss.detach().item()
                batch_accuracy = compute_batch_accuracy(o, y)

                # assert here, if nan value found then stop - check loss function why loss goes to nan
                if np.isnan(loss_value):
                    # print('loss value is nan at ', epoch, batch_num, division, loss_value)
                    break
                    '''

            x.requires_grad = True

            # Predict
            o, h = network(x, h)

            # Compute loss
            loss = criterion(o, y)

            # backpropagate
            loss.backward(retain_graph=True)
            loss_value = loss.detach().item()
            batch_accuracy = compute_batch_accuracy(o, y)

            # Store all training loss and accuracy for computing avg
            optimizer.step()
            # batch_loss += [loss_value]
            # batch_acc += [batch_accuracy]

            train_loss += [loss_value]
            train_accuracy += [batch_accuracy]

        # Update model parameters
        avg_train_loss, avg_train_accuracy = np.nanmean(train_loss), np.nanmean(train_accuracy)

        # if VERBOSE:
        #     print("[ EPOCH {}/{} --> Avg train loss: {:.4f} - Avg train accuracy: {:.4f} ]".
        #           format(epoch + 1, epochs, avg_train_loss, avg_train_accuracy))
        #
        # val_metrics, _, pred_probs, threshold = evaluate2(network, x_val, y_val, pids_val,
        #                                                   criterion, pre_trained_threshold=0.5)

        # # Update best model
        # avg_val_loss = val_metrics["loss"]
        # if avg_val_loss < best_val_loss:
        #     if VERBOSE == 2:
        #         print("\n Avg val loss ({:.4f}) better that current best val loss ({:.4f}) \n".format(avg_val_loss,
        #                                                                                               best_val_loss))
        #         print("\n --> Saving new best model... \n")
        #
        #     # save the best performing model for this fold, across all epochs
        #     best_val_loss = val_metrics["loss"]
        #     fold_val_metrics = val_metrics
        #     best_network = network
        #     early_stopping_counter = 0
        #
        # if early_stopping_counter >= 10:
        #     print('early stopping achieved, loss wasnt getting better', best_val_loss, avg_val_loss)
        #     break
        #
        # if epoch >= 100:
        #     early_stopping_counter += 1
        # print(avg_val_loss, best_val_loss, early_stopping_counter)
        # implement early stopping - check if loss changes too much across some epochs - if after 10 epochs the loss doesn't change then stop training

    return network  # fold_val_metrics, best_network


def evaluate2(network: torch.nn.Module,
              x_test: np.array, y_test: np.array, pids_test: np.array,
              criterion: torch.optim, pre_trained_threshold: float = None) -> tuple:
    network = network.eval()

    y_scores, y_true = [], []
    loss, accuracy = [], []

    preds = {}
    pred_probs = {}

    metric_names = ['roc', 'acc', 'f1', 'prec', 'recall', 'spec', 'tp', 'fp', 'fn', 'tn', 'loss', 'avg_batch_acc']
    batch_metrics = {metric: [] for metric in metric_names}

    # x_test = x_test[0]
    # y_test = y_test[0]
    num_seq = len(x_test)

    yhat_probs = []
    with torch.no_grad():
        for num in range(num_seq):
            x = torch.from_numpy(x_test[num]).float().to(DEVICE)[None, :, :]
            y = torch.from_numpy(y_test[num]).float().to(DEVICE)[None, :]

            # Initialize the hidden state of the RNN and move it to device
            h = network.init_state(x.shape[0]).to(DEVICE)

            # Predict
            o, _ = network(x, h)
            o = o.to(DEVICE)

            loss_value = criterion(o, y).item()
            batch_accuracy = compute_batch_accuracy(o, y)

            # yhat_probs = torch.sigmoid(o).detach().cpu().numpy().tolist()
            yhat_probs_seq = torch.softmax(o, dim=1).detach().cpu().numpy()
            pred_probs[pids_test[num]] = yhat_probs_seq[0]
            yhat_probs += [yhat_probs_seq[0]]

            # Store all validation loss and accuracy values for computing avg
            loss += [loss_value]
            accuracy += [batch_accuracy]

        if pre_trained_threshold:
            threshold = pre_trained_threshold
        else:
            threshold = compute_optimal_roc_threshold(y_test[:, 0], yhat_probs[:, 0])

        yhat_probs = np.array(yhat_probs)
        y_pred = np.array(yhat_probs[:, 0] >= threshold, dtype=int)

        # Compute the validation metrics
        avg_loss, avg_accuracy = np.mean(loss), np.mean(accuracy)
        metrics = compute_metrics(y_test[:, 0], y_pred)
        metrics["loss"] = avg_loss
        metrics["avg_batch_acc"] = avg_accuracy

        for k in metric_names:
            batch_metrics[k] += [metrics[k]]

    for k in metric_names:
        batch_metrics[k] = np.mean(batch_metrics[k])

    return batch_metrics, y_pred, pred_probs, threshold


"""# Cross Validation"""
# other things to do here -------
# change name of output_folders to show different configurations
# see if saving the network inside the nested CV is required, or if just returning the best network from Train function is enough
# check the time it takes, cross-check the uniqueness of PIDs in each split of the inner CV
# check if you're doing too many for loops or not
# check how to get the best network from inside the inner CV (best one across the nested folds)


def inner_cv(task, n_inner_folds, data, fold_pids, outer_seed, combination):
    epochs = EPOCHS
    lr, n_layers, n_hidden, dropout = combination

    # fold_pids = train_splits[0]
    inner_splits = np.array_split(fold_pids, n_inner_folds)
    inner_train_splits = []
    inner_test_splits = []

    for s in range(n_inner_folds):
        inner_test_splits.append(inner_splits[s])
        inner_train_splits.append(np.setdiff1d(fold_pids, inner_splits[s]))
    inner_metrics = {'roc': [], 'acc': [], 'f1': [], 'prec': [], 'recall': [], 'spec': [],
                     'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    avg_auc = []
    avg_loss = []
    best_inner_net = None
    best_inner_auc = -np.inf

    for nested_fold in range(n_inner_folds):
        x_train_inner = np.array([data[pid] for pid in inner_train_splits[nested_fold]])
        y_train_inner = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in inner_train_splits[nested_fold]])
        pids_train_inner = np.array(inner_train_splits[nested_fold])

        # test set
        x_test_inner = np.array([data[pid] for pid in inner_test_splits[nested_fold]])
        y_test_inner = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in inner_test_splits[nested_fold]])
        pids_test_inner = np.array(inner_test_splits[nested_fold])

        # initialize network with the given combination of hyparameters inside each fold
        net = GRU2(n_layers, n_hidden, dropout).float().to(DEVICE)
        # criterion = nn.CrossEntropyLoss()
        alpha = 0.5
        criterion = TR_custom_loss(alpha)

        # train the network on the inner train set, and validate with the inner validation set, get metrics
        best_net = train2(net, epochs, lr,
                          x_train_inner, y_train_inner,
                          criterion, task, outer_seed, nested_fold)

        # test the best_net found from training across epochs with inner test set, get metrics
        nested_test_metrics, _, _, _, = evaluate2(best_net, x_test_inner, y_test_inner, pids_test_inner,
                                                  criterion, pre_trained_threshold=0.5)

        # add the metrics across nested folds to make the final metrics for this combination of hyperparameters
        avg_auc += [nested_test_metrics['roc']]
        avg_loss += [nested_test_metrics['loss']]

        if best_inner_auc < nested_test_metrics['roc']:
            best_inner_net = best_net

    avg_auc = np.mean(avg_auc)
    avg_loss = np.mean(avg_loss)

    return avg_auc, avg_loss, best_inner_net


# Cross Validate function
def cross_validate(task, data, seed):
    torch.manual_seed(seed)
    pids = list(data.keys())
    random.Random(seed).shuffle(pids)

    # creating splits
    splits = np.array_split(pids, NFOLDS)
    train_splits = []
    val_splits = []
    test_splits = []

    if VAL_SET:
        for s in range(NFOLDS):
            test_splits.append(splits[s])
            val_splits.append(splits[(s + 1) % 10])
            train_splits.append(np.array(pids)[~np.in1d(pids, np.append(splits[s], splits[(s + 1) % 10]))])

    else:
        val_splits = None
        for s in range(NFOLDS):
            test_splits.append(splits[s])
            train_splits.append(np.array(pids)[~np.in1d(pids, splits[s])])

    metrics = {'roc': [], 'acc': [], 'f1': [], 'prec': [], 'recall': [], 'spec': [],
               'tp': [], 'fp': [], 'fn': [], 'tn': [],
               'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
               'n_train_hc': [], 'n_train_e': [], 'n_test_hc': [], 'n_test_e': [], 'n_val_hc': [], 'n_val_e': []}

    # to keep final prediction probabilities across all folds
    seed_pred_probs = {}
    seed_preds = {}

    '''
    import pickle
    for fold in range(NFOLDS):
        d = {'train': train_splits[fold], 'test': test_splits[fold]}  # , 'val': val_splits[fold]}
        with open(os.path.join(os.getcwd(), 'splits', 'seed_0_splits_{}'.format(fold)), 'wb') as file:
            pickle.dump(d, file)
    '''

    # going through all folds to create fold-specific train-test sets
    for fold in tqdm(FOLDS_TO_PROCESS, desc='seed: {} training'.format(seed)):
        # making train:test x, y, labels
        x_val = y_val = pids_val = None

        # train set
        x_train = np.array([data[pid] for pid in train_splits[fold]])
        y_train = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in train_splits[fold]])
        pids_train = np.array(train_splits[fold])

        # test set
        x_test = np.array([data[pid] for pid in test_splits[fold]])
        y_test = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in test_splits[fold]])
        pids_test = np.array(test_splits[fold])

        # if VAL_SET:
        #     x_val = np.array([data[pid] for pid in val_splits[fold]])
        #     y_val = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in val_splits[fold]])
        #     pids_val = np.array(val_splits[fold])

        ''' nested CV here '''
        n_inner_folds = 5

        # early-stopping with hyperparameters - if it's not improving with some change in hyperaparameters then don't
        # try the next ones (in combination zip or in grid loops)
        # epoch_tests = [200, 500, 1000]
        learning_rates = [0.001]
        n_layer_tests = [1, 2]
        n_hidden_tests = [64, 128, 256]
        dropout_tests = [0.1, 0.5]
        all_combinations = list(
            itertools.product(learning_rates, n_layer_tests, n_hidden_tests, dropout_tests))

        best_auc = -np.inf
        best_loss = np.inf
        best_configuration_in_this_fold = None
        best_configuration_network = None

        # hyperparameter tuning by calling inner_cv which is the nested CV by iterating through all possible
        # hyperparameter configurations (manual Grid Search) for 5 fold inner CV
        for combination in tqdm(all_combinations, desc='checking all hpp combinations for fold {}'.format(fold)):
            auc, loss, config_net = inner_cv(task, n_inner_folds, data, train_splits[fold], seed, combination)
            if auc > best_auc and loss < best_loss:
                print('this combination is the best so far:  ', combination)
                best_auc, best_loss, best_configuration_in_this_fold = auc, loss, combination
                best_configuration_network = config_net
                fold_model_save_path = os.path.join(model_save_path,
                                                    '{}_seed_{}_fold_{}_best_model_{}_{}_{}.pth'.
                                                    format(task, seed, fold, combination[1], combination[2], int(combination[3]*100)))

        torch.save(best_configuration_network.state_dict(), fold_model_save_path)

        print('\n HYPERPARAMETER TUNING DONE for FOLD ', fold)
        best_lr, best_n_layer, best_n_hidden, best_dropout = best_configuration_in_this_fold
        print('\n BEST HPP COMBINATION: {} {} {} '.format(best_n_layer, best_n_hidden, best_dropout))

        network = GRU2(best_n_layer, best_n_hidden, best_dropout).float().to(DEVICE)
        criterion = nn.CrossEntropyLoss()

        # train
        trained_network = train2(network, EPOCHS, best_lr, x_train, y_train,
                                 criterion, task, seed, fold)

        print('\nTraining complete --------------------------------------------------------------------')

        # test
        # saved_model_fold_path = os.path.join(model_save_path,
        #                                      '{}_seed_{}_fold_{}_best_model_{}_{}_{}.pth'.
        #                                      format(task, seed, fold, best_n_layer, best_n_hidden, int(best_dropout*100)))
        # network.load_state_dict(torch.load(saved_model_fold_path))
        test_metrics, fold_preds, fold_pred_probs, _ = evaluate2(trained_network, x_test, y_test, pids_test,
                                                                 criterion, pre_trained_threshold=0.5)

        # saving metrics
        for m in list(test_metrics.keys()):
            if m not in metrics.keys():
                continue
            metrics[m].append(test_metrics[m])

        # updating prediction probabilities for the seed
        seed_pred_probs.update(fold_pred_probs)

        save_results(task, [metrics], seed_pred_probs, seed=fold)
    return metrics


# save results function
def save_results(task, saved_metrics, pred_probs=None, seed=None, averaged_with_lang=False,
                 output_foldername=OUTPUT_FOLDERNAME):
    models_folder = os.path.join(os.getcwd(), 'models', 'chunk_' + DATASET, output_foldername)
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    models_params = os.path.join(models_folder, 'torch_params')
    ParamsHandler.save_parameters(TORCH_PARAMS, models_params)

    output_folder = os.path.join(os.getcwd(), 'results', 'chunk_' + DATASET,
                                 output_foldername) if not averaged_with_lang \
        else os.path.join(os.getcwd(), 'results', 'chunk_webcam_canary_lang', output_foldername + '_lang')

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    shutil.copy(os.path.join(os.getcwd(), 'params', 'torch_params.yaml'), output_folder)
    shutil.copy(os.path.join(os.getcwd(), os.path.basename(__file__)), output_folder)

    feature_set_names = {'PupilCalib': 'ET_Basic', 'CookieTheft': 'Eye', 'Reading': 'Eye_Reading', 'Memory': 'Audio'}
    metrics = ['acc', 'roc', 'fms', 'precision', 'recall', 'specificity', 'tp', 'fp', 'fn', 'tn']
    metric_names = {'acc': 'acc', 'roc': 'roc', 'fms': 'f1', 'precision': 'prec', 'recall': 'recall',
                    'specificity': 'spec', 'tp': 'tp', 'fp': 'fp', 'fn': 'fn', 'tn': 'tn'}

    print('saving result for specified seed, not iterated', seed)
    seed_path = os.path.join(output_folder, str(seed))
    if not os.path.exists(seed_path):
        os.mkdir(seed_path)

    seed_metrics = saved_metrics[0]
    dfs = []
    for metric in metrics:
        metric_name = metric_names[metric]
        metric_data = seed_metrics[metric_name]
        data = pd.DataFrame(metric_data, columns=['1'])
        data['metric'] = metric
        data['model'] = 'GRU'
        data['method'] = 'multi-chunk'
        dfs += [data]

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_csv(os.path.join(seed_path, 'results_new_features_{}.csv'.format(feature_set_names[task])), index=False)

    # only execute this if averaged_with_lang is False - average pred_probs will be written by average_results function
    if not averaged_with_lang:
        # for writing prediction probabilities into a file
        headers = ['model', 'PID', 'prob_0', 'prob_1', 'pred']
        df_pred_probs = []
        for pid in pred_probs.keys():
            pid_vals = pred_probs[pid]
            prob_0 = pid_vals[1]
            prob_1 = pid_vals[0]
            pred = prob_1 > 0.5
            row = ['GRU', pid, prob_0, prob_1, pred]
            df_pred_probs.append(row)

        df_pred_probs = pd.DataFrame(df_pred_probs, columns=headers, index=None)
        df_pred_probs.to_csv(os.path.join(seed_path, 'predictions_results_new_features_{}.csv'.
                                          format(feature_set_names[task])), index=False)

    print('results saved for {}'.format(task))


def average_pred_probs(gru_result_foldername=OUTPUT_FOLDERNAME):
    """
    function average_pred_probs --> takes in the seed path, and averages prediction probabilites from canary's language
    predictions and the GRU predictions from this file. Webcam or Tobii. Happens after GRU has completed training and
    evaluating, saves results
    :param gru_result_foldername: foldername of the GRU results
    """

    GRU_result_path = os.path.join(results_path, gru_result_foldername)
    canary_lang_path = os.path.join(os.getcwd(), 'results', 'canary_lang')
    feature_set_names_eye = {'PupilCalib': 'ET_Basic', 'CookieTheft': 'Eye', 'Reading': 'Eye_Reading',
                             'Memory': 'Audio'}
    feature_set_names_lang = {'CookieTheft': 'Language', 'Reading': 'NLP_Reading', 'Memory': 'Audio'}

    for seed in tqdm(SEEDS, desc='averaging GRU results with canary lang', disable=not VERBOSE):
        GRU_seed_path = os.path.join(GRU_result_path, str(seed))
        canary_seed_path = os.path.join(canary_lang_path, str(seed))

        for task in TASKS:
            filename_eye = 'predictions_results_new_features_{}.csv'.format(feature_set_names_eye[task])
            filename_lang = 'predictions_results_new_features_{}.csv'.format(feature_set_names_lang[task])
            GRU_seed_file = pd.read_csv(os.path.join(GRU_seed_path, filename_eye))
            canary_lang_file = pd.read_csv(os.path.join(canary_seed_path, filename_lang))
            cols = GRU_seed_file.columns

            canary_model = 'RandomForest'  # currently choosing RandomForest as the only model to average predictions with
            canary_lang_data = canary_lang_file[canary_lang_file.model == canary_model]
            superset_pids = np.union1d(GRU_seed_file.PID, canary_lang_data.PID)

            # using superset_pids to make averaged results - probabilities from each pid is averaged across all found
            # instances. If PID in both sets (GRU and Canary lang, then average across 2, otherwise keep it the way it is)
            averaged_df = []
            for pid in superset_pids:
                thingy = np.zeros(3)
                if pid in list(GRU_seed_file.PID):
                    GRU_pid_data = GRU_seed_file[GRU_seed_file.PID == pid]
                    thingy[0] += GRU_pid_data.prob_0.iat[0]
                    thingy[1] += GRU_pid_data.prob_1.iat[0]
                    thingy[2] += 1

                if pid in list(canary_lang_data.PID):
                    canary_pid_data = canary_lang_data[canary_lang_data.PID == pid]
                    thingy[0] += canary_pid_data.prob_0.iat[0]
                    thingy[1] += canary_pid_data.prob_1.iat[0]
                    thingy[2] += 1

                prob_0 = thingy[0] / thingy[2]
                prob_1 = thingy[1] / thingy[2]
                pred = prob_1 > prob_0
                averaged_df.append(['Data_Ensemble_LF GRU-Lang', pid, prob_0, prob_1, pred])

            averaged_df = pd.DataFrame(averaged_df, columns=cols)
            averaged_save_path = os.path.join(os.getcwd(), 'results',
                                              'chunk_webcam_canary_lang', gru_result_foldername + '_lang')

            # recalculate metrics after doing averaging
            recalculate_metrics_after_averaging(task, superset_pids, seed, averaged_df)

            if not os.path.exists(averaged_save_path):
                os.mkdir(averaged_save_path)

            averaged_seed_path = os.path.join(averaged_save_path, str(seed))
            if not os.path.exists(averaged_seed_path):
                os.mkdir(averaged_seed_path)

            averaged_df.to_csv(os.path.join(averaged_seed_path, '{}_{}.csv'.
                                            format('predictions_results_new_features', task)), index=False)

    # ResultsHandler.compile_results('chunk_webcam_canary_lang', OUTPUT_FOLDERNAME + '_lang')
    ResultsHandler.compile_results('chunk_webcam_canary_lang', gru_result_foldername + '_lang')


def recalculate_metrics_after_averaging(task: str, superset_pids: np.array, seed: int, averaged_df: pd.DataFrame):
    superset_pids = list(superset_pids)
    random.Random(seed).shuffle(superset_pids)
    splits = np.array_split(superset_pids, 10)

    metric_names = ['roc', 'acc', 'f1', 'prec', 'recall', 'spec', 'tp', 'fp', 'fn', 'tn']
    seed_metrics = {metric: [] for metric in metric_names}

    for i in splits:
        split_data = averaged_df[averaged_df.PID.isin(i)]
        pids_split = list(split_data.PID)
        y_true_split = [1 if pid.startswith('E') else 0 for pid in pids_split]
        y_pred_split = list(split_data.pred)

        fold_metrics = compute_metrics(y_true_split, y_pred_split)
        for k in metric_names:
            seed_metrics[k] += [fold_metrics[k]]

    save_results(task, [seed_metrics], seed=seed, averaged_with_lang=True)


def main():
    # global SEEDS
    global TASKS
    # global NFOLDS
    global FOLDS_TO_PROCESS

    if CLUSTER:
        # after this, this variable SEEDS should not change
        FOLDS_TO_PROCESS = np.arange(int(sys.argv[2]), int(sys.argv[3]))
        TASKS = [sys.argv[1]]

    # getting data and removing outliers
    data = get_data()
    new_pids = remove_outliers(data, percentile_threshold=100)  # percentile threshold 100 removes none

    # getting data for each task, PreProcessing them and running Cross Validation
    for task in TASKS:
        print('processing {} task'.format(task))
        task_info = new_pids[new_pids.task == task]
        task_pids = list(task_info.PID)
        # task_median_length = task_info.len.median()                     # for FINAL_LENGTH to be task_median_length, if needed
        task_90_perc_length = round(np.percentile(task_info.len,
                                                  90))  # this is used to keep 100% PIDs, but truncate everything to 90% ile length

        task_data = get_data(task_pids, task)[task]
        data = Preprocess(task_90_perc_length, PAD_WHERE, TRUNCATE_WHERE, chunk_compatible=False). \
            pad_and_truncate(task_data)

        print('SEEDS are ', SEEDS)
        for seed in range(1):
            cross_validate(task, data, seed)

    # ResultsHandler.compile_results('chunk_' + DATASET, OUTPUT_FOLDERNAME)

    # average results of GRU predictions and Canary language results and save them
    # average_pred_probs()


if __name__ == '__main__':
    main()

# Extra functions
"""
def process_input_data_into_all_strategy():
    for task in TASKS:
        for pid in PIDS:
            pid_path = os.path.join(data_path, 'LSTM', pid)
            pid_save_path = os.path.join(data_path, 'webcam_all', pid)
            if not os.path.exists(pid_save_path):
                os.mkdir(pid_save_path)

            file = pd.read_csv(os.path.join(pid_path, task + '.csv'))
            x_cols = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']

            df = file[x_cols]
            x = np.array(df)
            avg_df = pd.DataFrame(np.mean((x[:, :3], x[:, 3:6]), axis=0),
                                  columns=['gaze_avg_x', 'gaze_avg_y', 'gaze_avg_z'])

            new_df = pd.concat((df, avg_df), axis=1)
            new_df.to_csv(os.path.join(pid_save_path, task + '.csv'), index=False)
"""

'''
def process_tobii_data_into_tasks_all_strategy():
    pids_to_skip = []
    for task in TASKS:
        for pid in tqdm(PIDS):
            pid_path = os.path.join(data_path, 'eye_movement')
            pid_save_path = os.path.join(data_path, 'Tobii_all', pid)
            if not os.path.exists(pid_save_path):
                os.mkdir(pid_save_path)

            # if file has already been saved, skip PID
            # if os.path.exists(os.path.join(pid_save_path, task+'.csv')):
            #     continue

            filepath = os.path.join(pid_path, 'Gaze_' + pid + '.tsv')
            file = pd.read_csv(filepath, sep='\t')
            x_cols = ['GazePointLeftX (ADCSpx)', 'GazePointLeftY (ADCSpx)',
                      'GazePointRightX (ADCSpx)', 'GazePointRightY (ADCSpx)',
                      'GazePointX (ADCSpx)', 'GazePointY (ADCSpx)',
                      'PupilLeft', 'PupilRight',
                      'ValidityLeft', 'ValidityRight']

            task_start = ttf[(ttf.StudyID == pid) & (ttf.Task == task)].timestampIni.iat[0]
            task_end = ttf[(ttf.StudyID == pid) & (ttf.Task == task)].timestampEnd.iat[0]
            within_task_data = file[file.RecordingTimestamp.between(task_start+1, task_end-1)]
            df = within_task_data[x_cols]

            """
            Code from Shane for pre-processing Tobii raw data
            """
            # percentage of invalid rows in dataset for this PID and task - from Shane
            num_invalid = df[(df['ValidityLeft'] == 4.0) & (df['ValidityRight'] == 4.0)].shape[0]
            total_rows = df.shape[0]
            if total_rows == 0:
                pids_to_skip.append(pid)
            perc_invalid = num_invalid / total_rows

            # fix_missing_eye - if one eye is invalid, but the other is valid, then copy valid values into invalid eye
            x = df.values
            i_left = (x[:, 8] == 4.0) & (x[:, 9] == 0.0)  # get indicies of rows with ValidityLeft = 4.0 (invalid)
            i_right = (x[:, 8] == 0.0) & (x[:, 9] == 4.0)  # get indicies of rows with ValidityRight = 4.0 (invalid)

            # fixing gazepoint values
            x[i_left, 0:2] = x[i_left, 2:4]
            x[i_right, 2:4] = x[i_right, 0:2]

            # fixing pupil values
            x[i_left, 6] = x[i_left, 7]
            x[i_right, 7] = x[i_right, 6]

            # fixing validity values
            x[i_left, 8] = x[i_left, 9]
            x[i_right, 9] = x[i_right, 8]

            df = pd.DataFrame(x)

            # fix invalid_rows - if both eyes are invalid, put -1.0 everywhere
            x = df.values
            i_left = (x[:, 8] == 4.0)  # get indicies of rows with ValidityLeft = 4.0 (invalid)
            i_right = (x[:, 9] == 4.0)  # get indicies of rows with ValidityRight = 4.0 (invalid)
            x[i_left] = -1.0
            x[i_right] = -1.0

            # find nan values if there are any
            nan_rows = np.where(np.isnan(x).all(axis=1))
            for row in nan_rows:
                x[row] = -1.0
            df = pd.DataFrame(x, columns=x_cols)

            df.to_csv(os.path.join(pid_save_path, task+'.csv'), index=False)

            """
            Note: Found PIDS = ['EO-028', 'HI-045', 'EA-046', 'EL-114', 'ET-171'] as outliers
            They either don't have any data in their files, or the files have only the keypoints (when space was pressed
            to advance to the next task) and no data in between keypoints. 
            """
'''

'''
def subset_data(x_train, x_test, x_val, strategy):
    """
    function subset_data --> given train, test and validation inputs, create subsets based on the strategy required
    :param x_train:
    :param x_test:
    :param x_val:
    :param strategy: the data subset strategy used, depending on the strategy, combine the columns of the data
    :return:
    """
    fold_train = fold_test = fold_val = None
    if strategy == 'left':
        fold_train = np.array([x_train[i][:, :3] for i in range(len(x_train))])
        fold_test = np.array([x_test[j][:, :3] for j in range(len(x_test))])
        fold_val = np.array([x_val[j][:, :3] for j in range(len(x_val))]) if x_val is not None else None

    elif strategy == 'right':
        fold_train = np.array([x_train[i][:, 3:6] for i in range(len(x_train))])
        fold_test = np.array([x_test[j][:, 3:6] for j in range(len(x_test))])
        fold_val = np.array([x_val[j][:, 3:6] for j in range(len(x_val))]) if x_val is not None else None

    elif strategy == 'average':
        fold_train = np.array([np.mean((x_train[i][:, :3], x_train[i][:, 3:6]), axis=0) for i in range(len(x_train))])
        fold_test = np.array([np.mean((x_test[i][:, :3], x_test[i][:, 3:6]), axis=0) for i in range(len(x_test))])
        fold_val = np.array([np.mean((x_val[i][:, :3], x_val[i][:, 3:6]), axis=0) for i in
                             range(len(x_val))]) if x_val is not None else None

    elif strategy == 'both_eyes':
        fold_train = np.array([x_train[i][:, :6] for i in range(len(x_train))])
        fold_test = np.array([x_test[j][:, :6] for j in range(len(x_test))])
        fold_val = np.array([x_val[j][:, :6] for j in range(len(x_val))]) if x_val is not None else None

    elif strategy == 'all':
        both_train, both_test, both_val = subset_data(x_train, x_test, x_val, 'both_eyes')
        avg_train, avg_test, avg_val = subset_data(x_train, x_test, x_val, 'average')
        fold_train = np.array([np.hstack((both_train[i], avg_train[i])) for i in range(len(both_train))])
        fold_test = np.array([np.hstack((both_test[i], avg_test[i])) for i in range(len(both_test))])
        fold_val = np.array(
            [np.hstack((both_val[i], avg_val[i])) for i in range(len(both_val))]) if x_val is not None else None

    return fold_train, fold_test, fold_val
'''
