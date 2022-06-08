import pandas as pd
import numpy as np
import os
import sys
import torch
import math
import random

from typing import Union, Callable
from torch import nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, \
    roc_curve
from tqdm import tqdm

from ParamsHandler import ParamsHandler
from ResultsHandler import ResultsHandler

# Torch Device
if not torch.cuda.is_available():
    print("WARNING: running on CPU since GPU is not available")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(0)

# torch params
torch_params = ParamsHandler.load_parameters('torch_params')
# MP_FLAG = torch_params['multi_processing']
CLUSTER = torch_params['cluster']
SEEDS = np.arange(torch_params['seeds'])
NFOLDS = torch_params['folds']
TASKS = torch_params['tasks']
STRATEGY = torch_params['strategy']
OUTPUT_FOLDERNAME = torch_params['output_foldername']
DATA_DIM_DICT = torch_params['data_dim_dict']

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
MAX_SEQ_LEN = torch_params['max_seq_len']
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

if not OUTPUT_FOLDERNAME:
    OUTPUT_FOLDERNAME = 'full_CV_torch'. \
        format(NN_TYPE, NUM_LAYERS, LEARNING_RATE, DROPOUT, MAX_SEQ_LEN)

TORCH_PARAMS = torch_params
if not os.path.exists(os.path.join(os.getcwd(), 'models', OUTPUT_FOLDERNAME)):
    os.mkdir(os.path.join(os.getcwd(), 'models', OUTPUT_FOLDERNAME))

results_path = os.path.join(os.getcwd(), 'results', 'LSTM', 'stateful')
data_path = os.path.join(os.getcwd(), 'data')
ttf = pd.read_csv(os.path.join(data_path, 'TasksTimestamps.csv'))


def get_data(pids, tasks):
    if type(tasks) == str:
        tasks = [tasks]

    data = {task: {pid: None for pid in pids} for task in tasks}
    for task in tqdm(tasks, 'getting task data'):
        for pid in pids:
            pid_save_path = os.path.join(data_path, 'LSTM', pid)
            file = pd.read_csv(os.path.join(pid_save_path, task + '.csv'))
            x_cols = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
                      'gaze_angle_x', 'gaze_angle_y']
            x = np.array(file[x_cols])
            data[task][pid] = x

    return data


def remove_outliers(data, pids, tasks, percentile_threshold=100, save_stats=False):
    task_stats = {task: {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None, 'total': None,
                         'count': None, '90%ile': None, '95%ile': None}
                  for task in tasks}

    new_task_stats = {task: {'mean': None, 'std': None, 'median': None, 'min': None, 'max': None, 'total': None,
                             'count': None, '90%ile': None, '95%ile': None}
                      for task in tasks}

    new_pids = pd.DataFrame(columns=['PID', 'len', 'task'])
    for task in tasks:
        lens = np.array([[pid, len(data[task][pid])] for pid in pids], dtype='object')
        counts = lens[:, 1]
        md = pd.DataFrame(lens, columns=['PID', 'len'])
        md['task'] = task

        task_stats[task]['count'] = len(counts)
        task_stats[task]['mean'] = np.mean(counts)
        task_stats[task]['std'] = np.std(counts)
        task_stats[task]['median'] = np.median(counts)
        task_stats[task]['min'] = np.min(counts)
        task_stats[task]['max'] = np.max(counts)
        task_stats[task]['total'] = np.sum(counts)
        task_stats[task]['90%ile'] = np.percentile(counts, 90)
        task_stats[task]['95%ile'] = np.percentile(counts, 95)

        l_0 = 0
        u_90 = np.percentile(counts, percentile_threshold)

        new_lens = md[(md.len < u_90) & (md.len > l_0)]
        outliers = md[(md.len > u_90) | (md.len < l_0)]
        new_counts = new_lens.len

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

    if save_stats:
        stats = pd.DataFrame(task_stats).transpose()
        stats.to_csv(os.path.join('stats', 'LSTM', 'task_info', 'more_pids_task_stats.csv'))
        new_stats = pd.DataFrame(new_task_stats).transpose()
        new_stats.to_csv(os.path.join('stats', 'LSTM', 'task_info', 'outliers_removed_more_pids_task_stats.csv'))

    return new_pids


def subset_data(x_train, x_test, x_val, strategy):
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


class Preprocess:

    def __init__(self, max_sequence_length, pad_side, truncation_side, task_max_length=None):
        self.max_sequence_length = max_sequence_length
        self.truncation_side = truncation_side
        self.pad_side = pad_side
        self.min_seq_length_factor = 100
        if task_max_length:
            self.max_sequence_length = math.ceil(
                task_max_length / self.min_seq_length_factor) * self.min_seq_length_factor

    def truncate(self, sequence: np.array) -> np.array:
        truncated = {
            'post': sequence[:self.max_sequence_length, :],
            'pre': sequence[(sequence.shape[0] - self.max_sequence_length):, :]
        }
        return truncated[self.truncation_side]

    def pad_sequence(self, sequence: np.array) -> np.array:
        padding = np.full((self.max_sequence_length - len(sequence), sequence.shape[1]), PAD_VAL)
        padded = {
            'post': np.append(sequence, padding, axis=0),
            'pre': np.append(padding, sequence, axis=0)
        }
        return padded[self.pad_side]

    def transform(self, sequence: np.array) -> np.array:
        sequence = self.truncate(sequence)

        if len(sequence) < self.max_sequence_length:
            sequence = self.pad_sequence(sequence)

        return sequence

    def pad_and_truncate(self, task_data: dict) -> dict:
        pids = list(task_data.keys())
        truncated_data = {}
        for pid in pids:
            pid_data = task_data[pid]
            truncated_data[pid] = self.transform(pid_data)

        return truncated_data


def make_batches(x, y, batch_size=None):
    num_batches = x.shape[0] // batch_size if batch_size is not None else 1
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
    wrongs, rights = np.bincount(preds == y_true)
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

    return {
        "roc": roc_auc_score(y_true, y_pred),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "spec": tn / (tn + fp),
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


class GRU(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS

        self.gru = nn.GRU(input_size=INPUT_SIZE,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          bidirectional=BIDIRECTIONAL,
                          dropout=DROPOUT,
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
        return self.fc(o[:, -1, :]), h


def train(network: torch.nn.Module,
          x_train: np.array, y_train: np.array, num_batches: int,
          x_val: np.array, y_val: np.array, val_batches: int,
          criterion: Callable,
          fold: int) -> dict:
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    best_val_loss = 1000.0
    cv_val_metrics = None

    for epoch in tqdm(range(EPOCHS), desc='training fold {}'.format(fold), disable=True):
        network = network.train()
        train_loss, train_accuracy = [], []
        running_loss, running_accuracy = 0.0, 0.0

        for num in tqdm(range(num_batches), desc='training epoch {}/{}'.format(epoch + 1, EPOCHS), disable=True):
            # Zero out all the gradients
            optimizer.zero_grad()

            # Initialize the hidden state of the RNN and move it to device
            h = network.init_state(x_train.shape[1]).to(DEVICE)

            batch_loss, batch_acc = [], []
            batch_run_loss, batch_run_acc = 0.0, 0.0

            num_divisions = x_train.shape[2] // MAX_SEQ_LEN
            x_batch = x_train[num]
            y_batch = y_train[num]

            for division in range(num_divisions):
                # Move training inputs and labels to device
                x = x_batch[:, division * MAX_SEQ_LEN: (division + 1) * MAX_SEQ_LEN, :]
                x = torch.from_numpy(x).float().to(DEVICE)
                y = torch.from_numpy(y_batch).float().to(DEVICE)

                x.requires_grad = True
                # y.requires_grad = True

                # Predict
                o, h = network(x, h)
                # o = o.to(DEVICE)
                # h = h.to(DEVICE)

                # Compute the error
                loss = criterion(o, y)
                # Backpropagate
                loss.backward(retain_graph=True)
                loss_value = loss.item()
                batch_accuracy = compute_batch_accuracy(o, y)

                if np.isnan(loss_value):
                    print('loss value is nan at ', epoch, division, loss_value)

                # Accumulate training loss and accuracy for the log
                # batch_run_loss += loss_value
                # batch_run_acc += batch_accuracy

            # Store all training loss and accuracy for computing avg
            optimizer.step()
            # print(batch_loss)
            batch_loss += [loss_value]
            batch_acc += [batch_accuracy]

        train_loss += [np.nanmean(batch_loss)]
        train_accuracy += [np.nanmean(batch_acc)]

        # Update model parameters
        avg_train_loss, avg_train_accuracy = np.nanmean(train_loss), np.nanmean(train_accuracy)

        if VERBOSE:
            print("[ EPOCH {}/{} --> Avg train loss: {:.4f} - Avg train accuracy: {:.4f} ]".
                  format(epoch + 1, EPOCHS, avg_train_loss, avg_train_accuracy))

        val_metrics = evaluate(network, x_val, y_val, val_batches, criterion)

        # Pretty print the validation metrics
        # if VERBOSE:
        #     print("\n Validation metrics for epoch {}/{}: \n".format(epoch + 1, EPOCHS))
        #     pprint_metrics(val_metrics)

        # Update best model
        avg_val_loss = val_metrics["loss"]
        if avg_val_loss < best_val_loss:
            if VERBOSE:
                print("\n Avg val loss ({:.4f}) better that current best val loss ({:.4f}) \n".format(avg_val_loss,
                                                                                                      best_val_loss))
                print("\n --> Saving new best model... \n")
            torch.save(network.state_dict(),
                       os.path.join(os.getcwd(), 'models', OUTPUT_FOLDERNAME, 'fold_{}_best_model.pth'.format(fold)))
            best_val_loss = val_metrics["loss"]
            cv_val_metrics = val_metrics

    return cv_val_metrics


"""## Evaluation"""


def evaluate(network: torch.nn.Module,
             x_test: np.array, y_test: np.array, num_test: int,
             criterion: torch.optim) -> dict:
    network = network.eval()

    y_scores, y_true = [], []
    loss, accuracy = [], []
    running_loss, running_accuracy = 0.0, 0.0

    with torch.no_grad():
        for num in range(num_test):
            x = torch.from_numpy(x_test[num]).float().to(DEVICE)
            y = torch.from_numpy(y_test[num]).float().to(DEVICE)

            # Initialize the hidden state of the RNN and move it to device
            h = network.init_state(x.shape[0]).to(DEVICE)

            # Predict
            o, _ = network(x, h)
            o = o.to(DEVICE)

            loss_value = criterion(o, y).item()
            batch_accuracy = compute_batch_accuracy(o, y)

            # Accumulate validation loss and accuracy for the log
            # running_loss += loss_value
            # running_accuracy += batch_accuracy

            # Store all validation loss and accuracy values for computing avg
            loss += [loss_value]
            accuracy += [batch_accuracy]

            # Store predicted scores and ground truth labels
            y_scores += torch.exp(o).detach().cpu().numpy().tolist()
            y_true += y.cpu().numpy().tolist()

            y_scores, y_true = np.array(y_scores).reshape((len(y_scores), 2)), np.array(y_true)

    # Compute predicted labels based on the optimal ROC threshold
    threshold = compute_optimal_roc_threshold(y_true[:, 0], y_scores[:, 0])
    y_pred = np.array(y_scores[:, 0] >= threshold, dtype=np.int)

    # Compute the validation metrics
    avg_loss, avg_accuracy = np.mean(loss), np.mean(accuracy)
    metrics = compute_metrics(y_true[:, 0], y_pred)
    metrics["loss"] = avg_loss
    metrics["accuracy"] = avg_accuracy

    return metrics


"""# Cross Validation"""


# Cross Validate function
def cross_validate(task, data, seed):
    torch.manual_seed(seed)

    nfolds = NFOLDS
    val_set = VAL_SET

    pids = list(data.keys())
    random.Random(seed).shuffle(pids)

    if val_set:
        nfolds = NFOLDS // 2
        test_splits_c = np.array_split(pids, nfolds)  # currently nfolds = 10, this divides into 5 parts
        train_splits = [np.array(pids)[~np.in1d(pids, i)] for i in test_splits_c]  # makes 80:20 splits
        val_splits = [np.array_split(i, 2)[0] for i in test_splits_c]
        test_splits = [test_splits_c[i][~np.in1d(test_splits_c[i], val_splits[i])] for i in range(len(test_splits_c))]

    else:
        test_splits = np.array_split(pids, nfolds)
        train_splits = [np.array(pids)[~np.in1d(pids, i)] for i in test_splits]
        val_splits = None

    metrics = {'roc': [], 'acc': [], 'f1': [], 'prec': [], 'recall': [], 'spec': [],
               'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
               'n_train_hc': [], 'n_train_e': [], 'n_test_hc': [], 'n_test_e': []}

    # going through all folds to create fold-specific train-test sets
    for fold in tqdm(range(nfolds), desc='seed: {} training'.format(seed)):
        # making train:test x, y, labels
        # fold = 0
        x_val = y_val = None
        x_train = np.array([data[pid] for pid in train_splits[fold]])
        y_train = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in train_splits[fold]])
        # labels_train = np.array([pid for pid in train_splits[fold]])

        x_test = np.array([data[pid] for pid in test_splits[fold]])
        y_test = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in test_splits[fold]])
        # labels_test = np.array([pid for pid in test_splits[fold]])

        n_train_hc, n_train_e = np.bincount(y_train[:, 0])
        n_test_hc, n_test_e = np.bincount(y_test[:, 0])

        metrics['n_train_hc'].append(n_train_hc)
        metrics['n_train_e'].append(n_train_e)
        metrics['n_test_hc'].append(n_test_hc)
        metrics['n_test_e'].append(n_test_e)

        if val_set:
            x_val = np.array([data[pid] for pid in val_splits[fold]])
            y_val = np.array([[[1, 0] if pid.startswith('E') else [0, 1]][0] for pid in val_splits[fold]])
            # labels_val = np.array([pid for pid in val_splits[fold]])

        # creating subset based on strategy
        x_train, x_test, x_val = subset_data(x_train, x_test, x_val, STRATEGY)

        x_train, y_train, num_batches = make_batches(x_train, y_train, batch_size=BATCH_SIZE)
        x_test, y_test, num_test = make_batches(x_test, y_test, batch_size=BATCH_SIZE)
        x_val, y_val, num_val = make_batches(x_val, y_val, batch_size=BATCH_SIZE)

        network = GRU().float().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        cv_val_metrics = train(network, x_train, y_train, num_batches, x_val, y_val, num_val, criterion, fold)

        # saving metrics
        for metric in list(cv_val_metrics.keys()):
            if metric == 'loss' or metric == 'accuracy':
                continue
            metrics[metric].append(cv_val_metrics[metric])

    # print('saving {} seed metrics'.format(seed))
    save_results(task, [metrics], seed=seed)
    return metrics


# save results function
def save_results(task, saved_metrics, output_foldername=OUTPUT_FOLDERNAME, seed=None):
    models_folder = os.path.join(os.getcwd(), 'models', OUTPUT_FOLDERNAME, 'torch_params')
    ParamsHandler.save_parameters(TORCH_PARAMS, models_folder)

    output_folder = os.path.join(os.getcwd(), 'results', 'LSTM', 'stateful', OUTPUT_FOLDERNAME)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    feature_set_names = {'PupilCalib': 'ET_Basic', 'CookieTheft': 'Eye', 'Reading': 'Eye_Reading', 'Memory': 'Audio'}
    metrics = ['acc', 'roc', 'fms', 'precision', 'recall', 'specificity']
    metric_names = {'acc': 'acc', 'roc': 'roc', 'fms': 'f1', 'precision': 'prec', 'recall': 'recall',
                    'specificity': 'spec'}

    seed_metrics = saved_metrics[0]
    dfs = []
    for metric in metrics:
        metric_name = metric_names[metric]
        metric_data = seed_metrics[metric_name]
        data = pd.DataFrame(metric_data, columns=['1'])
        data['metric'] = metric
        data['model'] = 'LSTM_median'
        data['method'] = 'end_to_end'
        dfs += [data]

    df = pd.concat(dfs, axis=0, ignore_index=True)
    print('saving result for specified seed, not iterated', seed)
    seed_path = os.path.join(output_folder, str(seed))
    if not os.path.exists(seed_path):
        os.mkdir(seed_path)

    df.to_csv(os.path.join(seed_path, 'results_new_features_{}.csv'.format(feature_set_names[task])), index=False)
    print('results saved for {}'.format(task))


def main():
    global TASKS
    TASKS = [sys.argv[1]]
    print('doing this for only task', TASKS, type(TASKS))
    global SEEDS
    pids = list(ttf.StudyID.unique())
    pids_to_remove = ['HH-076']  # HH-076 being removed because the task timings are off compared to the video length

    pids.remove(pids_to_remove[0])

    data = get_data(pids, TASKS)
    new_pids = remove_outliers(data, pids, TASKS, percentile_threshold=OUTLIER_THRESHOLD,
                               save_stats=False)  # percentile threshold 100 removes none

    task_meta_data = {task: {'PIDs': None, 'median sequence length': None} for task in TASKS}

    SEEDS = np.arange(int(sys.argv[2]), int(sys.argv[3]))
    if CLUSTER:
        SEEDS = [int(sys.argv[1])]

    for task in TASKS:
        print('processing {} task'.format(task))
        task_info = new_pids[new_pids.task == task]
        task_meta_data[task]['PIDs'] = task_pids = list(task_info.PID)
        task_meta_data[task]['median sequence length'] = task_median_length = task_info.len.median()
        task_meta_data[task]['max sequence length'] = task_max_length = task_info.len.max()

        task_data = get_data(task_pids, task)[task]

        truncated_data = Preprocess(FINAL_LENGTH, PAD_WHERE, TRUNCATE_WHERE, task_max_length).pad_and_truncate(
            task_data)

        print('SEEDS are ', SEEDS)
        for seed in SEEDS:
            cross_validate(task, truncated_data, seed)

    # average results across seeds and place it in a CSV file
    ResultsHandler.compile_results(os.path.join('LSTM', 'stateful'), OUTPUT_FOLDERNAME)


if __name__ == '__main__':
    main()
