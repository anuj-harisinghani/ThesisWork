import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def average_results(clf, window_iter, mode):
    clf_path = os.path.join('results', clf)
    seeds = [i for i in os.listdir(clf_path) if len(i.split('.')) < 2]
    seed_paths = [os.path.join(clf_path, s) for s in seeds]
    filename = '{}_{}_{}.csv'.format(clf, window_iter, mode)
    seed_mode_paths = [os.path.join(seed_paths[s], filename) for s in range(len(seeds))]

    csv_files = np.array([np.array(pd.read_csv(seed_mode_paths[s]))[:, 1] for s in range(len(seeds))])
    average_vals = np.mean(csv_files, axis=0)
    win_range = list(range(1, window_iter))

    plt.clf()
    plt.title('{} {} {}'.format(clf, window_iter, mode))
    plt.xlabel('window size')
    plt.ylabel('mean error')
    plt.plot(win_range, average_vals)
    print('saving plot {}_{}_{}.png'.format(clf, window_iter, mode))
    plt.savefig(os.path.join(clf_path, '{}_{}_{}.png'.format(clf, window_iter, mode)))
    plt.close()

    error_filename = os.path.join(clf_path, '{}_{}_{}.csv'.format(clf, window_iter, mode))
    pd.DataFrame(average_vals, columns=['mean absolute error'], index=win_range).to_csv(error_filename)
