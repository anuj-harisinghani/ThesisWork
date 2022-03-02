import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


result_path = os.path.join('results_new', 'pixel')
cwm_csv = '{}_{}_{}.csv'
cwm_png = '{}_{}_{}.png'


def average_results(clf, window_iter, mode):
    clf_path = os.path.join('results', clf)
    seeds = [i for i in os.listdir(clf_path) if len(i.split('.')) < 2]
    seed_paths = [os.path.join(clf_path, s) for s in seeds]
    filename = cwm_csv.format(clf, window_iter, mode)
    seed_mode_paths = [os.path.join(seed_paths[s], filename) for s in range(len(seeds))]

    csv_files = np.array([np.array(pd.read_csv(seed_mode_paths[s]))[:, 1] for s in range(len(seeds))])
    average_vals = np.mean(csv_files, axis=0)
    win_range = list(range(1, window_iter))

    plt.clf()
    plt.title('{} {} {}'.format(clf, window_iter, mode))
    plt.xlabel('window size')
    plt.ylabel('mean error')
    plt.plot(win_range, average_vals)
    print('saving average plot {}_{}_{}.png'.format(clf, window_iter, mode))
    plt.savefig(os.path.join(clf_path, cwm_png.format(clf, window_iter, mode)))
    plt.close()

    error_filename = os.path.join(clf_path, '{}_{}_{}.csv'.format(clf, window_iter, mode))
    pd.DataFrame(average_vals, columns=['mean absolute error'], index=win_range).to_csv(error_filename)


def plot_all_averaged_models(window_iter, mode, clfs):
    # clfs = [i for i in os.listdir(result_path) if len(i.split('.')) < 2]
    # clfs = [i for i in os.listdir(result_path) if len(i.split('.')) < 2 and
    #       len(os.listdir(os.path.join(result_path, i))) > 10]

    clfs_paths = [os.path.join(result_path, c) for c in clfs]

    filenames = [cwm_csv.format(c, window_iter, mode) for c in clfs]
    filepaths = [os.path.join(clfs_paths[c], filenames[c]) for c in range(len(filenames))]
    win_range = list(range(1, window_iter))

    csv_files = np.array([list(pd.read_csv(c)['mean absolute error']) for c in filepaths]).T
    main_file = pd.DataFrame(csv_files, columns=clfs, index=win_range)
    main_file.to_csv(os.path.join(result_path, 'across_all_models_{}_{}.csv'.format(window_iter, mode)))

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.title('across all models {} {}'.format(window_iter, mode))
    plt.xlabel('window size')
    plt.ylabel('mean error')

    for col in main_file:
        plt.plot(win_range, main_file[col], label=col)
    plt.legend()
    print('saving plot {}_{}.png'.format(window_iter, mode))
    plt.savefig(os.path.join(result_path, 'across_all_models_{}_{}.png'.format(window_iter, mode)))
    plt.close()


def average_errors(mode):
    result_path = os.path.join('results', mode)
    folders = [i for i in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, i))]

    df = pd.DataFrame(columns=['clf', 'strategy', 'mae'])
    for seed in folders:
        fold_file = pd.read_csv(os.path.join(result_path, seed, 'avg_errors.csv')).drop('Unnamed: 0', axis=1)
        df = df.append(fold_file, ignore_index=True)

    avg_across_seeds = df.groupby(['clf', 'strategy']).mean().reset_index()
    avg_across_seeds.to_csv(os.path.join(result_path, 'errors_across_seeds.csv'))



