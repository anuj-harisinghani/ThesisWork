import os
import pandas as pd
import seaborn as sns
import math

from ParamsHandler import ParamsHandler

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

'''from scipy.stats import ttest_rel
# Calculate the t-test on TWO RELATED samples of scores, a and b.
#
# This is a two-sided test for the null hypothesis that 2 related or repeated samples have identical average (expected) values.
'''

params = ParamsHandler.load_parameters('settings')
mode = params['mode']
subsets = params['subsets']
clfs = params['classifiers']

stat_path = os.path.join('stats', mode)
result_path = os.path.join('results', mode)

if not os.path.exists(stat_path):
    os.mkdir(stat_path)

"""
LR vs Dummy ------------------------------------------------------------------------------------------------------------
"""
    # grouped bar plot
data = pd.read_csv(os.path.join(result_path, 'errors_across_seeds_unaveraged.csv'))
g = sns.catplot(data=data, kind='bar', x='strategy', y='mae', hue='clf', ci='sd')

    # tukey HSD
m_comp = pairwise_tukeyhsd(endog=data['mae'], groups=data['clf'] + " / " + data['strategy'], alpha=0.05)
print(m_comp)

tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns=m_comp._results_table.data[0])
ind = [i for i in tukey_data.index if tukey_data['group1'][i].split(' / ')[-1] == tukey_data['group2'][i].split(' / ')[-1]]
tukey = tukey_data.iloc[ind]
tukey.to_csv(os.path.join(stat_path, 'tukey_LR_vs_dummy.csv'))

    # cohen's d
# dummy_vs_LR_means = data.groupby(['clf', 'strategy']).mean().reset_index()
ttest_cols = ['strategy', 'clf1', 'clf2', 'cohens d']
pairwise_ttest_linear = pd.DataFrame()
for st in subsets:
    # data = dummy_vs_LR_means
    clf1 = 'Dummy'
    clf2 = 'LinearReg'  # control

    # for m in modes:
    # for m in modes:
    mode_data_1 = data[(data['clf'] == clf1) & (data['strategy'] == st)]['mae']
    mode_data_2 = data[(data['clf'] == clf2) & (data['strategy'] == st)]['mae']
    # cohen's effect sizes
    mean1 = mode_data_1.mean()
    mean2 = mode_data_2.mean()

    std1 = mode_data_1.std()
    std2 = mode_data_2.std()
    pooled_sd = math.sqrt((std1 ** 2 + std2 ** 2) / 2)

    # this will give negative values. In our case, lower is better for mean abs error
    # so instead of mean1 - mean2, we're gonna do mean2 - mean1 (not absolute)
    cohen_d = (mean1 - mean2) / pooled_sd

    pairwise_ttest_linear = pairwise_ttest_linear.append([[st, clf1, clf2, abs(cohen_d)]])
    # pairwise_ttest_linear = pairwise_ttest_linear.append(
    #     [[clf1, mode, m, stat, p_val, cohen_d, U1, p_val_mann, effect_mann, cliff_d]])

pairwise_ttest_linear.columns = ttest_cols
pairwise_ttest_linear.to_csv(os.path.join(stat_path, 'cohens_d_dummy_vs_LR.csv'))


"""
LR strategies against each other ---------------------------------------------------------------------------------------
"""
# tukey-HSD strategies against each other for LR
data_LR = data[data['clf'] == 'LinearReg']
m_comp = pairwise_tukeyhsd(endog=data_LR['mae'], groups=data_LR['clf'] + " / " + data_LR['strategy'], alpha=0.05)
print(m_comp)
tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns=m_comp._results_table.data[0])
tukey_data.to_csv(os.path.join(stat_path, 'tukey_LR_against_modes.csv'))

# Cohen's D
ttest_cols = ['clf', 'strat1', 'strat2', 'effect_t']
pairwise_ttest_linear = pd.DataFrame()

for clf1 in subsets:
    st = 'LinearReg'
    data = data_LR
    for clf2 in subsets:
        if clf1 == clf2:
            continue

        # for m in modes:
        # for m in modes:
        mode_data_1 = data[(data['strategy'] == clf1) & (data['clf'] == st)]['mae']
        mode_data_2 = data[(data['strategy'] == clf2) & (data['clf'] == st)]['mae']
        # cohen's effect sizes
        mean1 = mode_data_1.mean()
        mean2 = mode_data_2.mean()

        std1 = mode_data_1.std()
        std2 = mode_data_2.std()
        pooled_sd = math.sqrt((std1 ** 2 + std2 ** 2) / 2)

        # this will give negative values. In our case, lower is better for mean abs error
        # so instead of mean1 - mean2, we're gonna do mean2 - mean1 (not absolute)
        cohen_d = (mean2 - mean1) / pooled_sd

        pairwise_ttest_linear = pairwise_ttest_linear.append([[st, clf1, clf2, cohen_d]])
    # pairwise_ttest_linear = pairwise_ttest_linear.append(
    #     [[clf1, mode, m, stat, p_val, cohen_d, U1, p_val_mann, effect_mann, cliff_d]])

pairwise_ttest_linear.columns = ttest_cols
pairwise_ttest_linear.to_csv(os.path.join(stat_path, 'cohens_d_LR_against_strategies.csv'))

"""
old window results against new results ---------------------------------------------------------------------------------
"""

# tukey-HSD for old (window) results against no-window starting from pupil results
old_path = r"C:\Users\Anuj\PycharmProjects\ThesisWork\backup\window_results\pixel"
old_files = [i for i in os.listdir(old_path) if i.startswith('across_all_models_20')]
old_filepaths = [os.path.join(old_path, i) for i in old_files if i.endswith('.csv')]

old_data = []
for f in old_filepaths:
    file = pd.read_csv(f)
    if f.split('/')[-1].split('_')[-2] == 'avg':
        strat = 'avg'
    elif f.split('/')[-1].split('_')[-2] == 'all':
        strat = 'all'
    elif f.split('/')[-1].split('_')[-2] == 'both':
        strat = 'both_eyes'
    else:
        strat = f.split('/')[-1].split('_')[-1].split('.')[0]

    for val in file['LinearReg']:
        old_data.append([val, strat, 'old'])

old_d = pd.DataFrame(old_data, columns=['mae', 'strat', 'which'])

data_new = []
for i in range(len(data_LR)):
    row = data_LR.iloc[i]
    data_new.append([row['mae'], row['strategy'], 'new'])

data_n = pd.DataFrame(data_new, columns=['mae', 'strat', 'which'])
data_o_n = old_d.append(data_n, ignore_index=True)

# bar plot
g = sns.catplot(data=data_o_n, kind='bar', x='strat', y='mae', hue='which', ci='sd')

# Tukey HSD
m_comp = pairwise_tukeyhsd(endog=data_o_n['mae'], groups=data_o_n['which'] + " / " + data_o_n['strat'], alpha=0.05)
print(m_comp)

tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns=m_comp._results_table.data[0])
ind = [i for i in tukey_data.index if tukey_data['group1'][i].split(' / ')[-1] == tukey_data['group2'][i].split(' / ')[-1]]
tukey = tukey_data.iloc[ind]
tukey.to_csv(os.path.join(stat_path, 'tukey_LR_old_window_results_vs_new_results.csv'))

# Cohens D
ttest_cols = ['strat', 'new', 'old', 'effect_t']
pairwise_ttest_linear = pd.DataFrame()

for st in subsets:
    data = data_o_n
    clf1 = 'new'
    clf2 = 'old'
    mode_data_1 = data[(data['which'] == clf1) & (data['strat'] == st)]['mae']
    mode_data_2 = data[(data['which'] == clf2) & (data['strat'] == st)]['mae']
    # cohen's effect sizes
    n1 = len(mode_data_1)
    n2 = len(mode_data_2)

    mean1 = mode_data_1.mean()
    mean2 = mode_data_2.mean()

    std1 = mode_data_1.std()
    std2 = mode_data_2.std()
    # pooled_sd = math.sqrt((std1 ** 2 + std2 ** 2) / 2)
    pooled_sd = math.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / n1+n2-2)

    # this will give negative values. In our case, lower is better for mean abs error
    # so instead of mean1 - mean2, we're gonna do mean2 - mean1 (not absolute)
    cohen_d = (mean2 - mean1) / pooled_sd

    pairwise_ttest_linear = pairwise_ttest_linear.append([[st, clf1, clf2, cohen_d]])

pairwise_ttest_linear.columns = ttest_cols
pairwise_ttest_linear.to_csv(os.path.join(stat_path, 'cohens_d_LR_old_vs_new.csv'))


"""
================================================================================================================================
"""
# need independent t-test
result_files = [i for i in os.listdir(result_path) if i.endswith('csv')]
result_file_paths = [os.path.join(result_path, i) for i in result_files]

rfp20 = [i for i in result_file_paths if '20' in i]  # result_file_path for those files with '20' time windows

subsets = ['left', 'right', 'both_eyes', 'avg_vector', 'all_vector']
# classifiers = ['Dummy', 'LinearReg', 'Ridge', 'Lasso']
classifiers = ['Dummy', 'LinearReg']  # only LR
# classifiers = ['LinearReg', 'Ridge', 'Lasso']  # only linears

data = pd.DataFrame()
data_mean = pd.DataFrame()
data2 = pd.DataFrame()
for sub in subsets:
    file = pd.read_csv([i for i in rfp20 if sub in i][0])
    file = file.drop(file.columns[0], axis=1)

    file = file[classifiers]
    # file['sub'] = sub
    file2 = file.melt(value_name='mean_abs_error')
    file2 = file2.rename(columns={'variable': 'clf'})
    file2['mode'] = mode
    data2 = data2.append(file2)

    # data = data.append(file)
    #
    #     mean_file = file.mean()
    #     x = classifiers  # mean_file.index
    #     # y = [file[i] for i in classifiers]
    #     y = [mean_file[i] for i in classifiers]  # mean_file.values
    #     # sns.barplot(x, y)
    #     m = [mode]*len(x)
    #     data_mean = data_mean.append([[m[i], x[i], y[i]] for i in range(len(x))])

# data_mean.columns = ['mode', 'clf', 'mean_abs_error']

#
# g = sns.catplot(data=data_mean, kind='bar', x='mode', y='mean_abs_error', hue='clf')
g = sns.catplot(data=data2, kind='bar', x='mode', y='mean_abs_error', hue='clf', ci='sd')
# only_linear = sns.catplot(data=data[data['clf'] == 'LinearReg'], kind='bar', x='mode', y='mean_abs_error', hue='clf')


# Two way ANOVA on data - has clfs and mode
formula = 'mean_abs_error ~ clf * mode'
anova = ols(formula, data).fit()
aov_table = anova_lm(anova, typ=1)

# tukey
data = data2[data2['clf']=='LinearReg']
m_comp = pairwise_tukeyhsd(endog=data['mean_abs_error'], groups=data['clf'] + " / " + data['mode'], alpha=0.05)
print(m_comp)

tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns=m_comp._results_table.data[0])
tukey_data.to_csv(os.path.join(stat_path, 'tukey_LR_against_modes.csv'))


data = data2
m_comp = pairwise_tukeyhsd(endog=data['mean_abs_error'], groups=data['clf'] + " / " + data['mode'], alpha=0.05)
print(m_comp)

tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns=m_comp._results_table.data[0])
ind = [i for i in tukey_data.index if tukey_data['group1'][i].split(' / ')[-1] == tukey_data['group2'][i].split(' / ')[-1]]
tukey = tukey_data.iloc[ind]
tukey.to_csv(os.path.join(stat_path, 'tukey_LR_vs_dummy.csv'))



# pairwise t-test
# metric = 'mean_abs_error'
# pairwise_ttest = pd.DataFrame()
# ttest_cols = ['clf1', 'clf2', 'statistic', 'p_val']
# for clf1 in classifiers:
#     if clf1 == 'Dummy':
#         continue
#     # clf1 = 'LinearReg'
#     clf2 = 'Dummy'
#     stat, p_val = ttest_ind(data[data['clf'] == clf1][metric], data[data['clf'] == clf2][metric])
#     pairwise_ttest = pairwise_ttest.append([[clf1, clf2, stat, p_val]])
#
# pairwise_ttest.columns = ttest_cols
# pairwise_ttest.to_csv(os.path.join(stat_path, 'pairwise_t_test_linear_v_dummy.csv'))


# clf pairwise t_test
metric = 'mean_abs_error'
ttest_cols = ['mode', 'clf1', 'clf2', 'statistic_t', 'p_val_t', 'effect_t', 'U1_clf1', 'p_val_mann', 'effect_size_mann', 'new']
pairwise_ttest_linear = pd.DataFrame()
clfs_no_dummy = classifiers
data = data2  # ['LinearReg', 'Ridge', 'Lasso']
for mode in subsets:
    # for clf1 in clfs_no_dummy:
    #     for clf2 in clfs_no_dummy:
    #         if clf1 == clf2:
    #             continue
    clf1 = 'Dummy'
    clf2 = 'LinearReg'  # control

    # for m in modes:
    # for m in modes:
    mode_data_1 = data[(data['clf'] == clf1) & (data['mode'] == mode)][metric]
    mode_data_2 = data[(data['clf'] == clf2) & (data['mode'] == mode)][metric]
    stat, p_val = ttest_ind(mode_data_1, mode_data_2)
    U1, p_val_mann = mannwhitneyu(mode_data_1, mode_data_2)
    n1, n2 = len(mode_data_1), len(mode_data_2)
    U2 = (n1 * n2) - U1
    z_mann = (U1 + 0.5) - (((U1+U2)/2) / math.sqrt((n1*n2*(n1+n2+1))/12))
    effect_mann = U1/(n1*n2)
    cliff_d = ((2*U1)/(n1*n2)) - 1
    # effect_mann = stat_mann / math.sqrt(len(mode_data_1) * len(mode_data_2))

    # cohen's effect sizes
    mean1 = mode_data_1.mean()
    mean2 = mode_data_2.mean()

    std1 = mode_data_1.std()
    std2 = mode_data_2.std()
    pooled_sd = math.sqrt((std1 ** 2 + std2 ** 2) / 2)

    # this will give negative values. In our case, lower is better for mean abs error
    # so instead of mean1 - mean2, we're gonna do mean2 - mean1 (not absolute)
    cohen_d = (mean2 - mean1) / pooled_sd

    pairwise_ttest_linear = pairwise_ttest_linear.append([[mode, clf1, clf2, stat, p_val, cohen_d, U1, p_val_mann, effect_mann, cliff_d]])
    # pairwise_ttest_linear = pairwise_ttest_linear.append(
    #     [[clf1, mode, m, stat, p_val, cohen_d, U1, p_val_mann, effect_mann, cliff_d]])

pairwise_ttest_linear.columns = ttest_cols
# pairwise_ttest_linear.to_csv(os.path.join(stat_path, 'pairwise_t_test_linear_reg_vs_dummy.csv'))
pairwise_ttest_linear.to_csv(os.path.join(stat_path, 'pairwise_t_test_linear_reg_avg_vector.csv'))


# ANOVA on linear models with models as factors, mean_absolute_error as dependent variable
# linear_data = data[data['clf'] != 'Dummy']
# formula = 'mean_abs_error ~ clf * mode'
# anova_linear = ols(formula, linear_data).fit()
# aov_table_linear = anova_lm(anova_linear, typ=1)

linear_data = data[data['clf'] != 'Dummy']
formula = 'mean_abs_error ~ mode'
formula = 'mean_abs_error ~ clf'
anova_linear = ols(formula, linear_data).fit()
aov_table_linear = anova_lm(anova_linear, typ=1)

ss_total = aov_table_linear['sum_sq'].sum()
ss_between = aov_table_linear['sum_sq']['clf']

eta_sq = ss_between/ss_total
cohen_f_anova = math.sqrt(eta_sq/(1-eta_sq))
aov_table_linear['eta_sq'] = eta_sq
aov_table_linear['cohens_f'] = cohen_f_anova

pd.DataFrame(aov_table_linear).to_csv(os.path.join(stat_path, 'anova_1_way_across_clfs.csv'))
