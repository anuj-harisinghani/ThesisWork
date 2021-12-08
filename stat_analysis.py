import os
import pandas as pd
import seaborn as sns

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind

'''from scipy.stats import ttest_rel
# Calculate the t-test on TWO RELATED samples of scores, a and b.
#
# This is a two-sided test for the null hypothesis that 2 related or repeated samples have identical average (expected) values.
'''

# need independent t-test


stat_path = os.path.join('stats')
result_path = os.path.join('results')
result_files = [i for i in os.listdir(result_path) if i.endswith('csv')]
result_file_paths = [os.path.join(result_path, i) for i in result_files]

rfp20 = [i for i in result_file_paths if '20' in i]

modes = ['left', 'right', 'both_eyes', 'avg_vector', 'all', 'all_vector']
classifiers = ['Dummy', 'LinearReg', 'Ridge', 'Lasso']

data = pd.DataFrame()
for mode in modes:
    file = pd.read_csv([i for i in rfp20 if mode in i][0])
    file = file.drop(file.columns[0], axis=1)

    mean_file = file.mean()

    x = classifiers  # mean_file.index
    y = [mean_file[i] for i in classifiers]  # mean_file.values
    # sns.barplot(x, y)

    m = [mode]*len(x)
    data = data.append([[m[i], x[i], y[i]] for i in range(len(x))])

data.columns = ['mode', 'clf', 'mean_abs_error']

#
# g = sns.catplot(data=data, kind='bar', x='mode', y='mean_abs_error', hue='clf')
# only_linear = sns.catplot(data=data[data['clf'] == 'LinearReg'], kind='bar', x='mode', y='mean_abs_error', hue='clf')


# Two way ANOVA on data - has clfs and mode
formula = 'mean_abs_error ~ clf * mode'
anova = ols(formula, data).fit()
aov_table = anova_lm(anova, typ=1)

# tukey
m_comp = pairwise_tukeyhsd(endog=data['mean_abs_error'], groups=data['clf'] + " / " + data['mode'], alpha=0.05)
print(m_comp)

# pairwise t-test
metric = 'mean_abs_error'
pairwise_ttest = pd.DataFrame()
ttest_cols = ['clf1', 'clf2', 'statistic', 'p_val']
for clf1 in classifiers:
    if clf1 == 'Dummy':
        continue
    # clf1 = 'LinearReg'
    clf2 = 'Dummy'
    stat, p_val = ttest_ind(data[data['clf'] == clf1][metric], data[data['clf'] == clf2][metric])
    pairwise_ttest = pairwise_ttest.append([[clf1, clf2, stat, p_val]])

pairwise_ttest.columns = ttest_cols
pairwise_ttest.to_csv(os.path.join(stat_path, 'pairwise_t_test_linear_v_dummy.csv'))


# clf pairwise t_test
pairwise_ttest_linear = pd.DataFrame()
clfs_no_dummy = ['LinearReg', 'Ridge', 'Lasso']
for clf1 in clfs_no_dummy:
    for clf2 in clfs_no_dummy:
        if clf1 == clf2:
            continue
        stat, p_val = ttest_ind(data[data['clf'] == clf1][metric], data[data['clf'] == clf2][metric])
        pairwise_ttest_linear = pairwise_ttest_linear.append([[clf1, clf2, stat, p_val]])

pairwise_ttest_linear.columns = ttest_cols
pairwise_ttest_linear.to_csv(os.path.join(stat_path, 'pairwise_t_test_linear_models.csv'))


# ANOVA on linear models with models as factors, mean_absolute_error as dependent variable
# linear_data = data[data['clf'] != 'Dummy']
# formula = 'mean_abs_error ~ clf * mode'
# anova_linear = ols(formula, linear_data).fit()
# aov_table_linear = anova_lm(anova_linear, typ=1)

linear_data = data[data['clf'] != 'Dummy']
formula = 'mean_abs_error ~ mode * clf'
anova_linear = ols(formula, linear_data).fit()
aov_table_linear = anova_lm(anova_linear, typ=1)
pd.DataFrame(aov_table_linear).to_csv(os.path.join(stat_path, 'anova_2way_linear_models.csv'))
