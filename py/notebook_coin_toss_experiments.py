# +
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
# -

print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
import matplotlib
print(f"matplotlib: {matplotlib.__version__}")
import scipy
print(f"scipy: {scipy.__version__}")

# +
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

FIG_WIDTH, FIG_HEIGHT = 8, 6

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams["figure.figsize"] = FIG_WIDTH, FIG_HEIGHT
# plt.rcParams["hatch.linewidth"] = 0.2

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
# -

# # One Sequence Experiment

from utils_stats import sequence_to_sequential_pvalues
from utils_data import generate_biomial_sequence
from utils_viz import plot_sequence_experiment_nhst_combo_results

# +
seed = 13 #98 #31
success_rate = 0.5 #0.65 #0.5
n_samples = 1500

sequence = generate_biomial_sequence(success_rate=success_rate,
                                     n_samples=n_samples,
                                     seed=seed
                                    )
len(sequence)
# -

# ## Null Hypothesis Statistic Testing 
# (p-Value Stopping Criterion)
#
# As per [blog post](http://doingbayesiandataanalysis.blogspot.com/2013/11/optional-stopping-in-data-collection-p.html):
# > *For every new flip of the coin, stop and reject the null hypothesis, that θ=0.50, if p < .05 (two-tailed, conditionalizing on the current N), otherwise flip again.*

# +
success_rate_null=0.5
p_values = sequence_to_sequential_pvalues(sequence, success_rate_null=success_rate_null)

len(p_values)

# +
plot_sequence_experiment_nhst_combo_results(sequence, 
                                            success_rate, success_rate_null, p_values, 
                                            p_value_thresh=0.05, 
                                            xlabel="iteration")


# -

# # Many Sequence Experiements

# ## NHST

# +
# NHST stop criterion:
# For every new flip of the coin, stop and reject the null hypothesis, that θ=0.50, if p < .05 (two-tailed, conditionalizing on the current N), otherwise flip again.

# used for to show that success_rate = 0.5 can go higher than 50%
#experiments = 50 # 200
#n_samples = 30000

experiments = 1000 #200
n_samples = 350 #1500
success_rate = 0.65
success_rate_null = 0.5

alternative = 'two-sided' # 'greater'
p_value_thresh = 0.05
# -



# +
samples = generate_biomial_sequence(success_rate=success_rate,
                                     n_samples=n_samples,
                                    n_experiments = experiments,
                                     seed=seed
                                    )

samples.shape

# +
from scipy.stats import binom_test

experiement_stop_results = {'successes': [], 'trials': [], 'p_value': []}
iteration_stopping_on_or_prior = {iteration: 0 for iteration in range(1, n_samples + 1)}

for sample in samples:
    successes = 0
    this_iteration = 0
    for toss in sample:
        successes += toss
        this_iteration += 1
        
        p_value = binom_test(successes, n=this_iteration, p=success_rate_null, alternative=alternative)
        
        if p_value < p_value_thresh:
            for iteration in range(this_iteration, n_samples+1):
                iteration_stopping_on_or_prior[iteration] += 1
                
            break
    experiement_stop_results['successes'].append(successes)
    experiement_stop_results['trials'].append(this_iteration)
    experiement_stop_results['p_value'].append(p_value)

# +
msize = 5
xlabel = "bah"
title = "df"

sr_iteration_stopping_on_or_prior = pd.Series(iteration_stopping_on_or_prior)
sr_nhst_reject = sr_iteration_stopping_on_or_prior / experiments

plt.scatter(sr_nhst_reject.index, sr_nhst_reject + 0.01, alpha=0.7, s=msize, color="purple")
plt.scatter(sr_nhst_reject.index, 1. - sr_nhst_reject, alpha=0.7, s=msize, color="gray")

plt.xscale('log')
plt.xlabel(xlabel)
plt.title(title)

# +
df_stop_results = pd.DataFrame(experiement_stop_results)
df_stop_results.index.name = 'experiment_number'
df_stop_results['sample_success_rate'] = df_stop_results['successes'] * 1. / df_stop_results['trials']

df_plot = df_stop_results.copy()

df_plot = df_plot.query("p_value < @p_value_thresh")

#df_plot = df_stop_results.query(f"trials < {df_stop_results['trials'].describe()['25%'] / 2}").copy()

#print(len(df_plot))
print(df_plot['trials'].describe())
mean_success_rate = df_plot['sample_success_rate'].mean()

plt.hist(df_plot['sample_success_rate'], alpha=0.3, color="purple", bins=20)
print(mean_success_rate)
print(df_plot['sample_success_rate'].median())
#plt.scatter([mean_success_rate], [0], marker='^', s=400,color="red")
#plt.scatter([success_rate], [0], marker='^', s=400,color="black", alpha=0.1)

marker_style = dict(color='purple', linestyle=':', marker='^',
                    markersize=30, markerfacecoloralt='tab:red')
plt.plot([mean_success_rate], [0], **marker_style)

marker_style = dict(color='black', linestyle=':', marker='^',
                    markersize=30, markerfacecoloralt='tab:black')
plt.plot([success_rate], [0], fillstyle='none' , **marker_style)

plt.title(title)
pass
# -

plt.scatter(df_stop_results['trials'], df_stop_results['sample_success_rate'], color="purple", alpha=0.1)
plt.xlabel('no. of trials')
plt.ylabel('sample stop success rate')
plt.title(title)
plt.hlines(success_rate, 0, df_stop_results['trials'].max() + 1, color="gray", linestyle='--', alpha=0.3)

# +
df_plot = df_stop_results.copy()

df_plot['p_value'].hist(bins=20)
# -


