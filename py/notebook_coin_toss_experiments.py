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
success_rate_null = 0.5
p_values = sequence_to_sequential_pvalues(sequence, success_rate_null=success_rate_null)

len(p_values)

# +
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plot_sequence_experiment_nhst_combo_results(sequence, 
                                            success_rate, success_rate_null, p_values, 
                                            p_value_thresh=0.05, 
                                            xlabel="iteration")


# -

# ## HDI + ROPE

from utils_stats import successes_failures_to_hdi_ci_limits
from utils_viz import plot_sequence_experiment_hdi_rope_combo_results


def sequence_to_ci_details(sequence):
    ci_mins = []
    ci_maxs = []
    
    for idx, successes in enumerate(sequence.cumsum()):
        failures = (idx + 1) - successes
        
        if not failures:
            failures += 1
            successes += 1
        if not successes:
            failures += 1
            successes += 1
             
        ci_min, ci_max = successes_failures_to_hdi_ci_limits(successes, failures)

        ci_mins.append(ci_min)
        ci_maxs.append(ci_max)
    
    ci_mins = np.array(ci_mins)
    ci_maxs = np.array(ci_maxs)
    
    return ci_mins, ci_maxs


ci_mins, ci_maxs = sequence_to_ci_details(sequence)

# +
#success_rate_null = 0.5
#success_rate = 0.55
rope_width = 0.1 #success_rate * 0.1
# --------

rope_min = success_rate_null - rope_width / 2
rope_max = success_rate_null + rope_width / 2

# +
within_rope = (ci_mins >= rope_min) & (ci_maxs <= rope_max)

reject_lower = np.where(ci_maxs < rope_min, True, False)
reject_higher = np.where(ci_mins > rope_max, True, False)
accept_within = (ci_mins >= rope_min) & (ci_maxs <= rope_max)
reject_outside = reject_lower | reject_higher
inconclusive_hdi_plus_rope = ~(accept_within | reject_outside)

# +
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plot_sequence_experiment_hdi_rope_combo_results(sequence, success_rate, success_rate_null, 
                                                ci_mins, ci_maxs, accept_within, rope_min, rope_max, xlabel="trial no.", msize=5)

# +
from utils_viz import _get_sequence_idx

plt.plot(_get_sequence_idx(sequence), ci_maxs - ci_mins, color="purple")
plt.ylabel("HDI 95% CI width")
#plt.xlabel(xlabel)
#plt.xscale('log')

plt.tight_layout()
# -
# ## Precisions Is The Goal

from utils_viz import plot_sequence_experiment_pitg_combo_results

# +
rope_precision_fraction = 0.8
precision_goal = rope_width * rope_precision_fraction

print(f"{success_rate_null:0.2}: null")
print(f"{rope_min:0.2}: ROPE min")
print(f"{rope_max:0.2}: ROPE max")
print("-" * 20)
print(f"{precision_goal:0.2}: Precision Goal")
print("-" * 20)
print(f"{success_rate:0.2}: true")

# +
#ci_mins, ci_maxs = sequence_to_ci_details(sequence)
# -

precision_goal_achieved = np.where(ci_maxs - ci_mins <= precision_goal, True, False)

# +
setup_pitg = {'success_rate': success_rate,
         'success_rate_null': success_rate_null,
              'rope_min': rope_min,
              'rope_max': rope_max
        }

results_pitg = {
    'ci_mins': ci_mins,
    'ci_maxs': ci_maxs,
    'accept_within': accept_within,
    'reject_outside': reject_outside,
    'inconclusive_hdi_plus_rope': inconclusive_hdi_plus_rope,
    'precision_goal_achieved': precision_goal_achieved
}

plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plot_sequence_experiment_pitg_combo_results(sequence, setup_pitg, results_pitg, xlabel="test no.", msize=5, conservative=True)

# -

# # Many Sequence Experiments

from utils_stats import stop_decision_multiple_experiments
from utils_viz import plot_decision_rates_nhst

# +
# used for to show that success_rate = 0.5 can go higher than 50%
#experiments = 50 # 200
#n_samples = 30000

experiments = 100 #1000 #200
n_samples = 1500
success_rate = 0.65
success_rate_null = 0.5

# +
samples = generate_biomial_sequence(success_rate=success_rate,
                                     n_samples=n_samples,
                                    n_experiments = experiments,
                                     seed=seed
                                    )

samples.shape

# +
for isample, sample in enumerate(samples[::1]):
    None
    
isample
# -





# ## NHST

# +
# NHST stop criterion:
# For every new flip of the coin, stop and reject the null hypothesis, that θ=0.50, if p < .05 (two-tailed, conditionalizing on the current N), otherwise flip again.

alternative = 'two-sided' # 'greater'
p_value_thresh = 0.05 # alpha
# +
nhst_details = {'success_rate_null': success_rate_null,'p_value_thresh': p_value_thresh, 'alternative': alternative}
step = 1

nhst_experiment_stop_results, nhst_iteration_stopping_on_or_prior = \
stop_decision_multiple_experiments(samples, nhst_details)
# -

plot_decision_rates_nhst(experiments, nhst_iteration_stopping_on_or_prior)

plot_decision_rates_nhst(experiments, nhst_iteration_stopping_on_or_prior)

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
# ## HDI + ROPE


from utils_stats import stop_decision_multiple_experiments_bayesian
from utils_viz import plot_decision_rates

# +
bayes_details = {}
bayes_details['rope_min'] = rope_min
bayes_details['rope_max'] = rope_max

bayes_details

# +
hidrope_experiment_stop_results, df_decision_counts_hdirope = \
stop_decision_multiple_experiments_bayesian(samples, bayes_details=bayes_details)


print(df_decision_counts_hdirope.shape)
df_decision_counts_hdirope.head(4)

# +
plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plot_decision_rates(experiments, df_decision_counts_hdirope.rename(columns={'within':'accept'}))
# -

# ## Precision Is The Goal

from utils_stats import stop_decision_multiple_experiments_pitg

# +
precision_goal

bayes_details
# -

pitg_experiment_stop_results, df_decision_counts_pitg = \
stop_decision_multiple_experiments_pitg(samples, precision_goal, bayes_details=bayes_details)

print(df_decision_counts_pitg.sum().sum())
print(df_decision_counts_pitg.shape)
df_decision_counts_pitg.head(4)

plot_decision_rates(experiments, df_decision_counts_pitg)

plot_decision_rates(experiments, df_decision_counts_hdirope.rename(columns={'within':'accept'}))
