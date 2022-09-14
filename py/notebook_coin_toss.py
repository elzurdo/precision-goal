# -*- coding: utf-8 -*-
# +
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# https://en.wikipedia.org/wiki/Binomial_test
from scipy.stats import binom_test # binomtest (in later versions ...)
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

# # Null Hypothesis Statistic Testing 
# (p-Value Stopping Criterion)
#
# As per [blog post](http://doingbayesiandataanalysis.blogspot.com/2013/11/optional-stopping-in-data-collection-p.html):
# > *For every new flip of the coin, stop and reject the null hypothesis, that θ=0.50, if p < .05 (two-tailed, conditionalizing on the current N), otherwise flip again.*

def sequence_to_pvalues(sequence, success_rate_null=0.5):
    p_values = []
    
    for idx, successes in enumerate(sequence.cumsum()):
        p_value = binom_test(successes, n=idx + 1, p=success_rate_null, alternative='two-sided') # alternative {‘two-sided’, ‘greater’, ‘less’},
        p_values.append(p_value)
    
    p_values = np.array(p_values)
    
    return p_values


# +
seed = 13 #98 #31

success_rate = 0.65 #0.65 #0.5
n_samples = 1500

np.random.seed(seed)

sequence = np.random.binomial(1, success_rate, n_samples)

success_rate_null=0.5
p_values = sequence_to_pvalues(sequence, success_rate_null=success_rate_null)


# +
p_value_thresh = 0.001 #0.001 #0.01 #0.05 # 0.0001 #0.05
# ----------
xlabel = "iteration"
title = f"true success rate = {success_rate:0.2f}"
msize = 5
dsuccess_rate_plot = 0.07

sequence_idx = np.arange(n_samples)+ 1
sequence_average = sequence.cumsum() / sequence_idx

plt.subplot(2, 1, 1)

#errorbar = 1.96 * np.sqrt((sequence_average * (1. - sequence_average)) / sequence_idx )
#plt.errorbar(sequence_idx, sequence_average, yerr=errorbar, color="gray", alpha=0.05)
plt.scatter(sequence_idx[sequence == 1], sequence_average[sequence == 1], color = "green", alpha=0.7, s=msize)
plt.scatter(sequence_idx[sequence == 0], sequence_average[sequence == 0], color = "red", alpha=0.7, s=msize)
plt.hlines(success_rate, sequence_idx[0], sequence_idx[-1], color="gray", linestyle='--', alpha=0.3)
plt.xlabel(xlabel)
plt.ylabel("cumsum average")
plt.ylim(success_rate - dsuccess_rate_plot, success_rate + dsuccess_rate_plot)
plt.title(title)


plt.subplot(2, 1, 2)
plt.hlines(p_value_thresh, sequence_idx[0], sequence_idx[-1], color="gray", linestyle='--', alpha=0.3)
plt.scatter(sequence_idx[p_values >= p_value_thresh], p_values[p_values >= p_value_thresh], color = "gray", alpha=0.7, s=msize)
plt.scatter(sequence_idx[p_values < p_value_thresh], p_values[p_values < p_value_thresh], color = "blue", marker='x', s=msize * 10)
plt.xlabel(xlabel)
plt.ylabel("p-value")
plt.ylim(-0.1, 0.5)


plt.tight_layout()
print(len(sequence_idx[p_values < p_value_thresh]))

# +
# NHST stop criterion:
# For every new flip of the coin, stop and reject the null hypothesis, that θ=0.50, if p < .05 (two-tailed, conditionalizing on the current N), otherwise flip again.

# used for to show that success_rate = 0.5 can go higher than 50%
#experiments = 50 # 200
#n_samples = 30000

experiments = 1000 #200
n_samples = 350 #1500

alternative = 'two-sided' # 'greater'
# ----------

experiement_stop_results = {'successes': [], 'trials': [], 'p_value': []}
iteration_stopping_on_or_prior = {iteration: 0 for iteration in range(1, n_samples + 1)}

np.random.seed(seed)
samples = np.random.binomial(1, success_rate, [experiments, n_samples])

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
sr_iteration_stopping_on_or_prior = pd.Series(iteration_stopping_on_or_prior)
sr_nhst_reject = sr_iteration_stopping_on_or_prior / experiments

plt.scatter(sr_nhst_reject.index, sr_nhst_reject + 0.01, alpha=0.7, s=msize, color="purple")
plt.scatter(sr_nhst_reject.index, 1. - sr_nhst_reject, alpha=0.7, s=msize, color="gray")

plt.xscale('log')
plt.xlabel(xlabel)
plt.title(title)

# +
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

# # Bayesian HDI with ROPE 
#
# As per [blog post](http://doingbayesiandataanalysis.blogspot.com/2013/11/optional-stopping-in-data-collection-p.html):
#
# > *For every flip of the coin, compute the 95% HDI on θ. If the HDI is completely contained in a ROPE from 0.45 to 0.55, stop and accept the null. If the HDI falls completely outside the ROPE stop and reject the null. Otherwise flip again.*

# +
from utils_stats import (
    hdi_ci_limits,
    successes_failures_to_hdi_ci_limits
)

from utils_viz import (
    plot_success_rates
)


# -

def successes_failures_to_hdi_within_rope(successes, failures, rope_min, rope_max):
    ci_min, ci_max = successes_failures_to_hdi_ci_limits(successes, failures)
    
    if (ci_min >= rope_min) & (ci_max <= rope_max):
        return True, ci_min, ci_max
    
    return False, ci_min, ci_max


# +
def sequence_to_hdi_within_rope(sequence, rope_min, rope_max):
    within_rope = []
    ci_mins = []
    ci_maxs = []
    
    for idx, successes in enumerate(sequence.cumsum()):
        failures = (idx + 1) - successes
        
        if not successes:
            successes += 0.01
            
        if not failures:
            failures += 0.01
        
        this_within_rope, ci_min, ci_max = successes_failures_to_hdi_within_rope(successes, failures, rope_min, rope_max)
        #print(idx, successes, failures, ci_min, ci_max)
        within_rope.append(this_within_rope)
        ci_mins.append(ci_min)
        ci_maxs.append(ci_max)
    
    within_rope = np.array(within_rope)
    ci_mins = np.array(ci_mins)
    ci_maxs = np.array(ci_maxs)
    
    return within_rope, ci_mins, ci_maxs

#sequence_to_hdi_within_rope(sequence[:4], rope_min, rope_max)


# +
#successes_failures_to_hdi_within_rope(successes, failures, rope_min, rope_max)

# +
success_rate_null = 0.5
success_rate = 0.55
dsuccess_rate = 0.05 #success_rate * 0.1
# --------

rope_min = success_rate_null - dsuccess_rate
rope_max = success_rate_null + dsuccess_rate

print(f"{success_rate_null:0.2}: null")
print(f"{rope_min:0.2}: ROPE min")
print(f"{rope_max:0.2}: ROPE max")
print("-" * 20)
print(f"{success_rate:0.2}: true")


# +
seed = 317 #13 #98 #31

n_samples = 1500

np.random.seed(seed)
sequence = np.random.binomial(1, success_rate, n_samples)
# -

within_rope, ci_mins, ci_maxs = sequence_to_hdi_within_rope(sequence, rope_min, rope_max)
hdi_widths = ci_maxs - ci_mins

# +
title = f"true success rate = {success_rate:0.2f}"

sequence_idx = np.arange(n_samples)+ 1
sequence_average = sequence.cumsum() / sequence_idx

plt.subplot(2, 1, 1)

lower_uncertainty = sequence_average - ci_mins
upper_uncertainty = ci_maxs - sequence_average

plt.errorbar(sequence_idx[within_rope], sequence_average[within_rope], yerr=(upper_uncertainty[within_rope], lower_uncertainty[within_rope]), color="green", alpha=0.3)
reject_higher = ci_mins > rope_max
plt.errorbar(sequence_idx[reject_higher], sequence_average[reject_higher], yerr=(upper_uncertainty[reject_higher], lower_uncertainty[reject_higher]), color="red", alpha=0.3)
reject_lower = ci_maxs < rope_min
plt.errorbar(sequence_idx[reject_lower], sequence_average[reject_lower], yerr=(upper_uncertainty[reject_lower], lower_uncertainty[reject_lower]), color="orange", alpha=0.3)
inconclusive = ~(within_rope + reject_higher + reject_lower)
plt.errorbar(sequence_idx[inconclusive], sequence_average[inconclusive], yerr=(upper_uncertainty[inconclusive], lower_uncertainty[inconclusive]), color="gray", alpha=0.3)

plt.hlines(success_rate, sequence_idx[0], sequence_idx[-1], color="gray", linestyle='--', alpha=0.3)
plt.ylabel("cumulative sum\nHDI 95% CI")
plt.ylim(success_rate - 0.3, success_rate + 0.3)
plt.title(title)
#plt.xscale('log')


plt.subplot(2, 1, 2)
plt.plot(sequence_idx, ci_maxs - ci_mins, color="purple")
plt.ylabel("HDI 95% CI width")
plt.xlabel(xlabel)
#plt.xscale('log')

plt.tight_layout()

# +
experiments = 300 #200
n_samples = 2500 #1000 #1500
# ----------

experiment_stop_results_hdi = {'successes': [], 'trials': [], 'within_rope': [], 'ci_min': [], 'ci_max': []}
iteration_stopping_on_or_prior_hdi_within = {iteration: 0 for iteration in range(1, n_samples + 1)}
iteration_stopping_on_or_prior_hdi_below = iteration_stopping_on_or_prior_hdi_within.copy()
iteration_stopping_on_or_prior_hdi_above = iteration_stopping_on_or_prior_hdi_within.copy()

np.random.seed(seed)
samples = np.random.binomial(1, success_rate, [experiments, n_samples])

for sample in samples:
    successes = 0
    this_iteration = 0
    for toss in sample:
        successes += toss
        this_iteration += 1
        
        failures = this_iteration - successes
     
        if this_iteration > 5: # cannot rely on below 5
            ci_min, ci_max = successes_failures_to_hdi_ci_limits(successes, failures)
            this_within_rope = (ci_min >= rope_min) & (ci_max <= rope_max)

            if this_within_rope:
                for iteration in range(this_iteration, n_samples+1):
                    iteration_stopping_on_or_prior_hdi_within[iteration] += 1
                break

            if (ci_max < rope_min): 
                for iteration in range(this_iteration, n_samples+1):
                    iteration_stopping_on_or_prior_hdi_below[iteration] += 1
                break

            if (ci_min > rope_max):
                for iteration in range(this_iteration, n_samples+1):
                    iteration_stopping_on_or_prior_hdi_above[iteration] += 1
                break
            
            
    experiment_stop_results_hdi['successes'].append(successes)
    experiment_stop_results_hdi['trials'].append(this_iteration)
    experiment_stop_results_hdi['within_rope'].append(this_within_rope)
    experiment_stop_results_hdi['ci_min'].append(ci_min)
    experiment_stop_results_hdi['ci_max'].append(ci_max)
# -

rope_min, rope_max

# +
#pd.Series(iteration_stopping_on_or_prior_hdi_within).value_counts()

# +
df_hdi_counts = pd.DataFrame({'within': iteration_stopping_on_or_prior_hdi_within,
               'below': iteration_stopping_on_or_prior_hdi_below, 
              'above': iteration_stopping_on_or_prior_hdi_above,
              })

df_hdi_counts.index.name = "iteration_number"
df_hdi_counts['reject'] = df_hdi_counts['above'] + df_hdi_counts['below']
df_hdi_counts['inconclusive'] = experiments - df_hdi_counts['within'] - df_hdi_counts['reject']

print(df_hdi_counts.shape)
df_hdi_counts.head(4)
# -

df_hdi_counts.tail(4)

# +
#(df_hdi_counts['within'] / experiments).describe()

# +
plt.plot(df_hdi_counts.index, df_hdi_counts['within'] / experiments, color="green")
plt.plot(df_hdi_counts.index, df_hdi_counts['reject'] / experiments, color="red")
plt.plot(df_hdi_counts.index, df_hdi_counts['inconclusive'] / experiments, color="gray")

#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(xlabel)
plt.title(title)

# +
plt.plot(df_hdi_counts.index, df_hdi_counts['within'] / experiments, color="green")
plt.plot(df_hdi_counts.index, df_hdi_counts['reject'] / experiments, color="purple")
plt.plot(df_hdi_counts.index, df_hdi_counts['inconclusive'] / experiments, color="gray")

plt.xscale('log')
plt.xlabel(xlabel)
plt.title(title)

# +
df_stop_results_hdi = pd.DataFrame(experiment_stop_results_hdi)
#df_stop_results_hdi['success_rate'] = df_stop_results_hdi['successes'] / df_stop_results_hdi['trials']

df_stop_results_hdi.index.name = 'experiment_number'
df_stop_results_hdi['sample_success_rate'] = df_stop_results_hdi['successes'] * 1. / df_stop_results_hdi['trials']


print(df_stop_results_hdi.shape)
df_stop_results_hdi.head(4)
# -

df_stop_results_hdi.tail()

df_stop_results_hdi.describe()

df_stop_results_hdi['outside'] = False
df_stop_results_hdi.loc[df_stop_results_hdi.query('(ci_min > @rope_max) | (ci_max > @rope_min)').index, 'outside'] = True

# +
df_plot = df_stop_results_hdi.copy()
#df_plot = df_stop_results.query(f"trials < {df_stop_results['trials'].describe()['25%'] / 2}").copy()
df_plot = df_plot.query('within_rope | outside')

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
plt.xlabel("sample success rate at decision")
plt.ylabel("frequency")

pass
# -



# # Precision Is The Goal
#
# > *For every flip of the coin, compute the 95% HDI on θ. If its width is less than 0.08 (.8*width of ROPE) then stop, otherwise flip again. Once stopped, check whether null can be  accepted or rejected according to HDI with ROPE criteria.*

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
        # print(successes, failures, ci_min, ci_max)

        ci_mins.append(ci_min)
        ci_maxs.append(ci_max)
    
    ci_mins = np.array(ci_mins)
    ci_maxs = np.array(ci_maxs)
    
    return ci_mins, ci_maxs


# +
success_rate_null = 0.5
dsuccess_rate = 0.05 #success_rate * 0.1
rope_precision_fraction = 0.8


success_rate = 0.55
# --------

rope_min = success_rate_null - dsuccess_rate
rope_max = success_rate_null + dsuccess_rate

precision_goal = (2 * dsuccess_rate) * rope_precision_fraction

print(f"{success_rate_null:0.2}: null")
print(f"{rope_min:0.2}: ROPE min")
print(f"{rope_max:0.2}: ROPE max")
print("-" * 20)
print(f"{precision_goal:0.2}: Precision Goal")
print("-" * 20)
print(f"{success_rate:0.2}: true")


# +
seed =  13 #317 #98 #31

n_samples = 1500

np.random.seed(seed)
sequence = np.random.binomial(1, success_rate, n_samples)
# -

ci_mins, ci_maxs = sequence_to_ci_details(sequence)

# +
reject_lower = np.where(ci_maxs < rope_min, True, False)
reject_higher = np.where(ci_mins > rope_max, True, False)
accept_within = (ci_mins >= rope_min) & (ci_maxs <= rope_max)
reject_outside = reject_lower | reject_higher
inconclusive_hdi_plus_rope = ~(accept_within | reject_outside)

precision_goal_achieved = np.where(ci_maxs - ci_mins <= precision_goal, True, False)

# +
title = f"true success rate = {success_rate:0.2f}"

sequence_idx = np.arange(n_samples)+ 1
sequence_average = sequence.cumsum() / sequence_idx

plt.subplot(2, 1, 1)

lower_uncertainty = sequence_average - ci_mins
upper_uncertainty = ci_maxs - sequence_average

plt.errorbar(sequence_idx[precision_goal_achieved], sequence_average[precision_goal_achieved], yerr=(upper_uncertainty[precision_goal_achieved], lower_uncertainty[precision_goal_achieved]), color="purple", alpha=0.05)
plt.errorbar(sequence_idx[~precision_goal_achieved], sequence_average[~precision_goal_achieved], yerr=(upper_uncertainty[~precision_goal_achieved], lower_uncertainty[~precision_goal_achieved]), color="gray", alpha=0.3)

bool_ = precision_goal_achieved & accept_within
plt.errorbar(sequence_idx[bool_], sequence_average[bool_], yerr=(upper_uncertainty[bool_], lower_uncertainty[bool_]), color="green", alpha=0.3)

bool_ = precision_goal_achieved & reject_outside
plt.errorbar(sequence_idx[bool_], sequence_average[bool_], yerr=(upper_uncertainty[bool_], lower_uncertainty[bool_]), color="red", alpha=0.3)


plt.hlines(success_rate, sequence_idx[0], sequence_idx[-1], color="gray", linestyle='--', alpha=0.3)
plt.ylabel("cumulative sum\nHDI 95% CI")
plt.ylim(success_rate - 0.3, success_rate + 0.3)
plt.title(title)

plt.subplot(2, 1, 2)

bool_ = reject_outside
plt.errorbar(sequence_idx[reject_outside], sequence_average[reject_outside], yerr=(upper_uncertainty[reject_outside], lower_uncertainty[reject_outside]), color="red", alpha=0.3)
plt.errorbar(sequence_idx[accept_within], sequence_average[accept_within], yerr=(upper_uncertainty[accept_within], lower_uncertainty[accept_within]), color="green", alpha=0.3)

plt.errorbar(sequence_idx[inconclusive_hdi_plus_rope], sequence_average[inconclusive_hdi_plus_rope], yerr=(upper_uncertainty[inconclusive_hdi_plus_rope], lower_uncertainty[inconclusive_hdi_plus_rope]), color="gray", alpha=0.3)


plt.ylabel("cumulative sum\nHDI 95% CI")
plt.xlabel(xlabel)
plt.ylim(success_rate - 0.3, success_rate + 0.3)
plt.hlines(success_rate, sequence_idx[0], sequence_idx[-1], color="gray", linestyle='--', alpha=0.3)



plt.tight_layout()

# +
experiments = 500 #200 #300 #200
n_samples = 2500 #2500 #1000 #1500
# ----------

experiment_stop_results_pitg = {'successes': [], 'trials': [], 'ci_min': [], 'ci_max': []}
iteration_stopping_on_or_prior_pitg_accept = {iteration: 0 for iteration in range(1, n_samples + 1)}
iteration_stopping_on_or_prior_pitg_reject_below = iteration_stopping_on_or_prior_pitg_accept.copy()
iteration_stopping_on_or_prior_pitg_reject_above = iteration_stopping_on_or_prior_pitg_accept.copy()


np.random.seed(seed)
samples = np.random.binomial(1, success_rate, [experiments, n_samples])

for sample in samples:
    successes = 0
    this_iteration = 0
    for toss in sample:
        successes += toss
        this_iteration += 1
        
        failures = this_iteration - successes
        
        aa = int(successes)
        bb = int(failures)
        
        if not failures:
            aa += 1
            bb += 1
            
        if not successes:
            aa += 1
            bb += 1
            
        ci_min, ci_max = successes_failures_to_hdi_ci_limits(aa, bb)
        
        this_precision_goal_achieved = (ci_max - ci_min) < precision_goal
     
        if this_precision_goal_achieved: 
            break
            
    this_accept_within = (ci_min >= rope_min) & (ci_max <= rope_max)

    if this_accept_within & this_precision_goal_achieved:
        for iteration in range(this_iteration, n_samples+1):
            iteration_stopping_on_or_prior_pitg_accept[iteration] += 1

    if (ci_max < rope_min) & this_precision_goal_achieved: 
        for iteration in range(this_iteration, n_samples+1):
            iteration_stopping_on_or_prior_pitg_reject_below[iteration] += 1

    if (ci_min > rope_max) & this_precision_goal_achieved:
        for iteration in range(this_iteration, n_samples+1):
            iteration_stopping_on_or_prior_pitg_reject_above[iteration] += 1

    experiment_stop_results_pitg['successes'].append(successes)
    experiment_stop_results_pitg['trials'].append(this_iteration)
    experiment_stop_results_pitg['ci_min'].append(ci_min)
    experiment_stop_results_pitg['ci_max'].append(ci_max)

# -

len(iteration_stopping_on_or_prior_pitg_reject_above)

# +

df_pitg_counts = pd.DataFrame({'accept': iteration_stopping_on_or_prior_pitg_accept,
               'reject_below': iteration_stopping_on_or_prior_pitg_reject_below, 
              'reject_above': iteration_stopping_on_or_prior_pitg_reject_above,
              })

df_pitg_counts.index.name = "iteration_number"
df_pitg_counts['reject'] = df_pitg_counts['reject_above'] + df_pitg_counts['reject_below']
df_pitg_counts['inconclusive'] = experiments - df_pitg_counts['accept'] - df_pitg_counts['reject']

print(df_pitg_counts.shape)
df_pitg_counts.head(4)
#hdi

# -

df_pitg_counts.tail(4)

df_pitg_counts['accept'].notnull().sum()

# +
plt.plot(df_pitg_counts.index, df_pitg_counts['accept'] / experiments, color="green")
plt.plot(df_pitg_counts.index, df_pitg_counts['reject'] / experiments, color="red")
plt.plot(df_pitg_counts.index, df_pitg_counts['inconclusive'] / experiments, color="gray")

#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(xlabel)
plt.title(title)

# +
df_stop_results_pitg = pd.DataFrame(experiment_stop_results_pitg)
#df_stop_results_hdi['success_rate'] = df_stop_results_hdi['successes'] / df_stop_results_hdi['trials']

df_stop_results_pitg.index.name = 'experiment_number'
df_stop_results_pitg['sample_success_rate'] = df_stop_results_pitg['successes'] * 1. / df_stop_results_pitg['trials']


df_stop_results_pitg['stop_criterion'] = df_stop_results_pitg['ci_max'] - df_stop_results_pitg['ci_min'] < precision_goal

print(df_stop_results_pitg.shape)
df_stop_results_pitg.head(4)
# -

df_stop_results_pitg['stop_criterion'].value_counts(dropna=False)

df_stop_results_pitg.describe()

# +
df_plot = df_stop_results_pitg.copy()
#df_plot = df_stop_results.query(f"trials < {df_stop_results['trials'].describe()['25%'] / 2}").copy()
df_plot = df_plot.query('stop_criterion')

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
plt.xlabel("sample success rate at decision")
plt.ylabel("frequency")

# -

# # Ultra Precision Is The Goal

# +
success_rate_null = 0.5
dsuccess_rate = 0.05 #success_rate * 0.1
rope_precision_fraction = 0.8


success_rate = 0.55
# --------

rope_min = success_rate_null - dsuccess_rate
rope_max = success_rate_null + dsuccess_rate

precision_goal = (2 * dsuccess_rate) * rope_precision_fraction

print(f"{success_rate_null:0.2}: null")
print(f"{rope_min:0.2}: ROPE min")
print(f"{rope_max:0.2}: ROPE max")
print("-" * 20)
print(f"{precision_goal:0.2}: Precision Goal")
print("-" * 20)
print(f"{success_rate:0.2}: true")


# +
experiments = 500 #200 #300 #200
n_samples = 2500 #2500 #1000 #1500
# ----------

experiment_stop_results_upitg = {'successes': [], 'trials': [], 'ci_min': [], 'ci_max': []}
iteration_stopping_on_or_prior_upitg_accept = {iteration: 0 for iteration in range(1, n_samples + 1)}
iteration_stopping_on_or_prior_upitg_reject_below = iteration_stopping_on_or_prior_upitg_accept.copy()
iteration_stopping_on_or_prior_upitg_reject_above = iteration_stopping_on_or_prior_upitg_accept.copy()


np.random.seed(seed)
samples = np.random.binomial(1, success_rate, [experiments, n_samples])

for sample in samples:
    successes = 0
    this_iteration = 0
    for toss in sample:
        successes += toss
        this_iteration += 1
        
        failures = this_iteration - successes
        
        aa = int(successes)
        bb = int(failures)
        
        if not failures:
            aa += 1
            bb += 1
            
        if not successes:
            aa += 1
            bb += 1
            
        ci_min, ci_max = successes_failures_to_hdi_ci_limits(aa, bb)
        
        this_precision_goal_achieved = (ci_max - ci_min) < precision_goal
     
        if this_precision_goal_achieved: 
            this_accept_within = (ci_min >= rope_min) & (ci_max <= rope_max)

            if this_accept_within & this_precision_goal_achieved:
                for iteration in range(this_iteration, n_samples+1):
                    iteration_stopping_on_or_prior_upitg_accept[iteration] += 1
                    
                break

            if (ci_max < rope_min) & this_precision_goal_achieved: 
                for iteration in range(this_iteration, n_samples+1):
                    iteration_stopping_on_or_prior_upitg_reject_below[iteration] += 1
                    
                break

            if (ci_min > rope_max) & this_precision_goal_achieved:
                for iteration in range(this_iteration, n_samples+1):
                    iteration_stopping_on_or_prior_upitg_reject_above[iteration] += 1
                    
                break

    experiment_stop_results_upitg['successes'].append(successes)
    experiment_stop_results_upitg['trials'].append(this_iteration)
    experiment_stop_results_upitg['ci_min'].append(ci_min)
    experiment_stop_results_upitg['ci_max'].append(ci_max)

# +

df_upitg_counts = pd.DataFrame({'accept': iteration_stopping_on_or_prior_upitg_accept,
               'reject_below': iteration_stopping_on_or_prior_upitg_reject_below, 
              'reject_above': iteration_stopping_on_or_prior_upitg_reject_above,
              })

df_upitg_counts.index.name = "iteration_number"
df_upitg_counts['reject'] = df_upitg_counts['reject_above'] + df_upitg_counts['reject_below']
df_upitg_counts['inconclusive'] = experiments - df_upitg_counts['accept'] - df_upitg_counts['reject']

print(df_upitg_counts.shape)
df_upitg_counts.head(4)

# +
title = f"true success rate = {success_rate:0.2f}"


plt.plot(df_upitg_counts.index, df_upitg_counts['accept'] / experiments, color="green")
plt.plot(df_upitg_counts.index, df_upitg_counts['reject'] / experiments, color="red")
plt.plot(df_upitg_counts.index, df_upitg_counts['inconclusive'] / experiments, color="gray")

#plt.xscale('log')
#plt.yscale('log')
plt.xlabel(xlabel)
plt.title(title)

# +
title = f"true success rate = {success_rate:0.2f}"


plt.plot(df_upitg_counts.index, df_upitg_counts['accept'] / experiments, color="green")
plt.plot(df_upitg_counts.index, df_upitg_counts['reject'] / experiments, color="red")
plt.plot(df_upitg_counts.index, df_upitg_counts['inconclusive'] / experiments, color="gray")

linestyle = '--'
plt.plot(df_pitg_counts.index, df_pitg_counts['accept'] / experiments, color="green", linestyle=linestyle)
plt.plot(df_upitg_counts.index, df_upitg_counts['reject'] / experiments, color="red", linestyle=linestyle)
plt.plot(df_upitg_counts.index, df_upitg_counts['inconclusive'] / experiments, color="gray", linestyle=linestyle)

# +
df_pitg_counts

df_upitg_counts.equals(df_pitg_counts)
# -





# +
df_stop_results_upitg = pd.DataFrame(experiment_stop_results_upitg)
#df_stop_results_hdi['success_rate'] = df_stop_results_hdi['successes'] / df_stop_results_hdi['trials']

df_stop_results_upitg.index.name = 'experiment_number'
df_stop_results_upitg['sample_success_rate'] = df_stop_results_upitg['successes'] * 1. / df_stop_results_upitg['trials']


df_stop_results_upitg['stop_criterion'] = df_stop_results_upitg['ci_max'] - df_stop_results_upitg['ci_min'] < precision_goal

print(df_stop_results_upitg.shape)
df_stop_results_upitg.head(4)

# +
df_plot = df_stop_results_upitg.copy()
#df_plot = df_stop_results.query(f"trials < {df_stop_results['trials'].describe()['25%'] / 2}").copy()
df_plot = df_plot.query('stop_criterion')

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
plt.xlabel("sample success rate at decision")
plt.ylabel("frequency")

# -







# # Old Scripts
#
# Consdier deleting

# +
# NHST stop criterion:
# For every new flip of the coin, stop and reject the null hypothesis, that θ=0.50, if p < .05 (two-tailed, conditionalizing on the current N), otherwise flip again.

experiments = 1000
success_rate = 0.5
samples = 200 #1500
# ----------

experiment_stopping = {}
iteration_stopping = {iteration: 0 for iteration in range(1, samples + 1)}
iteration_counts = {iteration: 0 for iteration in range(1, samples + 1)}

# iteration_stop = []
for iexperiment in np.arange(experiments):
    experiment_stopping[iexperiment] = {}
    
    values = []
    for iteration in range(1, samples + 1):
        
        np.random.seed(iexperiment * 100000 + iteration)
        values.append(np.random.choice([0,1]))
        
        p_value = binom_test(sum(values), n=len(values), p=success_rate_null, alternative='two-sided')
        
        experiment_stopping[iexperiment]['stop_iteration'] = len(values)
        experiment_stopping[iexperiment]['stop_value'] = sum(values) *1. / len(values)
        
        iteration_counts[iteration] += 1
        if p_value < p_value_thresh:
            iteration_stopping[iteration] += 1 
            experiment_stopping[iexperiment]['stop_p_value'] = p_value
            break
        
    #np.random.seed(seed)

    #sequence = np.random.binomial(1, success_rate, samples)
    #p_values = sequence_to_pvalues(sequence, success_rate_null=success_rate_null)

# +
sr_iteration_stopping = pd.Series(iteration_stopping)
sr_iteration_counts = pd.Series(iteration_counts).sort_index()

plt.scatter(sr_iteration_stopping.index, (sr_iteration_stopping * 1. / sr_iteration_counts))
# -





# +
# searching for extreme cases

success_rate = 0.5
samples = 500

for seed in np.arange(100):
    np.random.seed(seed)

    sequence = np.random.binomial(1, success_rate, samples)

    success_rate_null=0.5
    p_values = sequence_to_pvalues(sequence, success_rate_null=success_rate_null)

    n_reject = np.sum(p_values < p_value_thresh)
    
    if n_reject:
        print(f"{seed}: {n_reject}")

# +
# NHST stop criterion:
# For every new flip of the coin, stop and reject the null hypothesis, that θ=0.50, if p < .05 (two-tailed, conditionalizing on the current N), otherwise flip again.

experiments = 1000
success_rate = 0.5
samples = 500 #1500

experiment_results = np.zeros([experiments, samples])

# iteration_stop = []
for seed in np.arange(experiments):
    np.random.seed(seed)

    sequence = np.random.binomial(1, success_rate, samples)
    p_values = sequence_to_pvalues(sequence, success_rate_null=success_rate_null)
    
    experiment_results[seed, :] = (p_values < p_value_thresh)


# +
df_experiment_results = pd.DataFrame(experiment_results)
df_experiment_results.index += 1

print(df_experiment_results.shape)


# +
def temp(x):
    if x > 0:
        return x
    else:
        return None

iter_to_stop = df_experiment_results.idxmax(axis=1).map(temp)

# +
iteration_stopping_on_or_prior = {iteration: 0 for iteration in range(1, samples + 1)}

for iteration in iter_to_stop.values:
    if iteration == iteration:
        for this_iteration in range(int(iteration), samples + 1):
            iteration_stopping_on_or_prior[this_iteration] += 1


# +
sr_iteration_stopping_on_or_prior = pd.Series(iteration_stopping_on_or_prior)

sr_nhst_reject = sr_iteration_stopping_on_or_prior / experiments

plt.scatter(sr_nhst_reject.index, sr_nhst_reject, alpha=0.7, s=msize, color="purple")
plt.scatter(sr_nhst_reject.index, 1. - sr_nhst_reject, alpha=0.7, s=msize, color="gray")
plt.xscale('log')
# -



sequence_idx = np.arange(samples)+ 1
plt.scatter(sequence_idx, experiment_results.mean(axis=0))
plt.xscale('log')


# +

iteration_stop = []
for iexperiment in range(experiment_results.shape[0]):
    if experiment_results[iexperiment,:].sum():
        iteration_stop.append(np.argmax(experiment_results[iexperiment,:]))
    else:
        iteration_stop.append(-1)
        
# -

for n in range(1, experiment_results.shape[0] + 1):
    None

list(range(1, 10 + 1))

# +

np.unique(iteration_stop, return_counts=True)
# -



sequence_idx = np.arange(samples)+ 1
plt.scatter(sequence_idx, experiment_results.mean(axis=0))
plt.xscale('log')


iexperiment = 0

for idx in range(len(sequence)):
    p_value = binom_test(sum(sequence[:idx]), n=idx + 1, p=success_rate_null, alternative='two-sided') 
    
    if p_value < p_value_thresh:
        iteration_stop.append()
        break




