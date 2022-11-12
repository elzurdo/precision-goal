# Testing [Binomal Test](https://en.wikipedia.org/wiki/Binomial_test) as used in [scipy.stats](https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_morestats.py#L2648-L2741).

# +
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# https://en.wikipedia.org/wiki/Binomial_test
from scipy.stats import binom_test # binomtest (in later versions ...)
from scipy.special import comb
from scipy.stats import binom

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

theta_true_label = r"$\theta_{true}$"
theta_null_label = r"$\theta_{null}$"

# # One Binomial Test

# +
n, k, theta_null = 10, 8, 0.5
#n, k, p = 256, 130, 0.5

# -------
rerr = 1 + 1e-7

# one sided: testing p_alt < p_null
p_value_low = np.sum([binom.pmf(i, n, theta_null) for i in range(k + 1)])

# one sided: testing p_alt > p_null
p_value_high = np.sum([binom.pmf(i, n, theta_null) for i in range(k, n + 1)])

# +
probs_all = [binom.pmf(i, n, theta_null) for i in range(n + 1)] 

probs_all = pd.Series(probs_all)
probs_all[probs_all <= binom.pmf(k, n, theta_null) * rerr].sum() 

# +
width = 0.4


thresh = binom.pmf(k, n, theta_null) * rerr 

probs_gt_thresh = probs_all[probs_all >  thresh]
probs_le_thresh = probs_all[probs_all <= thresh]

plt.bar(probs_le_thresh.index, probs_le_thresh.values, width=width, color="red", alpha=0.7, hatch="*", label=f"p-value = {100. * probs_le_thresh.sum():0.2f}%")
plt.bar(probs_gt_thresh.index, probs_gt_thresh.values, width=width, color="purple", alpha=0.7, fill=False, label=f"1 - p-value = {100. * probs_gt_thresh.sum():0.2f}%")

plt.legend()

khat = r"$\hat{k}$"
plt.yscale("log")
plt.xlabel(fr"k (successes of n={n} trials)")
plt.ylabel(f"log(P($k$))")
plt.title(fr"$n$={n}, {khat}={k}, {theta_null_label}={theta_null}")
# -

probs_le_thresh.sum()

# # Many Binomial Tests

# +
# Simulation setup

theta_null = 0.5 # null hypothesis success rate
alternative = 'two-sided' # 'greater'
# -----

n_experiments = 5000 # 5000 #1000 #10000
n_samples = 2**12

theta_true = 0.5 #0.55 # true success rate
# -----

seed = 1


np.random.seed(seed)
samples = np.random.binomial(1, theta_true, [n_experiments, n_samples])

# +
# exploring different stop criterions

#all_n_samples = [n_samples // 16, n_samples // 8, n_samples // 4, n_samples // 2, n_samples]

all_n_samples = [n_samples // 16, n_samples]

# +
all_p_values = np.zeros((len(all_n_samples), n_experiments)) - 1

all_success_values = np.zeros((len(all_n_samples), n_experiments)) - 1 # this variable is for debugging purposes

for idx_experiment, sample in enumerate(samples):
    for idx_n_sample, this_n_sample in enumerate(all_n_samples):
        this_sample = sample[:this_n_sample]
        
        this_p_value = binom_test(sum(this_sample), n=this_n_sample, p=theta_null, alternative=alternative)        
        all_p_values[idx_n_sample, idx_experiment] = this_p_value
        
        all_success_values[idx_n_sample, idx_experiment] = this_sample.sum() # this variable is for debugging purposes


# +
plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT))
hatches = ["/", "\\", "-", ".", "*"]


n_ntosses = all_p_values.shape[0]


width = 1./(n_ntosses + 1)

for idx_n_sample, this_n_sample in enumerate(all_n_samples):
    label = f"{this_n_sample:,}"
    
    sr_aux = pd.Series(np.floor(all_p_values[idx_n_sample] * 10)).value_counts(normalize=True).sort_index() * 100.
    plt.bar(sr_aux.index - 0.5 *(width*(n_ntosses - 1)) + idx_n_sample * width, sr_aux.values, alpha=0.6, width=width, hatch=hatches[idx_n_sample], label=label)
 

xtick_vals = np.arange(10 + 1)
xtick_strs = [f"{xtick_val/10:0.1f}-{xtick_val/10 +0.1 -0.001:0.3f}" for xtick_val in xtick_vals]
xtick_strs[-1] = "1"

plt.xticks(xtick_vals, xtick_strs)
plt.legend(title='tosses', loc='right')
plt.xlabel("p-value")
plt.ylabel("%")

plt.title(f"{theta_true_label}: {theta_true:0.2}, {theta_null_label}: {theta_null:0.2}, NHST: {alternative}")
# -

# # Testing

# +
idx_n_sample = 0

floored = pd.Series(np.floor(all_p_values[idx_n_sample] * 10), index=all_p_values[idx_n_sample])

floored.index.value_counts()
# -

pd.Series(all_success_values[idx_n_sample]).value_counts()

# +
idx_n_sample = 0


floored = np.floor(all_p_values[idx_n_sample] * 10)
sr_floored = pd.Series(floored).value_counts(normalize=True).sort_index() * 100.
idx_mode = 0
plt.bar(sr_floored.index - 0.5 *(width*(n_ntosses - 1)) + idx_mode * width, sr_floored.values, alpha=0.6, width=width, hatch=hatches[idx_n_sample], label="floored")


# ceiled = np.ceil(all_p_values[idx_n_sample] * 10)
# sr_ceiled = pd.Series(ceiled).value_counts(normalize=True).sort_index() * 100.
# idx_mode = 1
# plt.bar(sr_ceiled.index - 0.5 *(width*(n_ntosses - 1)) + idx_mode * width, sr_ceiled.values, alpha=0.6, width=width, hatch=hatches[idx_n_sample], label="ceiled")

rounded = np.round(all_p_values[idx_n_sample] * 10)
sr_rounded = pd.Series(rounded).value_counts(normalize=True).sort_index() * 100.
idx_mode = 1
plt.bar(sr_rounded.index - 0.5 *(width*(n_ntosses - 1)) + idx_mode * width, sr_rounded.values, alpha=0.6, width=width, hatch=hatches[idx_n_sample], label="rounded")


plt.legend()
# -

sr_floored

sr_floored[3:5].mean()

# +
test_ = np.array([0.44, 0.45, 0.46])

np.round(test_ * 10)
# -


