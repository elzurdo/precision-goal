# +
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# https://en.wikipedia.org/wiki/Binomial_test
from scipy.stats import binom_test # binomtest (in later versions ...)

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

# +
success_rate_null = 0.5
alternative = 'two-sided' # 'greater'
# -----

n_experiments = 5000 # 5000 #1000 #10000
n_samples = 2**12

success_rate = 0.55
# -----

seed = 1


np.random.seed(seed)
samples = np.random.binomial(1, success_rate, [n_experiments, n_samples])

# +
#all_n_samples = [n_samples // 16, n_samples // 8, n_samples // 4, n_samples // 2, n_samples]

all_n_samples = [n_samples // 16, n_samples]
# -

# ## Testing
#
# Might be worth comparing with [source](https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_morestats.py#L2648-L2741).

# +
from scipy.stats import binom

def binomal_prob_x_equal_k(k,n,p):
    return comb(n, k, repetition=False) * (p**k) * (1 - p)**(n - k)
 
binomal_prob_x_equal_k(5, 10, 0.5), binom.pmf(5, 10, 0.5)

# +
_1 = np.sum([comb(tosses, i_, repetition=False) * (success_rate_null**i_) * (1 - success_rate_null)**(tosses - i_) for i_ in range(k_ + 1)])
_2 = np.sum([binomal_prob_x_equal_k(i_, tosses, success_rate_null) for i_ in range(k_ + 1)])
_3 = np.sum([binom.pmf(i_, tosses, success_rate_null) for i_ in range(k_ + 1)])

_1, _2, _3
# -

binom.pmf(129.5, tosses, success_rate_null), i_

# +
from scipy.special import comb

tosses = 256  #samples.shape[1] # 256

k_ = pd.Series(samples[0][:tosses]).value_counts()[1]
#k_ = 100

# one sided: testing p_alt < p_null
p_value_low = np.sum([binom.pmf(i_, tosses, success_rate_null) for i_ in range(k_ + 1)])

# one sided: testing p_alt > p_null
p_value_high = np.sum([binom.pmf(i_, tosses, success_rate_null) for i_ in range(k_, tosses + 1)])

# two sided 
sr_aux = pd.Series([binom.pmf(i_, tosses, success_rate_null) for i_ in range(tosses + 1)]) #, index=range(tosses + 1))
p_value_2sided = sr_aux[sr_aux <= binom.pmf(k_, tosses, success_rate_null)].sum()

k_, tosses, tosses/2, k_/tosses,p_value_low, p_value_high, p_value_high*2, p_value_2sided

# +
# low side

plt.bar(range(k_ + 1), [binom.pmf(i_, tosses, success_rate_null) for i_ in range(k_ + 1)], alpha=0.3, color="purple")

# +
# high side

plt.bar(range(k_, tosses + 1), [binom.pmf(i_, tosses, success_rate_null) for i_ in range(k_, tosses + 1)], alpha=0.3, color="purple")

# +
sr_aux = pd.Series([binom.pmf(i_, tosses, success_rate_null) for i_ in range(tosses + 1)]) #, index=range(tosses + 1))
#p_value_2sided = sr_aux[sr_aux <= binom.pmf(k_, tosses, success_rate_null)].sum()

plt.bar(sr_aux.index, sr_aux.values, width=0.2)

#plt.xlim(40., 85.)

# +
#sr_aux2.sort_values()
# -

sr_aux2.sum()

# +
sr_aux2 = sr_aux[sr_aux <= binom.pmf(k_, tosses, success_rate_null)]

plt.bar(sr_aux2.index, sr_aux2.values, width=0.2)
plt.yscale("log")

# +

binom.pmf(k_, tosses, success_rate_null)
# -

sr_aux[sr_aux <= binom.pmf(k_, tosses, success_rate_null)].sum()

ii = np.arange(np.ceil(success_rate_null * tosses), tosses+1)

# +
dd = binom.pmf(k_, tosses, success_rate_null)
rerr = 1 + 1e-7

yy = np.sum(binom.pmf(ii, tosses, success_rate_null) <= dd*rerr, axis=0)
yy


# -

pd.Series(samples[0][:tosses]).value_counts()

sum(this_sample), this_n_sample, success_rate_null

# +
sr_aux = pd.Series([binom.pmf(i_, this_n_sample, success_rate_null) for i_ in range(this_n_sample + 1)]) #, index=range(tosses + 1))
p_value_2sided = sr_aux[sr_aux < binom.pmf(sum(this_sample), this_n_sample, success_rate_null)].sum()

p_value_2sided

# +
k_ = 79
n_ = 256 // 2

# 'two-sided', 'greater', 'less'
binom_test(k_, n=n_, p=success_rate_null, alternative='two-sided') 
# -

# ## End Testing

# +
all_p_values = np.zeros((len(all_n_samples), n_experiments)) - 1

for idx_experiment, sample in enumerate(samples):
    for idx_n_sample, this_n_sample in enumerate(all_n_samples):
        this_sample = sample[:this_n_sample]
        
        this_p_value = binom_test(sum(this_sample), n=this_n_sample, p=success_rate_null, alternative=alternative)        
        all_p_values[idx_n_sample, idx_experiment] = this_p_value
    


# +
import pandas as pd

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

theta_true = r"$\theta_{true}$"
theta_null = r"$\theta_{null}$"
plt.title(f"{theta_true}: {success_rate:0.2}, {theta_null}: {success_rate_null:0.2}, NHST: {alternative}")
# -

pd.Series(all_p_values[0,:]).describe(percentiles=np.arange(0.,1.,0.1)) * 100



# +
dxx = 0.1
bins = np.arange(0, 1 + dxx, dxx)

for idx_n_sample, this_n_sample in enumerate(all_n_samples):
    label = f"{this_n_sample:,}"
    plt.hist(all_p_values[idx_n_sample], bins=bins, histtype="step", label=label, linewidth=3, density=True)
#plt.hist(all_p_values[1], bins=bins, histtype="step")
plt.legend(title='tosses', loc='center')
plt.xlabel("p-value")
plt.ylabel("frequency")
plt.title(f"true: {success_rate:0.2}, null: {success_rate_null:0.2}, NHST: {alternative}")
# -

(all_p_values < 0.001).sum()

# +
from utils_stats import (
    #hdi_ci_limits,
    successes_failures_to_hdi_ci_limits
)

successes = sum(this_sample)
failures = this_n_sample - successes

ci_min, ci_max = successes_failures_to_hdi_ci_limits(successes, failures)

ci_max - ci_min
# -


