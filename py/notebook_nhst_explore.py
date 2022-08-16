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
# -

from utils_stats import (
    #hdi_ci_limits,
    successes_failures_to_hdi_ci_limits
)



# +
success_rate_null = 0.5
alternative = 'two-sided' # 'greater'
# -----

n_experiments = 5000 #1000 #10000
n_samples = 2**12

success_rate = 0.55
# -----

seed = 1


np.random.seed(seed)
samples = np.random.binomial(1, success_rate, [n_experiments, n_samples])
# -

all_n_samples = [n_samples // 16, n_samples // 8, n_samples // 4, n_samples // 2, n_samples]
all_n_samples

# +
all_p_values = np.zeros((len(all_n_samples), n_experiments)) - 1


for idx_experiment, sample in enumerate(samples):
    for idx_n_sample, this_n_sample in enumerate(all_n_samples):
        this_sample = sample[:this_n_sample]
        
        this_p_value = binom_test(sum(this_sample), n=this_n_sample, p=success_rate_null, alternative=alternative)        
        all_p_values[idx_n_sample, idx_experiment] = this_p_value
    


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

# +
successes = sum(this_sample)
failures = this_n_sample - successes

ci_min, ci_max = successes_failures_to_hdi_ci_limits(successes, failures)

ci_max - ci_min
# -


