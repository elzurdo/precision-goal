# +
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pymc as pm
import scipy
import arviz as az


# %load_ext autoreload
# %autoreload 2
# -

print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
import matplotlib
print(f"matplotlib: {matplotlib.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"pymc: {pm.__version__}")
print(f"arviz: {az.__version__}")

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

from utils_stats import sequence_to_sequential_pvalues
from utils_data import generate_biomial_sequence
from utils_viz import plot_sequence_experiment_nhst_combo_results

# +
seed1 = 13 #98 #31
success_rate1 = 0.5 #0.65 #0.5
n_samples1 = 1500

sequence1 = generate_biomial_sequence(success_rate=success_rate1,
                                     n_samples=n_samples1,
                                     seed=seed1
                                    )
len(sequence1)

# +
seed2 = 14 #98 #31
success_rate2 = 0.4 #0.65 #0.5
n_samples2 = 17 #00

sequence2 = generate_biomial_sequence(success_rate=success_rate2,
                                     n_samples=n_samples2,
                                     seed=seed2
                                    )
len(sequence2)

# +
both_sequences = np.concatenate([sequence1, sequence2])

_, bins = np.histogram(both_sequences)

plt.hist(sequence1, bins=bins, density=True, alpha=0.4)
plt.hist(sequence2, bins=bins, density=True, alpha=0.4)

# +
successes1 = 175 #sequence1.sum()
failures1 = 2826 #len(sequence1) - successes1

successes2 = 493  # sequence2.sum()
failures2 = 6081 # len(sequence2) - successes2

# successes = both_sequences.sum()
# failures = len(both_sequences) - successes

# del successes, failures
# -

with pm.Model() as model:
    p1_mean = pm.Beta("p1_mean", alpha=successes1, beta=failures1)
    p2_mean = pm.Beta("p2_mean", alpha=successes2, beta=failures2)
    
    Δp = pm.Deterministic("Δp", p1_mean-p2_mean)

with model:
    idata = pm.sample()

# +
axes = az.plot_posterior(
    idata,
    var_names=["p1_mean", "p2_mean"],
    color="#87ceeb",
);

xlims1 = axes[0].get_xlim()
xlims2 = axes[1].get_xlim()
xlims = [np.min([xlims1[0], xlims2[0]]), np.max([xlims1[1], xlims2[1]])]

axes[0].set_xlim(xlims)
axes[1].set_xlim(xlims)
# -

az.plot_posterior(
    idata,
    var_names=["Δp"],
    color="#87ceeb",
    ref_val=0,
);

az.plot_forest(idata, var_names=["p1_mean", "p2_mean"]);

az.plot_forest(idata, var_names=["Δp"]);

az.summary(idata, var_names=["p1_mean", "p2_mean", "Δp"])

# # Resources
#
# * PyMC
#     * [PyMC BEST tutorial](https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html)
#     * [Beta distribution](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Beta.html)
# * [NYU page](https://cims.nyu.edu/~brenden/courses/labincp/tips/ultimate-guide-ttest-python.html)
# * [blogspot](http://nmouatta.blogspot.com/2016/08/the-bayesian-t-test-in-python_25.html)


