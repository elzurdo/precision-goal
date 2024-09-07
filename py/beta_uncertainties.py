# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: scrappy-3.8.11
#     language: python
#     name: python3
# ---

# %%
# IPython extension to reload modules before executing user code.
# useful to see immediate results in notebook when modifying imported scripts
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import pandas as pd

from scipy.stats import beta

from utils_stats import (
    #hdi_ci_limits,
    successes_failures_to_hdi_ci_limits
)

from utils_viz import (
    #plot_success_rates
    plot_vhlines_lines,
    plot_parity_line,
    #plot_multiple_decision_rates_jammed,
    #plot_multiple_decision_rates_separate,
    #scatter_stop_iter_sample_rate,
    #viz_one_sample_results,
    #plot_sample_pdf_methods,
)

# %%

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

# %% [markdown]
# # Beta Function HDI vs. Variance

# %%
means = [0.5, 0.75, 0.9]
ns = [30, 100, 1000]



# %%
means = [0.5, 0.75, 0.9]
ns = [30, 70, 100, 300, 700, 1000]

z_0pt05 = 1.96
twoside_factor = 2

dict_stats = {}
idx = 0

for n_ in ns:  # number of trials (success + failures)
    for mean_ in means:  # mean value of beta distribution
        dict_stats[idx] = {
                        "n": n_,
            "mean": mean_,
            "a": n_ * mean_,  # number of successes
            "b": n_ - n_ * mean_,  # number of failures
        }
        idx += 1

df_stats = pd.DataFrame(dict_stats).T

df_stats["var"] = df_stats.apply(lambda x: beta.var(x["a"], x["b"]), axis=1)
df_stats["ci_width"] = df_stats.apply(lambda x: np.sqrt(x["var"]) * twoside_factor* z_0pt05, axis=1)

df_hdi = pd.DataFrame(df_stats.apply(lambda x: successes_failures_to_hdi_ci_limits(x["a"], x["b"]), axis=1).tolist(), columns=["hdi_min", "hdi_max"])
df_hdi["hdi_width"] = df_hdi["hdi_max"] - df_hdi["hdi_min"]
df_stats = df_stats.join(df_hdi)

df_stats

# %%
# Consistency test
# I verified that beta.var yields the same result as the formula below
# def a_b_to_var(a, b):
#     n = a + b
#     n2 = n ** 2
#     n_plus_1 = n + 1
#     return a * b / n2 / n_plus_1

# df_stats.apply(lambda x: a_b_to_var(x["a"], x["b"]), axis=1)


# %%
for idx, mean in enumerate(means):
    df_subset = df_stats[df_stats["mean"] == mean]
    linewidth = idx + 1
    print(mean)
    plt.plot(df_subset["n"], df_subset["hdi_width"], linestyle=None, linewidth=linewidth)
    plt.plot(df_subset["n"], df_subset["ci_width"], linestyle="--", linewidth=linewidth)


#plt.legend(title="E[X]")
#plt.xlabel("Number of trials")
#plt.ylabel("credible interval")

# %%
df_stats["ci_diff"] = df_stats["ci_width"] - df_stats["hdi_width"]
df_stats["ci_frac"] = df_stats["ci_diff"] / df_stats["hdi_width"]

# %%

# %%
for mean in means:
    df_subset = df_stats[df_stats["mean"] == mean]
    plt.plot(df_subset["n"], df_subset["ci_diff"], label=f"{mean}")

plt.legend(title="E[X]")

# %%
