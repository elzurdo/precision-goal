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
#
#
# Assessing approximation $\text{CI}$ with $95% \text{HDI}$ as defined
#
# $$\text{CI} = 2 z_{0.05} \sqrt{\text{var}}$$
#
# where
#
# $$\text{var} = \frac{ab}{(a + b)^2(a + b +1)}$$
#

# %%
means = [0.5, 0.75, 0.9]
ns = [30, 100, 1000]

# %%
means = [0.5, 0.75, 0.9]
ns = [30, 40, 50, 60, 70, 80, 90, 100] #, 150, 200, 250, 300, 500, 700, 1000]

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
for idx, mean in enumerate(means):
    df_subset = df_stats[df_stats["mean"] == mean]
    linewidth = idx + 1
    plt.plot(df_subset["n"], df_subset["ci_diff"], label=f"{mean}", linewidth=linewidth)


plot_vhlines_lines(horizontal=0.01, vertical=None, label="1%", linestyle="--")
plot_vhlines_lines(horizontal=0.001, vertical=None, label="0.1%", linestyle="-.")
plt.yscale("log")
plt.legend(title="E[X]")
plt.ylabel("Approx - 95%HDI")

# %%
for idx, mean in enumerate(means):
    df_subset = df_stats[df_stats["mean"] == mean]
    linewidth = idx + 1
    plt.plot(df_subset["n"], df_subset["ci_frac"] * 100., label=f"{mean}", linewidth=linewidth)


plot_vhlines_lines(horizontal=1., vertical=None, label="1%", linestyle="--")
plot_vhlines_lines(horizontal=0.1, vertical=None, label="0.1%", linestyle="-.")
plt.legend(title="E[X]")
plt.yscale("log")
plt.yscale("log")
plt.ylabel("(Approx - 95%HDI)/95%HDI (%)")


# %% [markdown]
# # $N$(95% HDI)
#
# Starting point
#
# $$\text{var} = \frac{ab}{(a + b)^2(a + b +1)}$$
#
# where $a = n p$, $b = n * (1-p)$
#
# $$\text{var} = \frac{np \cdot n(1-p)}{n^2(n +1)} = \frac{p(1-p)}{(n+1)}$$
#
# $$n = \frac{p(1-p)}{\text{var}} - 1$$
#
# Using 
#
# $$\text{CI} = 2 z_{0.05} \sqrt{\text{var}}$$
#
#
# $$\frac{1}{\text{var}} = \frac{4 z_{0.05}^2}{\text{CI}^2}$$
#
#
# $$n = \frac{4 z_{0.05}^2}{\text{CI}^2}p(1-p) - 1$$

# %%
def p_ci_to_n(p, ci_width, z_star=1.96):
    return (4 * z_star**2) * p * (1 - p) / ci_width**2 - 1

p_ci_to_n(0.85, 0.08, z_star=1.96)

# %%
ci_widths = [0.08, 0.075, 0.07, 0.065, 0.06, 0.055, 0.05]
ps = [0.5, 0.75, 0.9]
z_star = 1.96

survey_sizes = {}

for ci_width in ci_widths:
    survey_sizes[ci_width] = {}
    for p in ps:
        survey_sizes[ci_width][p] = p_ci_to_n(p, ci_width, z_star=z_star)


df_sizes = pd.DataFrame(survey_sizes).T

df_sizes.index.name ="ci_width"  
df_sizes.columns.name = "p"

df_sizes

# %%
for idx, p in enumerate(ps):
    sr_ci = df_sizes[p]
    linewidth = idx + 1
    plt.plot(sr_ci.index, sr_ci, linewidth=linewidth, label=f"{p}")

p = 0.5
ci_width = 0.08
n_ = p_ci_to_n(p, ci_width, z_star=1.96)
plt.scatter(ci_width, n_, color="black", label=f"p={p}, CI={ci_width}", alpha=0.7)
#plot_vhlines_lines(horizontal=n_, vertical=ci_width, label="p=0.5, CI=0.08", linestyle="--")

plt.legend(title="p")
plt.xlabel("CI width")
plt.ylabel("Survey size")


# %% [markdown]
# # Implementing HDI + ROPE
#
# Location only
#
# horizontal - n
#
# vertical - p
#
# Useful definitions
#
# $$\text{CI}_\text{min} = p - 0.5 \text{CI}\\ \text{CI}_\text{max} = p + 0.5 \text{CI}$$
# $$\text{CI} = 2 z_{0.05} \sqrt{\text{var}}$$ 
# $$\text{var} = \frac{p(1-p)}{(n+1)}$$
# $$\text{ROPE}_\text{h} = \theta_\text{null} + 0.5 \text{ROPE}$$
# $$\text{ROPE}_\text{l} = \theta_\text{null} - 0.5 \text{ROPE}$$
#
#
# **Instances**
#
# reject high rule:
# $$\text{CI}_\text{min} \gt \text{ROPE}_\text{h}$$
#
# $$p - z_{0.05} \sqrt{\frac{p(1-p)}{(n+1)}} \gt \text{ROPE}_\text{h}$$
#
# $$( p - \text{ROPE}_\text{h})^2 \gt z_{0.05}^2 \frac{p(1-p)}{(n+1)}$$
#
# $$n\gt z_{0.05}^2 \frac{p(1-p)}{ ( p - \text{ROPE}_\text{h})^2}- 1$$
#
# reject low rule:
#
# $$\text{CI}_\text{max} \lt \text{ROPE}_\text{l}$$
#
# $$p + z_{0.05} \sqrt{\frac{p(1-p)}{(n+1)}} \lt \text{ROPE}_\text{l}$$
#
# $$n\gt z_{0.05}^2 \frac{p(1-p)}{ ( p - \text{ROPE}_\text{l})^2}- 1$$
#
#
# reject low rule: ci_max < p_null - ROPE/2
#
#
# accept rule ....
#
#
#
#
# p - ci/2 = p_null + ROPE/2
#
# p = p_null + ROPE/2 + ci/2
#
#

# %%
def null_p_to_n(p, null_p, rope, z_star=1.96, rope_high_low = "high"):
    if "high" == rope_high_low:
        rope_edege = null_p + 0.5 * rope  # rope_h
        if p <= rope_edege:
            return None
    elif "low" == rope_high_low:
        rope_edege = null_p - 0.5 * rope  # rope_l
        if p >= rope_edege:
            return None
    else:
        raise ValueError("rope_high_low must be either 'high' or 'low'")
        
    return (z_star**2) * p * (1-p) / (p-rope_edege)**2 - 1


null_p_to_n(0.6, 0.5, 0.1, z_star=1.96)

# %%
# size of circle depends on realtive probability of getting a data point there (or under)

rope = 0.1
null_p = 0.5
z_star = 1.96

ps = np.arange(null_p + 0.1, 0.9, 0.01)
ns_rejecth = []
ns_rejectl = []
for p in ps:
    ns_rejecth.append(null_p_to_n(p, null_p, rope, z_star=z_star))
    ns_rejectl.append(null_p_to_n(1 - p, null_p, rope, z_star=z_star, rope_high_low="low"))

plt.plot(ns_rejecth, ps, '-o', color="red", label=f"Reject High", alpha=0.7)
plt.plot(ns_rejectl, 1 - ps, '-s', color="red", label=f"Reject Low", alpha=0.7)
plot_vhlines_lines(horizontal=null_p, vertical=None, label="Null", linestyle="-", alpha=0.9, color="black")
plot_vhlines_lines(horizontal=null_p + 0.5 * rope, vertical=None, label="ROPE", linestyle="--", alpha=0.7, color="black")
plot_vhlines_lines(horizontal=null_p - 0.5 * rope, vertical=None, label=None, linestyle="--", alpha=0.4, color="black")

plt.legend()

# %%
p, 1 - p, null_p, rope

# %%
null_p_to_n(1 - p, null_p, rope, z_star=z_star, rope_high_low="low")


# %%

# %% [markdown]
#

# %% [markdown]
# # Impelmenting PitG

# %%

# %% [markdown]
# # Implementing EPitG

# %% [markdown]
#
