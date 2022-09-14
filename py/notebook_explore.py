# -*- coding: utf-8 -*-
# # Viz

# +
# Visualising
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

from utils_viz import (
    plot_success_rates,
    plot_success_rates_methods,
    plot_boundary_true_false_positive_rates,
    plot_metric_accuracy
)

# +
success = 10
failure = 10

plot_success_rates(success, failure)

# +
sample_success_rate = 0.51
sample_size = 600
ci_fraction = 0.95

plot_success_rates_methods(sample_success_rate, sample_size, ci_fraction,)

# +
success_rate_boundary = 0.6

plot_boundary_true_false_positive_rates(sample_success_rate, sample_size, success_rate_boundary=success_rate_boundary)
# -

plot_metric_accuracy()

# # Stats

from utils_stats import hdi_ci_full_width

p = 0.5
n = 1000 
hdi_ci_full_width(p, n)

# # Comparing Models

# +
success_a = 702
failure_a = 6754

success_b = 1259
failure_b = 11455

plot_success_rates(success_a, failure_a, color="orange")
plot_success_rates(success_b, failure_b, color="red")
plt.xlim(0.08, 0.12)

# +
# emergent__pc_treatable + ed_care_needed__preventable_avoidable

# preventive_visits_all_time = 0
n_a = 690146
mean_a = 0.2758028407032731
std_a = 0.34372249540149785

# preventive_visits_all_time > 0
n_b = 507189
mean_b = 0.28032955840919244
std_b = 0.34755655272795394

# +
import numpy as np

(mean_a - mean_b)/np.sqrt((std_a**2 + std_b**2)/2.)

# +
# emergent__pc_treatable + ed_care_needed__preventable_avoidable

# triage.total_completed_triage_instances = 0
n_a = 152504
mean_a = 0.28325206814247217
std_a = 0.34548452234119742

# triage.total_completed_triage_instances > 0
n_b = 31002
mean_b = 0.28023434939681441
std_b = 0.33488510853509423

# +
# emergent__pc_treatable + ed_care_needed__preventable_avoidable


# triage.total_completed_triage_instances = 0
n_a = 152504
mean_a = 0.28325206814247217
std_a = 0.34548452234119742

# triage.total_completed_triage_instances > 0
n_b = 31002
mean_b = 0.28023434939681441
std_b = 0.33488510853509423

# +
# paid_amount

# preventive_visits_all_time = 0
n_a = 690146
mean_a = 57.453371645489284
std_a = 168.15767129255698

# preventive_visits_all_time > 0
n_b = 507189
mean_b = 65.9646584095788
std_b = 187.36685086020245

# +
# paid_amount

# triage.total_completed_triage_instances = 0
n_a = 152504
mean_a = 57.1402223553858
std_a = 161.28927879252055

# triage.total_completed_triage_instances > 0
n_b = 31002
mean_b = 59.854394587159554
std_b = 171.69039287514522
# -



# # For Slides

# +
sample_success_rate = 0.7
sample_size = 30
ci_fraction = 0.95

plot_success_rates_methods(sample_success_rate, sample_size, ci_fraction, display_ci=True)
plt.xlim(0., 1.)

# +
rate_hypothesis = 0.5
rate_rope_half_width = 0.02

success = 45
failure = 55

# ----
rope_min = rate_hypothesis - rate_rope_half_width
rope_max = rate_hypothesis + rate_rope_half_width

plot_success_rates(success, failure)
plt.xlim(0., 1.)

ax = plt.gca()
ymin, ymax = ax.get_ylim()
plt.fill_between([rope_min, rope_max], [ymax, ymax], color='gray', alpha=0.3)

# +
success = 12 + 1
failure = 0 + 1

plot_success_rates(success, failure)
plt.xlim(0, 1)
# -
# # BEST
#
# [PyMC3](https://docs.pymc.io/en/v3/pymc-examples/examples/case_studies/BEST.html)


# +
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

print(f"Running on PyMC3 v{pm.__version__}")
# -

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(seed=42)

# +
μ_m = y.value.mean()
μ_s = y.value.std() * 2

with pm.Model() as model:
    group1_mean = pm.Normal("group1_mean", mu=μ_m, sigma=μ_s)
    group2_mean = pm.Normal("group2_mean", mu=μ_m, sigma=μ_s)
