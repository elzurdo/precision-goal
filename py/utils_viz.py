import matplotlib.pyplot as plt
import numpy as np

from utils_stats import (CI_FRACTION,
                         successes_failures_to_hdi_ci_limits,
                         get_success_rates,
                         beta,
                         test_value
                         )

def plot_success_rates(success, failure, ci_fraction=CI_FRACTION,
                       min_psuccess=0.85, max_psucess=1.,d_psuccess=0.0001,
                       color="purple", format='-', label=None, fill=False, display_ci=True,
                       alpha=1., factor=1., ci_label=None,
                       xlabel="success rate",
                       ylabel="probability distribution function", xmin=None, xmax=None):

    ci_min, ci_max = successes_failures_to_hdi_ci_limits(success, failure, ci_fraction=ci_fraction)

    _, p_success = get_success_rates(d_success = d_psuccess, min_range=0., max_range=1., including_max=True)
    beta_pdf = beta.pdf(p_success, test_value(success), test_value(failure))

    plt.plot(p_success, beta_pdf * factor, format, linewidth=3, color=color, label=label, alpha=alpha)

    xmin_ = np.max([p_success.min(), ci_min * 0.95])
    if xmin is not None:
        xmin_ = np.min([xmin_, xmin])
        xmin_ = np.max([xmin_, 0.])
    xmax_ = np.min([p_success.max(), ci_max * 1.05])
    if xmax is not None:
        xmax_ = np.max([xmax_, xmax])
        xmax_ = np.min([xmax_, 1.])



    plt.xlim(xmin_, xmax_)

    if display_ci == True:
        plt.plot([ci_min, ci_min],
                 [0, beta.pdf(ci_min, success, failure) * factor],
                 "--", color=color, alpha=alpha, label=ci_label)
        plt.plot([ci_max, ci_max],
                 [0, beta.pdf(ci_max, success, failure) * factor],
                 "--", color=color, alpha=alpha)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)


def _get_sequence_idx(sequence):
    n_samples = len(sequence)
    return np.arange(n_samples)+ 1

def plot_sequence_experiment_cumsum_average(sequence, success_rate_true=None, xlabel="iteration", msize=5):
    if success_rate_true:
        theta_true_str = r"$\theta_{true}$"
        title = f" {theta_true_str} = {success_rate_true:0.2f}"
    else:
        title = None
    dsuccess_rate_plot = 0.07

    sequence_idx = _get_sequence_idx(sequence)
    sequence_average = sequence.cumsum() / sequence_idx

    #errorbar = 1.96 * np.sqrt((sequence_average * (1. - sequence_average)) / sequence_idx )
    #plt.errorbar(sequence_idx, sequence_average, yerr=errorbar, color="gray", alpha=0.05)
    plt.scatter(sequence_idx[sequence == 1], sequence_average[sequence == 1], color = "green", alpha=0.7, s=msize)
    plt.scatter(sequence_idx[sequence == 0], sequence_average[sequence == 0], color = "red", alpha=0.7, s=msize)
    if success_rate_true:
        plt.hlines(success_rate_true, sequence_idx[0], sequence_idx[-1], color="gray", linestyle='--', alpha=0.3)
        plt.annotate(title, xy=(sequence_idx[-500], success_rate_true + 0.004), color="gray", alpha=0.4)

        plt.ylim(success_rate_true - dsuccess_rate_plot, success_rate_true + dsuccess_rate_plot)

    plt.xlabel(xlabel)
    plt.ylabel("cumsum average")
    if title:
        plt.title(title)

def plot_sequence_experiment_nhst_combo_results(sequence, success_rate_true, success_rate_null, p_values, p_value_thresh=0.05, xlabel="iteration", msize=5):

    sequence_idx = _get_sequence_idx(sequence)

    plt.subplot(2, 1, 1)
    plot_sequence_experiment_cumsum_average(sequence, success_rate_true=success_rate_true,xlabel=xlabel, msize=msize)

    plt.subplot(2, 1, 2)
    plt.hlines(p_value_thresh, sequence_idx[0], sequence_idx[-1], color="gray", linestyle='--', alpha=0.3)
    plt.scatter(sequence_idx[p_values >= p_value_thresh], p_values[p_values >= p_value_thresh], color = "gray", alpha=0.7, s=msize)
    plt.scatter(sequence_idx[p_values < p_value_thresh], p_values[p_values < p_value_thresh], color = "blue", marker='x', s=msize * 10)
    plt.xlabel(xlabel)
    plt.annotate(f"decision criterion p-value={p_value_thresh:0.2f}", xy=(sequence_idx[-500], p_value_thresh + 0.02), color="gray", alpha=0.4)
    theta_null_str = r"$\theta_{null}$"
    title = f" {theta_null_str} = {success_rate_null:0.2f}"
    plt.title(title)
    plt.ylabel("p-value")
    plt.ylim(-0.1, 0.5)

    plt.tight_layout()