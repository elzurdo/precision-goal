import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils_stats import (CI_FRACTION,
                         successes_failures_to_hdi_ci_limits,
                         get_success_rates,
                         beta,
                         test_value
                         )

theta_null_str = r"$\theta_{null}$"

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

def plot_sequence_experiment_cumsum_average(sequence, success_rate_true=None, xlabel="trial no.", msize=5):
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
        plt.annotate(title, xy=(sequence_idx[-500], success_rate_true + 0.004), color="black", alpha=0.7)

        plt.ylim(success_rate_true - dsuccess_rate_plot, success_rate_true + dsuccess_rate_plot)

    plt.xlabel(xlabel)
    plt.ylabel("cumsum average")
    if title:
        plt.title(title)

def plot_sequence_experiment_nhst_combo_results(sequence, success_rate_true, success_rate_null, p_values, p_value_thresh=0.05, xlabel="trial no.", msize=5):

    sequence_idx = _get_sequence_idx(sequence)

    plt.subplot(2, 1, 1)
    plot_sequence_experiment_cumsum_average(sequence, success_rate_true=success_rate_true,xlabel=xlabel, msize=msize)

    plt.subplot(2, 1, 2)
    
    
    plt.hlines(p_value_thresh, sequence_idx[0], sequence_idx[-1], color="gray", linestyle='--', alpha=0.3)
    plt.scatter(sequence_idx[p_values >= p_value_thresh], p_values[p_values >= p_value_thresh], color = "gray", alpha=0.7, s=msize)
    plt.scatter(sequence_idx[p_values < p_value_thresh], p_values[p_values < p_value_thresh], color = "blue", marker='x', s=msize * 10)
    plt.xlabel(xlabel)
    plt.annotate(f"decision criterion p-value={p_value_thresh:0.2f}", xy=(sequence_idx[-500], p_value_thresh + 0.02), color="black", alpha=0.7)
    
    idx_reject = sequence_idx[p_values < p_value_thresh][0] # zurda
    print(sequence_idx[p_values < p_value_thresh])
    sequence_average = sequence.cumsum() / sequence_idx
    value_reject = sequence_average[p_values < p_value_thresh][0]

    idx_decision = idx_reject
    value_decision = value_reject

    plt.annotate(f'decision: reject at {idx_decision}', xy=(idx_decision, p_value_thresh),  xycoords='data', color='black',
            xytext=(0.5, 0.1), textcoords='axes fraction',
            arrowprops=dict(facecolor='green', shrink=0.05),
            horizontalalignment='right', verticalalignment='top', alpha=0.7
            )
    
    title = f" {theta_null_str} = {success_rate_null:0.2f}"
    plt.title(title)
    plt.ylabel("p-value")
    plt.ylim(-0.1, 0.5)

    plt.tight_layout()


def plot_sequence_experiment_hdi_rope_combo_results(sequence, success_rate_true, success_rate_null, ci_mins, ci_maxs, within_rope, rope_min, rope_max, xlabel="trial no.", msize=5):

    sequence_idx = _get_sequence_idx(sequence)

    plt.subplot(2, 1, 1)
    plot_sequence_experiment_cumsum_average(sequence, success_rate_true=success_rate_true,xlabel=xlabel, msize=msize)

    plt.subplot(2, 1, 2)
    sequence_average = sequence.cumsum() / sequence_idx
    lower_uncertainty = sequence_average - ci_mins
    upper_uncertainty = ci_maxs - sequence_average

    idx_accept = sequence_idx[within_rope][0]
    value_accept = sequence_average[within_rope][0]
    # TODO: compare to reject_higher, reject_lower to see which is the first decision in sequence
    idx_decision = idx_accept
    plt.errorbar(sequence_idx[within_rope], sequence_average[within_rope], yerr=(upper_uncertainty[within_rope], lower_uncertainty[within_rope]), color="green", alpha=0.3)
    plt.annotate(f'decision: accept at trial {idx_decision}', xy=(idx_decision, value_accept),  xycoords='data', color='black',
            xytext=(0.8, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='green', shrink=0.05),
            horizontalalignment='right', verticalalignment='top', alpha=0.7
            )
    
    reject_higher = ci_mins > rope_max
    plt.errorbar(sequence_idx[reject_higher], sequence_average[reject_higher], yerr=(upper_uncertainty[reject_higher], lower_uncertainty[reject_higher]), color="red", alpha=0.3)
    reject_lower = ci_maxs < rope_min
    plt.errorbar(sequence_idx[reject_lower], sequence_average[reject_lower], yerr=(upper_uncertainty[reject_lower], lower_uncertainty[reject_lower]), color="orange", alpha=0.3)
    inconclusive = ~(within_rope + reject_higher + reject_lower)
    plt.errorbar(sequence_idx[inconclusive], sequence_average[inconclusive], yerr=(upper_uncertainty[inconclusive], lower_uncertainty[inconclusive]), color="gray", alpha=0.3)

    plt.hlines(rope_min, sequence_idx[0], sequence_idx[-1], color="black", linestyle='--', alpha=0.5)
    plt.hlines(rope_max, sequence_idx[0], sequence_idx[-1], color="black", linestyle='--', alpha=0.5)
    plt.annotate(f"rope min={rope_min:0.2f}", xy=(sequence_idx[-500], rope_min - 0.04), color="black", alpha=0.7)
    plt.annotate(f"rope max={rope_max:0.2f}", xy=(sequence_idx[-500], rope_max + 0.02), color="black", alpha=0.7)
    plt.ylabel("cumsum average\nHDI 95% CI")
    plt.ylim(success_rate_true - 0.3, success_rate_true + 0.3)
    theta_null_str = r"$\theta_{null}$"
    title = f" {theta_null_str} = {success_rate_null:0.2f}"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()


def plot_sequence_experiment_pitg_combo_results(sequence, setup, results, xlabel="test no.", msize=5, conservative=False):

    sequence_idx = _get_sequence_idx(sequence)

    plt.subplot(2, 1, 1)
    plot_sequence_experiment_cumsum_average(sequence, success_rate_true=setup['success_rate'],xlabel=xlabel, msize=msize)

    plt.subplot(2, 1, 2)
    sequence_average = sequence.cumsum() / sequence_idx
    lower_uncertainty = sequence_average - results['ci_mins']
    upper_uncertainty = results['ci_maxs'] - sequence_average

    goal = results['precision_goal_achieved']
    #plt.errorbar(sequence_idx[goal], sequence_average[goal], yerr=(upper_uncertainty[goal], lower_uncertainty[goal]), color="purple", alpha=0.05)
    plt.errorbar(sequence_idx[~goal], sequence_average[~goal], yerr=(upper_uncertainty[~goal], lower_uncertainty[~goal]), color="gray", alpha=0.3)

    accept = goal & results['accept_within']
    plt.errorbar(sequence_idx[accept], sequence_average[accept], yerr=(upper_uncertainty[accept], lower_uncertainty[accept]), color="green", alpha=0.3)

    reject = goal & results['reject_outside'] 
    plt.errorbar(sequence_idx[reject], sequence_average[reject], yerr=(upper_uncertainty[reject], lower_uncertainty[reject]), color="red", alpha=0.3)

    inconclusive = goal & results['inconclusive_hdi_plus_rope'] 
    plt.errorbar(sequence_idx[inconclusive], sequence_average[inconclusive], yerr=(upper_uncertainty[inconclusive], lower_uncertainty[inconclusive]), color="black", alpha=0.3)

    idx_goal = sequence_idx[goal][0]
    value_goal= sequence_average[goal][0]
    # TODO: compare to reject_higher, reject_lower to see which is the first decision in sequence
    idx_decision = idx_goal
    plt.annotate(f'decision: accept at {idx_decision}', xy=(idx_decision, value_goal),  xycoords='data', color='purple',
            xytext=(0.8, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='purple', shrink=0.05),
            horizontalalignment='right', verticalalignment='top', alpha=0.7
            )    
    
    if conservative:
        idx_accept = sequence_idx[accept][0]
        value_accept = sequence_average[accept][0]
        # TODO: compare to reject_higher, reject_lower to see which is the first decision in sequence
        idx_decision = idx_accept
        plt.annotate(f'conservative decision: accept at {idx_decision}', xy=(idx_decision, value_accept),  xycoords='data', color='black',
                xytext=(0.8, 0.2), textcoords='axes fraction',
                arrowprops=dict(facecolor='green', shrink=0.05),
                horizontalalignment='right', verticalalignment='top', alpha=0.7
                )

    plt.hlines(setup['rope_min'], sequence_idx[0], sequence_idx[-1], color="black", linestyle='--', alpha=0.5)
    plt.hlines(setup['rope_max'], sequence_idx[0], sequence_idx[-1], color="black", linestyle='--', alpha=0.5)
    plt.annotate(f"rope min={setup['rope_min']:0.2f}", xy=(sequence_idx[-500], setup['rope_min'] - 0.04), color="black", alpha=0.7)
    plt.annotate(f"rope max={setup['rope_max']:0.2f}", xy=(sequence_idx[-500], setup['rope_max'] + 0.02), color="black", alpha=0.7)
    plt.ylabel("cumulative sum\nHDI 95% CI")
    plt.ylim(setup['success_rate'] - 0.3, setup['success_rate'] + 0.3)
    title = f" {theta_null_str} = {setup['success_rate_null']:0.2f}"
    plt.title(title)
    plt.tight_layout()


def plot_decision_rates_nhst(n_experiments, iteration_stopping_on_or_prior):
    msize = 5
    xlabel = "trial no."
    ylabel = f"decision rate at {xlabel} (or lower)"
    title = f"{n_experiments:,} experiments"
    theta_null_str = r"$\theta_{null}$"

    sr_iteration_stopping_on_or_prior = pd.Series(iteration_stopping_on_or_prior)
    sr_nhst_reject = sr_iteration_stopping_on_or_prior / n_experiments

    plt.plot(sr_nhst_reject.index, sr_nhst_reject + 0.01, alpha=0.7, color="red", linewidth=3, label=f"reject {theta_null_str}")
    plt.plot(sr_nhst_reject.index, 1. - sr_nhst_reject, alpha=0.7, color="gray", linewidth=3, linestyle='--', label="not reject / inconclusive")

    plt.legend()
    #plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def plot_decision_rates(n_experiments, df_decision_counts):
    linewidth = 3
    xlabel = "trial no."
    ylabel = f"decision rate at {xlabel} (or lower)"
    title = f"{n_experiments:,} experiments"
    theta_null_str = r"$\theta_{null}$"

    plt.plot(df_decision_counts.index, df_decision_counts['accept'] / n_experiments, color="green", label=f"accept {theta_null_str}", linewidth=linewidth, linestyle='-.')
    plt.plot(df_decision_counts.index, df_decision_counts['reject'] / n_experiments, color="red", label=f"reject {theta_null_str}", linewidth=linewidth * 0.8)
    plt.plot(df_decision_counts.index, df_decision_counts['inconclusive'] / n_experiments, color="gray", label="inconclusive", linewidth=linewidth * 0.6, linestyle='--')

    plt.legend()
    # plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plot_vhlines_lines(vertical=None, horizontal=0, color="black", ax=None, alpha=0.2, linestyle=None, linewidth=1, label=None):
    if ax is None:
        ax = plt.gca()

    if horizontal is not None:
	    ax.axhline(horizontal, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, label=label)
        
    if vertical is not None:
        ax.axvline(vertical, color=color, linewidth=linewidth, alpha=alpha,linestyle=linestyle, label=label)


def plot_parity_line(ax=None):
    if ax is None:
        ax = plt.gca()

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    lims = [np.min([xlims[0], ylims[0]]), np.max([xlims[1], ylims[1]])]

    ax.plot(lims, lims, "k--", linewidth=1)
