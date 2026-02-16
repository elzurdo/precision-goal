import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import display
from IPython.display import display

from utils_stats import (CI_FRACTION,
                         successes_failures_to_hdi_ci_limits,
                         get_success_rates,
                         beta,
                         test_value,
                         binomial_rate_ci_width_to_sample_size,
                         )

FIG_WIDTH = 8
FIG_HEIGHT = 6

theta_str = r"$\theta$"
theta_null_str = r"$\theta_{\rm null}$"
theta_true_str = r"$\theta_{\rm true}$"

ALGO_COLORS = {"pitg": "blue", "epitg": "lightgreen", "hdi_rope": "red"}

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
        # theta_true_str = r"$\theta_{\rm true}$"
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
    
    idx_reject = sequence_idx[p_values < p_value_thresh][0]
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
    # theta_null_str = r"$\theta_{\rm null}$"
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
    # theta_null_str = r"$\theta_{\rm null}$"

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
    #theta_null_str = r"$\theta_{null}$"

    plt.plot(df_decision_counts.index, df_decision_counts['accept'] / n_experiments, color="green", label=f"accept {theta_null_str}", linewidth=linewidth, linestyle='-.')
    plt.plot(df_decision_counts.index, df_decision_counts['reject'] / n_experiments, color="red", label=f"reject {theta_null_str}", linewidth=linewidth * 0.8)
    plt.plot(df_decision_counts.index, df_decision_counts['inconclusive'] / n_experiments, color="gray", label="inconclusive", linewidth=linewidth * 0.6, linestyle='--')

    plt.legend()
    # plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def plot_multiple_decision_rates_jammed(method_df_iteration_counts, success_rate, experiments, iteration_values=None):
    title = f"{theta_true_str} = {success_rate:0.3f}"
    xlabel = "iteration"

    method_alpha = {"pitg": 0.4, "epitg": 0.7, "hdi_rope": 0.2}
    method_linewidth = {"pitg": 5, "epitg": 3, "hdi_rope": 1}
    method_linestyle = {"pitg": "-", "epitg": "--", "hdi_rope": "-."}

    for method_name, df_counts in method_df_iteration_counts.items():
        if iteration_values is None:
            iteration_values = df_counts["iteration"]

        linewidth = method_linewidth[method_name]
        alpha = method_alpha[method_name]
        linestyle = method_linestyle[method_name]

        plt.plot(iteration_values, df_counts['accept'] / experiments, color="green", linewidth=linewidth, alpha=alpha, linestyle=linestyle)
        plt.plot(iteration_values, df_counts['reject'] / experiments, color="red", linewidth=linewidth, alpha=alpha, linestyle=linestyle)
        plt.plot(iteration_values, df_counts['inconclusive'] / experiments, color="gray", linewidth=linewidth, alpha=alpha, linestyle=linestyle)

    plt.xlabel(xlabel)
    plt.ylabel(f"proportion of {experiments:,} experiments")
    plt.title(title)

def plot_multiple_decision_rates_separate(method_df_iteration_counts, success_rate, experiments, viz_epitg="separate", iteration_values=None):
    print("viz_epitg", viz_epitg)
    plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT))
    xlabel = "iteration"

    if success_rate is not None:
        suptitle = f"{theta_true_str} = {success_rate:0.3f}"
    else:
        suptitle = None

    for method_name, df_counts in method_df_iteration_counts.items():
        if iteration_values is None:
            iteration_values = df_counts["iteration"]

        linestyle_accept, linewidth_accept = None, 5
        linestyle_reject, linewidth_reject = "--", 3
        linestyle_inconclusive, linewidth_inconclusive = "-.", 1
        alpha=0.7
        label_accept = "accept"
        label_reject = "reject"
        label_inconclusive = "inconclusive/\ncollect more"
        
        if "hdi_rope" == method_name:
            if viz_epitg == "together":
                plt.subplot(1, 2, 1)
            elif viz_epitg == "separate":
                plt.subplot(1, 3, 1)
            else:
                plt.subplot(1, 2, 1)
            title = "HDI + ROPE"
        else:
            if viz_epitg == "together":
                plt.subplot(1, 2, 2)
                if "pitg" == method_name:
                    title = "Precision is the Goal (thin), Enhanced (thick)"
                if "epitg" == method_name:
                    linewidth_accept, linewidth_reject, linewidth_inconclusive = 6, 6, 6
                    alpha = 0.3
                    label_accept, label_reject, label_inconclusive = None, None, None
            elif viz_epitg == "separate":
                if "pitg" == method_name:
                    plt.subplot(1, 3, 2)
                    title = "Precision is the Goal"
                elif "epitg" == method_name:
                    plt.subplot(1, 3, 3)
                    title = "Enhanced Precision is the Goal"
            else:
                if "pitg" == method_name:
                    plt.subplot(1, 2, 1)
                    title = "Precision is the Goal"

        # plotting HDI+ROPE
        plt.plot(iteration_values, df_counts['accept'] / experiments, color="green", linewidth=linewidth_accept, alpha=alpha, linestyle=linestyle_accept, label=label_accept)
        plt.plot(iteration_values, df_counts['reject'] / experiments, color="red", linewidth=linewidth_reject, alpha=alpha, linestyle=linestyle_reject, label=label_reject)
        plt.plot(iteration_values, df_counts['inconclusive'] / experiments, color="gray", linewidth=linewidth_inconclusive, alpha=alpha, linestyle=linestyle_inconclusive, label=label_inconclusive)

        plot_grid(with_y=True, with_x=False, alpha=0.3)
        plt.legend(title="decision")
        plt.xlabel(xlabel)
        plt.ylabel(f"proportion of {experiments:,} experiments")
        plt.title(title)


    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=20)
    plt.tight_layout()


method_pretty_short_name = {
    "pitg": "PitG",
    "epitg": "EPitG",
    "hdi_rope": "HDI+ROPE"
}

def scatter_stop_iter_sample_rate_alt(method_df_stats, rope_min=None, rope_max=None, success_rate_true=None, success_rate_hypothesis=None, precision_goal=None, title=None, method_names=None):
    method_colors = {"pitg": "blue", "epitg": "lightgreen", "hdi_rope": "red"}
    method_markers = {"pitg": "o", "epitg": "x", "hdi_rope": "s"}
    method_mean_markers = {"pitg": "$\u25EF$", "epitg": "x", "hdi_rope": "$\u25A1$"}

    if method_names is None:
        method_names = ["hdi_rope", "pitg", "epitg"]

    for method_name in method_names:
        df_stats = method_df_stats[method_name].copy()
        color, marker = method_colors[method_name], method_markers[method_name]
        mean_marker = method_mean_markers[method_name]
        label = method_pretty_short_name[method_name]
        label_mean = f"{method_pretty_short_name[method_name]} mean"

        plt.scatter(df_stats["decision_iteration"], df_stats["success_rate"], alpha=0.3, color=color, label=label, marker=marker, s=20)
        plt.scatter(df_stats["decision_iteration"].mean(), df_stats["success_rate"].mean(), color=color, label=label_mean, s=200, marker=mean_marker)



    if success_rate_true is not None:
        plot_vhlines_lines(vertical=None, label=f'{theta_true_str}', horizontal=success_rate_true, alpha=0.7)

    if rope_min is not None:
        plot_vhlines_lines(vertical=None, label='ROPE', horizontal=rope_min, linestyle="--")
    if rope_max is not None:
        plot_vhlines_lines(vertical=None, horizontal=rope_max, linestyle="--")


    if precision_goal is not None:
        # adding horizontal lines for expected N in which precision goal is achieved
        # This might be different for true and hypothesis success rates
        n_true_str = r"$N_{\theta_\mathrm{true}}$"
        n_hypo_str = r"$N_{\theta_0}$"
        n_precision_goal_true, n_precision_goal_hypothesis = None, None
        if (success_rate_true is not None):
            n_precision_goal_true = binomial_rate_ci_width_to_sample_size(success_rate_true, precision_goal)
        if (success_rate_hypothesis is not None):
            n_precision_goal_hypothesis = binomial_rate_ci_width_to_sample_size(success_rate_hypothesis, precision_goal) 

        if (n_precision_goal_true == n_precision_goal_hypothesis) and (n_precision_goal_true is not None):
            label_n = f"{n_true_str}={n_hypo_str}={n_precision_goal_true:0.1f}"
            plt.axvline(n_precision_goal_true, color='gray', linestyle=':', label=label_n)
        else:
            label_n_true = f"{n_true_str}={n_precision_goal_true:0.1f}"
            laben_n_hypo = f"{n_hypo_str}={n_precision_goal_hypothesis:0.1f}"
            plt.axvline(n_precision_goal_true, color='gray', linestyle=':', label=label_n_true)
            plt.axvline(n_precision_goal_hypothesis, color='gray', linestyle='--', label=laben_n_hypo)


    plt.xlabel("stop iteration")
    theta_hat_str = r"$\hat{\theta}$"
    plt.ylabel(f"success rate at stop {theta_hat_str}")
    plt.legend(title=f"{len(df_stats):,} experiments", loc="upper right", fontsize=10)
    if title is not None:
        plt.title(title)

    #plt.xlim(400, 800)
    #plt.ylim(0.4, 0.6)

def scatter_stop_iter_sample_rate(method_df_stats, rope_min=None, rope_max=None, 
                                          success_rate_true=None, success_rate_hypothesis=None, 
                                          precision_goal=None, title=None, method_names=None,
                                          scatter_ratio=3, bins=30, imbalance_cutoff_ratio=3.0):
    """
    Creates a 3-panel plot: 
    - Main scatter plot (Top-Right)
    - Success rate histogram (Left, sharing y-axis)
    - Stop iteration histogram (Bottom, sharing x-axis)
    
    imbalance_cutoff_ratio: if the max density of one distribution > X * peak of another,
                            limit the view to help see the smaller one.
    """
    import matplotlib.gridspec as gridspec

    method_markers = {"pitg": "o", "epitg": "x", "hdi_rope": "s"}
    method_mean_markers = {"pitg": "$\u25EF$", "epitg": "x", "hdi_rope": "$\u25A1$"}

    if method_names is None:
        method_names = ["hdi_rope", "pitg", "epitg"]

    if success_rate_true:
        #theta_true_str = r"$\theta_{\rm true}$"
        title = f" {theta_true_str} = {success_rate_true:0.2f}"
    else:
        title = ""

    fig = plt.figure(figsize=(FIG_WIDTH * 1.5, FIG_HEIGHT * 1.5))
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, scatter_ratio], height_ratios=[scatter_ratio, 1], 
                           wspace=0.05, hspace=0.05)

    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_scatter)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_scatter)

    last_df_len = 0
    
    iteration_max_densities = []
    success_max_densities = []

    for method_name in method_names:
        if method_name not in method_df_stats:
            continue
        
        df_stats = method_df_stats[method_name].copy()
        last_df_len = len(df_stats)
        
        color, marker = ALGO_COLORS[method_name], method_markers[method_name]
        mean_marker = method_mean_markers[method_name]
        label = method_pretty_short_name[method_name]
        label_mean = f"{method_pretty_short_name[method_name]} mean"

        # --- Main Panel: Scatter ---
        ax_scatter.scatter(df_stats["decision_iteration"], df_stats["success_rate"], 
                           alpha=0.3, color=color, label=label, marker=marker, s=20)
        
        # --- Bottom Panel: Iteration Histogram ---
        # Calculate histogram first to capture densities
        iter_counts, iter_bins = np.histogram(df_stats["decision_iteration"], bins=bins, density=True)
        iteration_max_densities.append(np.max(iter_counts))
        
        ax_bottom.hist(df_stats["decision_iteration"], bins=bins, color=color, alpha=0.3, 
                       density=True, histtype='stepfilled')
        ax_bottom.hist(df_stats["decision_iteration"], bins=bins, color=color, alpha=0.8, 
                       density=True, histtype='step', linewidth=1.5)

        # --- Left Panel: Success Rate Histogram ---
        success_counts, success_bins = np.histogram(df_stats["success_rate"], bins=bins, density=True)
        success_max_densities.append(np.max(success_counts))

        ax_left.hist(df_stats["success_rate"], bins=bins, color=color, alpha=0.3, 
                     density=True, histtype='stepfilled', orientation='horizontal')
        ax_left.hist(df_stats["success_rate"], bins=bins, color=color, alpha=0.8, 
                     density=True, histtype='step', orientation='horizontal', linewidth=1.5)
    
    # --- Auto-Scaling Logic ---
    def get_limit(max_densities, ratio):
        if not max_densities: return None
        sorted_max = sorted(max_densities)
        if len(sorted_max) > 1:
            # Check if largest is outlier compared to second largest
            if sorted_max[-1] > ratio * sorted_max[-2]:
                return sorted_max[-2] * 1.5 # Show the second largest comfortably
        return None # Default scaling

    ylim_bottom = get_limit(iteration_max_densities, imbalance_cutoff_ratio)
    xlim_left = get_limit(success_max_densities, imbalance_cutoff_ratio)

    if ylim_bottom:
        ax_bottom.set_ylim(ylim_bottom, 0) # Inverted
    else:
        ax_bottom.invert_yaxis()

    if xlim_left:
        ax_left.set_xlim(xlim_left, 0) # Inverted
    else:
        ax_left.invert_xaxis()


    # --- Decorate Main Scatter Panel ---
    if success_rate_true is not None:
        plot_vhlines_lines(vertical=None, label=f'{theta_true_str}', horizontal=success_rate_true, alpha=0.7, ax=ax_scatter)

    if rope_min is not None:
        plot_vhlines_lines(vertical=None, label='ROPE', horizontal=rope_min, linestyle="--", ax=ax_scatter)
    if rope_max is not None:
        plot_vhlines_lines(vertical=None, horizontal=rope_max, linestyle="--", ax=ax_scatter)

    if precision_goal is not None:
        n_true_str = r"$N_{\theta_\mathrm{true}}$"
        n_hypo_str = r"$N_{\theta_\mathrm{null}}$"
        n_precision_goal_true, n_precision_goal_hypothesis = None, None
        if (success_rate_true is not None):
            n_precision_goal_true = binomial_rate_ci_width_to_sample_size(success_rate_true, precision_goal)
        if (success_rate_hypothesis is not None):
            n_precision_goal_hypothesis = binomial_rate_ci_width_to_sample_size(success_rate_hypothesis, precision_goal) 

        if (n_precision_goal_true == n_precision_goal_hypothesis) and (n_precision_goal_true is not None):
            label_n = f"{n_true_str}={n_hypo_str}={n_precision_goal_true:0.1f}"
            ax_scatter.axvline(n_precision_goal_true, color='gray', linestyle=':', label=label_n)
        else:
            if n_precision_goal_true:
                label_n_true = f"{n_true_str}={n_precision_goal_true:0.1f}"
                ax_scatter.axvline(n_precision_goal_true, color='gray', linestyle=':', label=label_n_true)
            if n_precision_goal_hypothesis:
                laben_n_hypo = f"{n_hypo_str}={n_precision_goal_hypothesis:0.1f}"
                ax_scatter.axvline(n_precision_goal_hypothesis, color='gray', linestyle='--', label=laben_n_hypo)

    # --- Labels & Legends ---
    
    # Hide ticks between panels
    plt.setp(ax_scatter.get_xticklabels(), visible=False)
    plt.setp(ax_scatter.get_yticklabels(), visible=False)

    # Bottom panel labels
    ax_bottom.set_xlabel("stop iteration")
    ax_bottom.set_ylabel("density")

    # Invert to have 0 near the scatter plot
    ax_bottom.invert_yaxis() 
    ax_left.invert_xaxis()   

    # Hide 0 label on bottom panel y-axis to avoid clash with left panel x-axis
    from matplotlib.ticker import FuncFormatter
    ax_bottom.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "" if np.isclose(x, 0) else f"{x:g}"))

    # Left panel labels
    theta_hat_str = r"$\hat{\theta}$"
    ax_left.set_ylabel(f"success rate at stop {theta_hat_str}")
    ax_left.set_xlabel("density")

    # Legend on Scatter
    ax_scatter.legend(title=None, loc="upper right", fontsize=10)

    # if title is not None:
    #     # Adjust title position to not overlap with top-left empty space if needed
    #     # but suptitle usually handles it well.
    title += f" ({last_df_len:,} experiments)"
    plt.suptitle(title, y=0.95)

    return fig

def viz_one_sample_results(df_sample_results, precision_goal, rope_min, rope_max, success_rate=None):
    df_conclusive_accept = df_sample_results.query("conclusive").query("accept")
    df_conclusive_reject = df_sample_results.query("conclusive").query("reject")
    df_sample_goal = df_sample_results.query("goal_achieved")

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    plt.plot(df_sample_results["decision_iteration"], df_sample_results["hdi_min"], color="gray", label=None)
    plt.plot(df_sample_results["decision_iteration"], df_sample_results["hdi_max"], color="gray", label=None)
    plt.fill_between(df_sample_results["decision_iteration"], df_sample_results["hdi_max"], df_sample_results["hdi_min"], color='gray', alpha=0.2, label="HDI")

    # experiments which are conclusive to accept null hypothesis
    for idx, (iteration, row) in enumerate(df_conclusive_accept.iterrows()):
        if idx == 0:
            label = "conclusive: accept"
        else:
            label = None
        plt.plot([iteration, iteration], [row['hdi_min'], row['hdi_max']], color='lightgreen', alpha=0.7, linewidth=1, label=label)

    # experiments which are conclusive to reject null hypothesis
    for idx, (iteration, row) in enumerate(df_conclusive_reject.iterrows()):
        if idx == 0:
            label = "conclusive: reject"
        else:
            label = None
        plt.plot([iteration, iteration], [row['hdi_min'], row['hdi_max']], color='red', alpha=0.7, linewidth=1, label=label, linestyle=":")

    #for iteration, row in df_sample_goal.iterrows():
    #    plt.plot([iteration, iteration], [row['hdi_min'], row['hdi_max']], color='blue', alpha=0.1, linewidth=1)
    plt.scatter(df_sample_goal["decision_iteration"], df_sample_goal["hdi_min"], color="purple", label=f"{precision_goal:0.3} goal achieved", marker="o", s=20)
    plt.scatter(df_sample_goal["decision_iteration"], df_sample_goal["hdi_max"], color="purple", label=None, marker="o", s=20)

    plot_vhlines_lines(vertical=None, label='ROPE', horizontal=rope_min, linestyle="--", color="purple")
    plot_vhlines_lines(vertical=None, horizontal=rope_max, linestyle="--", color="purple")

    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel(f"success rate {theta_str}")

    if success_rate is not None:
        plt.title(f"{theta_true_str}={success_rate:0.3f}")


def plot_pdf(sr_experiment_stats, rope_min, rope_max, xlim=None, xtitle=r"success rate $\theta$"):
    pp = np.linspace(0, 1, 1000)
    pp_hdi = np.linspace(sr_experiment_stats["hdi_min"], sr_experiment_stats["hdi_max"], 1000)

    successes = sr_experiment_stats["successes"]
    failures = sr_experiment_stats["failures"]
    rate = successes / (successes + failures)
    n_ = successes + failures

    hdi_min, hdi_max = successes_failures_to_hdi_ci_limits(successes, failures)

    pdf = beta.pdf(pp, successes, failures)
    pdf_hdi = beta.pdf(pp_hdi, successes, failures)

    theta_hat_str = r"$\hat{\theta}$"
    plt.plot(pp, pdf, color="purple", label=f"pdf {theta_hat_str}={rate:0.3f}; n={n_:,}")
    label_hdi = f"95% HDI: {hdi_max - hdi_min:0.3f}"
    plt.fill_between(pp_hdi, pdf_hdi, color="purple", alpha=0.2, label=label_hdi)
    plot_vhlines_lines(vertical=rope_min, label='ROPE', horizontal=None, linestyle="--")
    plot_vhlines_lines(vertical=rope_max, horizontal=None, linestyle="--")
    plt.legend()

    if xtitle is not None:
        plt.xlabel(xtitle)
    plt.ylabel(r"$p(\theta$)")

    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([rope_min - 0.1, rope_max + 0.1])

METHOD_FULL = {
    "hdi_rope": "HDI + ROPE",
    "pitg": "Precision is the Goal",
    "epitg": "Enhance Precision is the Goal",

}

METHOD_SHORT = {
    "hdi_rope": "HDI+ROPE",
    "pitg": "PitG",
    "epitg": "ePitG",

}

def plot_sample_pdf_methods(method_df_stats, isample, rope_min, rope_max, xlim = (0.2, 0.6), method_names=None):

    if method_names is None:
        method_names = list(method_df_stats.keys())

    ncols, nrows = 1, len(method_names)

    plt.subplots(nrows, ncols, figsize=(FIG_WIDTH, 1.2* FIG_HEIGHT))

    for imethod, method_name in enumerate(method_names):
        experiment_stats = method_df_stats[method_name].loc[isample]

        plt.subplot(nrows, ncols, imethod + 1)

        if imethod == len(method_names) - 1:
            xtitle = r"success rate $\theta$"
        else:
            xtitle = None
        plot_pdf(experiment_stats, rope_min, rope_max, xlim=xlim, xtitle=xtitle)
        plt.title(f"{METHOD_FULL[method_name]}")

    plt.suptitle(f"Outcomes depending on Stop Criterion", fontsize=18)
    plt.tight_layout()


def plot_grid(with_y=True, with_x=False, alpha=0.3):
    ax = plt.gca()

    if with_y:
        ax.grid(axis="y", alpha=alpha)
    else:
        ax.grid(False, axis="y")

    if with_x:
        ax.grid(axis="x", alpha=alpha)
    else:
        ax.grid(False, axis="x")
        

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


ALGO_HATCH = {
    "hdi_rope": None,
    "pitg": "/",
    "epitg": "\\"
}

def plot_success_by_truth(algo_stats_df, dsuccess_rate, subset_name = "conclusive", success_null=0.5):

    assert subset_name in ["conclusive", "inconclusive", "overall"]

    if  "conclusive" == subset_name:
        title = "Conclusive"
    elif "overall" == subset_name:
        title = "Conclusive + Inconclusive"
    elif "inconclusive" == subset_name:
        title = "Inconclusive"

    truth_values = np.array(algo_stats_df[subset_name]["epitg"]["success_median"].index.tolist())

    rope_min = success_null - dsuccess_rate
    rope_max = success_null + dsuccess_rate

    algo_alpha = {
       "hdi_rope": 0.2,
        "pitg": 0.5,
        "epitg": 0.5     
    }

    plt.title(title, fontsize=20)
    for algo_name in METHOD_SHORT.keys():
        #this_truths = algo_stats_df[subset_name][algo_name].query("success_p25 == success_p25").index.tolist()
        this_truths = algo_stats_df[subset_name][algo_name].query("count >= 20").index.tolist()

        label = f"{METHOD_SHORT[algo_name]}"

        try:
            plt.fill_between(
                this_truths, 
                algo_stats_df[subset_name][algo_name].loc[this_truths, "success_p25"].astype(float),
                algo_stats_df[subset_name][algo_name].loc[this_truths,"success_p75"].astype(float),
                color=ALGO_COLORS[algo_name], 
                alpha=algo_alpha[algo_name], 
                label=label,
                hatch=ALGO_HATCH[algo_name]
            )
            
        except Exception as e:
            print(f"Error plotting {algo_name}: {e}")
            df_aux = algo_stats_df[subset_name][algo_name].loc[this_truths]
            try:
                df_aux["diff_pcnt"] = (df_aux["success_p75"] - df_aux["success_p25"]) * 100.
                display(df_aux[["count", "stop_iter_median" ,"success_p25", "success_median","success_p75","diff_pcnt" ]])
            except:
                pass

        plt.plot(this_truths, algo_stats_df[subset_name][algo_name].loc[this_truths, "success_p25"], color=ALGO_COLORS[algo_name], alpha=1.)
        plt.plot(this_truths, algo_stats_df[subset_name][algo_name].loc[this_truths, "success_p75"], color=ALGO_COLORS[algo_name], alpha=1.)


    #plt.fill_between(truth_values, algo_stats_df[subset_name]["hdi_rope"]["success_p25"], algo_stats_df[subset_name]["hdi_rope"]["success_p75"], color=ALGO_COLORS["hdi_rope"], alpha=0.2, label=f"{METHOD_SHORT['hdi_rope']}")
    #plt.fill_between(truth_values, algo_stats_df[subset_name]["pitg"]["success_p25"], algo_stats_df[subset_name]["pitg"]["success_p75"], color=ALGO_COLORS["pitg"], alpha=0.5, label=f"{METHOD_SHORT['pitg']}", hatch="/")
    #plt.fill_between(truth_values, algo_stats_df[subset_name]["epitg"]["success_p25"], algo_stats_df[subset_name]["epitg"]["success_p75"], color=ALGO_COLORS["epitg"], alpha=0.5, label=f"{METHOD_SHORT['epitg']}", hatch="\\")


    plt.axhline(rope_min, linestyle=":", color="gray")
    plt.axhline(rope_max, linestyle=":", color="gray")
    #plt.plot(truth_values, rope_mins, color="gray", linestyle=":")
    #plt.plot(truth_values, rope_maxs, color="gray", linestyle=":")
    plt.plot(truth_values, truth_values, color="gray", linestyle=None, alpha=1)
    plt.axvline(x=0.5 + dsuccess_rate, color="black", linestyle="--", alpha=0.5)

    plt.xlabel(r"$\theta_{\rm true}$")
    plt.ylabel(r"$\hat{\theta}$")
    plt.legend(title="IQR")


    plt.grid(alpha=0.3)
    plt.ylim(0.4, 0.75)


ALGO_LINEWIDTH =  {"hdi_rope":1, "pitg": 2, "epitg":3}

def plot_success_by_truth_diff(algo_stats_df, dsuccess_rate, subset_name="conclusive", success_metrics=["success_median"]):
    METRIC_LINESTYLE = {"success_median": None, "success_mean": "--"}

    truth_values = np.array(algo_stats_df[subset_name]["epitg"]["success_median"].index.tolist())

    for success_metric in success_metrics:
        for algo_name in METHOD_SHORT.keys():
            result_diff = algo_stats_df[subset_name][algo_name][success_metric] - truth_values

            label = f"{METHOD_SHORT[algo_name]}"
            plt.plot(truth_values, result_diff, color=ALGO_COLORS[algo_name], linestyle=METRIC_LINESTYLE[success_metric], alpha=0.7, label=label, linewidth=ALGO_LINEWIDTH[algo_name])


    # hdirope_median_diff = algo_stats_df[subset_name]["hdi_rope"]["success_median"] - xvalues
    # hdirope_mean_diff = algo_stats_df[subset_name]["hdi_rope"]["success_mean"] - xvalues
    # epitg_median_diff = algo_stats_df[subset_name]["epitg"]["success_median"] - xvalues
    # pit_median_diff = algo_stats_df[subset_name]["pitg"]["success_median"] - xvalues

    # epitg_mean_diff = algo_stats_df[subset_name]["epitg"]["success_mean"] - xvalues
    # pit_mean_diff = algo_stats_df[subset_name]["pitg"]["success_mean"] - xvalues

    
    # plt.plot(xvalues, hdirope_median_diff, color=ALGO_COLORS["hdi_rope"], linestyle=None, alpha=0.7, label=f"{ALGO_NAME['hdi_rope']} median")

    # plt.plot(xvalues, pit_mean_diff, color=ALGO_COLORS["pitg"], linestyle="--", alpha=0.4, label=f"{ALGO_NAME['pitg']} mean", linewidth=ALGO_LINEWIDTH["pitg"])
    # plt.plot(xvalues, pit_median_diff, color=ALGO_COLORS["pitg"], linestyle=None, alpha=0.7, label=f"{ALGO_NAME['pitg']} median", linewidth=ALGO_LINEWIDTH["pitg"])

    # plt.plot(xvalues, epitg_mean_diff, color=ALGO_COLORS["epitg"], linestyle="--", alpha=0.4, label=f"{ALGO_NAME['epitg']} mean", linewidth=ALGO_LINEWIDTH["epitg"])
    # plt.plot(xvalues, epitg_median_diff, color=ALGO_COLORS["epitg"], linestyle=None, alpha=0.7, label=f"{ALGO_NAME['epitg']} median", linewidth=ALGO_LINEWIDTH["epitg"])

    plt.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    plt.xlabel(r"$\theta_{\rm true}$")
    plt.ylabel(r"$\hat{\theta} - \theta_{\rm true}$")
    plt.legend(title="median - true")
    plt.axvline(x=0.5 + dsuccess_rate, color="black", linestyle="--", alpha=0.5)


    plt.axhline(-dsuccess_rate, linestyle=":", color="gray")
    plt.axhline(dsuccess_rate, linestyle=":", color="gray")
    #plt.ylim(-dsuccess_rate,dsuccess_rate)
    plt.ylim(-0.1, 0.1)

    plt.grid(alpha=0.3)