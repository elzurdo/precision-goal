"""
Visualization utilities for continuous hypothesis testing.
"""
import numpy as np
import matplotlib.pyplot as plt

FIG_WIDTH = 8
FIG_HEIGHT = 6

mu_str = r"$\mu$"
mu_true_str = r"$\mu_{\rm true}$"
mu_hat_str = r"$\hat{\mu}$"


def viz_one_sample_results_continuous(df_sample_results, precision_goal, rope_min, rope_max, 
                                      true_mean=None, show_sample_mean=True, ylim=None):
    """
    Visualize one continuous experiment's evolution through all iterations.
    
    Shows how HDI bounds evolve and when precision/conclusiveness criteria are met.
    
    Parameters:
    -----------
    df_sample_results : pd.DataFrame
        Results from continuous_sample_all_iterations_results()
        Must contain: decision_iteration, hdi_min, hdi_max, sample_mean, 
                      conclusive, accept, reject, goal_achieved
    precision_goal : float
        Target precision (HDI width)
    rope_min : float
        Lower ROPE boundary
    rope_max : float
        Upper ROPE boundary
    true_mean : float, optional
        True population mean for reference line
    show_sample_mean : bool, optional
        Whether to plot the sample mean evolution (default True)
    """
    df_conclusive_accept = df_sample_results.query("conclusive").query("accept")
    df_conclusive_reject = df_sample_results.query("conclusive").query("reject")
    df_sample_goal = df_sample_results.query("goal_achieved")

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Plot HDI bounds as gray band
    plt.plot(df_sample_results["decision_iteration"], df_sample_results["hdi_min"], 
             color="gray", label=None, linewidth=1)
    plt.plot(df_sample_results["decision_iteration"], df_sample_results["hdi_max"], 
             color="gray", label=None, linewidth=1)
    plt.fill_between(df_sample_results["decision_iteration"], 
                     df_sample_results["hdi_max"], 
                     df_sample_results["hdi_min"], 
                     color='gray', alpha=0.2, label="HDI")

    # Optionally plot sample mean evolution
    if show_sample_mean:
        plt.plot(df_sample_results["decision_iteration"], 
                df_sample_results["sample_mean"], 
                color="black", alpha=0.4, linewidth=1.5, linestyle="-", 
                label=f"sample mean {mu_hat_str}")

    # Highlight conclusive accept iterations
    for idx, (iteration, row) in enumerate(df_conclusive_accept.iterrows()):
        label = "conclusive accept" if idx == 0 else None
        plt.plot([iteration, iteration], [row['hdi_min'], row['hdi_max']], 
                color='green', alpha=0.4, linewidth=2, label=label)

    # Highlight conclusive reject iterations
    for idx, (iteration, row) in enumerate(df_conclusive_reject.iterrows()):
        label = "conclusive reject" if idx == 0 else None
        plt.plot([iteration, iteration], [row['hdi_min'], row['hdi_max']], 
                color='red', alpha=0.4, linewidth=2, label=label)

    # Mark precision goal achieved iterations
    plt.scatter(df_sample_goal["decision_iteration"], df_sample_goal["hdi_min"], 
               color="purple", label=f"{precision_goal:0.3} goal achieved", marker="o", s=20)
    plt.scatter(df_sample_goal["decision_iteration"], df_sample_goal["hdi_max"], 
               color="purple", label=None, marker="o", s=20)

    # Plot ROPE boundaries
    plt.axhline(rope_min, linestyle="--", color="purple", alpha=0.5, label='ROPE')
    plt.axhline(rope_max, linestyle="--", color="purple", alpha=0.5)

    # Plot true mean reference line if provided
    if true_mean is not None:
        plt.axhline(true_mean, linestyle=":", color="black", alpha=0.3, 
                   label=f"{mu_true_str}={true_mean:0.3f}")

    plt.legend(loc="best")
    plt.xlabel("iteration")
    plt.ylabel(f"mean {mu_str}")
    plt.grid(alpha=0.3)
    plt.ylim(ylim)
    plt.tight_layout()
