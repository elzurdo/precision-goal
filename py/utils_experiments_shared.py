"""
Shared utilities for both binomial and continuous hypothesis testing experiments.
"""
import numpy as np
import pandas as pd
from IPython.display import display


def stats_dict_to_df(method_stats, data_type='binomial'):
    """
    Convert method statistics dictionary to DataFrame.
    
    Generic function that handles both binomial and continuous data types.
    
    Parameters:
    -----------
    method_stats : dict
        Dictionary of experiment statistics
    data_type : str
        'binomial' or 'continuous'
    
    Returns:
    --------
    pd.DataFrame
        Statistics organized by experiment number with computed derivatives
        
    Notes:
    ------
    Shared columns (both types):
    - hdi_max, hdi_min, decision_iteration
    - reject (sum of reject_below and reject_above)
    - precision (HDI width)
    
    Binomial-specific columns:
    - success_rate: estimated parameter (successes / total)
    
    Continuous-specific columns:
    - sample_mean: estimated parameter
    - sample_std: sample standard deviation
    - n: sample size
    - se: standard error (sample_std / sqrt(n))
    - coefficient_of_variation: relative std (sample_std / |sample_mean|)
    - relative_precision: relative HDI width (precision / |sample_mean|)
    """
    df = pd.DataFrame(method_stats).T
    df.index.name = "experiment_number"
    df["hdi_max"] = df["hdi_max"].astype(float)
    df["hdi_min"] = df["hdi_min"].astype(float)
    df["decision_iteration"] = df["decision_iteration"].astype(float)
    df["reject"] = df["reject_below"] + df["reject_above"]
    df["precision"] = df["hdi_max"] - df["hdi_min"]
    
    if data_type == 'binomial':
        df["success_rate"] = df["successes"] / (df["successes"] + df["failures"])
    elif data_type == 'continuous':
        # Ensure proper types
        df["sample_mean"] = df["sample_mean"].astype(float)
        df["sample_std"] = df["sample_std"].astype(float)
        df["n"] = df["n"].astype(int)
        
        # Add continuous-specific derivatives
        df["se"] = df["sample_std"] / np.sqrt(df["n"])
        df["coefficient_of_variation"] = df["sample_std"] / df["sample_mean"].abs()
        df["relative_precision"] = df["precision"] / df["sample_mean"].abs()
    else:
        raise ValueError(f"Unknown data_type '{data_type}'. Choose 'binomial' or 'continuous'.")
    
    return df


def iteration_counts_to_df(roperesult_iteration, experiments):
    """
    Convert ROPE result iteration counts to DataFrame.
    
    This function is generic and works for both binomial and continuous data.
    
    Parameters:
    -----------
    roperesult_iteration : dict
        Nested dict with ROPE results ('within', 'below', 'above') 
        and iteration counts
    experiments : int
        Total number of experiments
    
    Returns:
    --------
    pd.DataFrame
        Decision counts by iteration
    """
    # TODO: Examine for generalisability - already generic!
    df = pd.DataFrame({
        "iteration": list(roperesult_iteration["within"].keys()),
        "accept": list(roperesult_iteration["within"].values()),
        "reject_below": list(roperesult_iteration["below"].values()),
        "reject_above": list(roperesult_iteration["above"].values())
    })

    df['reject'] = df['reject_above'] + df['reject_below']
    df['inconclusive'] = experiments - df['accept'] - df['reject']

    return df


def report_success_rates(df_stats, data_type='binomial'):
    """
    Computes summary statistics for estimated parameter across different decision subgroups.
    
    TODO: Consider renaming to `report_parameter_stats` or `report_metric_rates` 
          to better reflect its generic nature.
    
    Parameters:
    -----------
    df_stats : pd.DataFrame
        Statistics DataFrame from stats_dict_to_df
    data_type : str
        'binomial' (uses 'success_rate' column) or 'continuous' (uses 'sample_mean' column)
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics by subgroup (overall, conclusive, inconclusive, accept, reject)
        
    Notes:
    ------
    Output columns use "param_" prefix for the estimated parameter:
    - param_mean, param_std, param_p25, param_median, param_p75
    """
    # Determine which column contains the estimated parameter
    param_col = 'success_rate' if data_type == 'binomial' else 'sample_mean'
    
    subgroups = {
        "overall": df_stats,
        "conclusive": df_stats.query("conclusive"),
        "inconclusive": df_stats.query("inconclusive"),
        "accept": df_stats.query("accept"),
        "reject": df_stats.query("reject")
    }

    records = []
    for group_name, df_group in subgroups.items():
        if len(df_group) == 0:
            continue

        sr_param = df_group[param_col]
        sr_stop_iter = df_group['decision_iteration']
        sr_conclusive = df_group['conclusive'].astype(int)  # convert boolean to int for stats
        sr_accept = df_group['accept'].astype(int)  # convert boolean to int for stats
        sr_reject = df_group['reject'].astype(int)  # convert boolean to int for stats

        records.append({
            "group": group_name,
            "count": int(sr_param.count()),
            "param_frac": sr_param.count() / len(df_stats),
            # estimated parameter statistics
            "param_mean": sr_param.mean(),
            "param_std": sr_param.std(),
            "param_p25": sr_param.quantile(0.25),
            "param_median": sr_param.median(),
            "param_p75": sr_param.quantile(0.75),
            # stop iteration statistics
            "stop_iter_mean": sr_stop_iter.mean(),
            "stop_iter_std": sr_stop_iter.std(),
            "stop_iter_p25": sr_stop_iter.quantile(0.25),
            "stop_iter_median": sr_stop_iter.median(),
            "stop_iter_p75": sr_stop_iter.quantile(0.75),
            # conclusive statistics
            "conclusive_mean": sr_conclusive.mean(),
            # accept/reject statistics
            "accept_mean": sr_accept.mean(),
            "reject_mean": sr_reject.mean()
        })

    return pd.DataFrame(records).set_index("group")


def report_success_rates_multiple_algos(method_df_stats, data_type='binomial', viz=True):
    """
    Aggregates parameter statistics for multiple algorithms into a single DataFrame.
    
    Parameters:
    -----------
    method_df_stats : dict
        Dictionary of DataFrames, one per algorithm/method
    data_type : str
        'binomial' or 'continuous' - passed to report_success_rates
    viz : bool
        Whether to display the resulting DataFrame
    
    Returns:
    --------
    pd.DataFrame
        MultiIndex DataFrame with (algorithm, group) index
    """
    all_reports = []

    for algo_name, df_stats in method_df_stats.items():
        # Get stats for this algorithm
        df_report = report_success_rates(df_stats, data_type=data_type)

        df_report["algorithm"] = algo_name
        all_reports.append(df_report)

    if not all_reports:
        return pd.DataFrame()

    df_combined = pd.concat(all_reports).reset_index().set_index(["algorithm", "group"])

    if viz:
        display(df_combined)

    return df_combined


def create_decision_correctness_df(method_stats, true_value, rope_min, rope_max, data_type='binomial'):
    """
    Evaluate decision correctness for all experiments.
    
    TODO: This solves for absolute decision correctness of accept/reject
          but not for the direction of rejection (e.g, higher or lower).
          This most likely should not impact PitG or ePiTG but it might impact HDI+ROPE
          which is likely to decide on the wrong side of the ROPE.
          This might be worth visualizing to examine prevalence.
    
    Parameters:
    -----------
    method_stats : dict
        Dictionary of method statistics from stop_decision_*_multiple_methods
    true_value : float
        True parameter value (success_rate for binomial, mean for continuous)
    rope_min : float
        Lower ROPE boundary
    rope_max : float
        Upper ROPE boundary
    data_type : str
        'binomial' or 'continuous'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with decision correctness for each experiment and method
    """
    accept_is_correct = rope_min <= true_value <= rope_max

    experiment_outcomes = {}
    method_names = ["hdi_rope", "pitg", "epitg"]

    for isample in range(len(method_stats[method_names[0]])):
        experiment_outcomes[isample] = {}
        for method_name in method_names:
            experiment_outcomes[isample][f"{method_name}_decision_iteration"] = \
                method_stats[method_name][isample]["decision_iteration"]
            experiment_outcomes[isample][f"{method_name}_accept"] = \
                method_stats[method_name][isample]["accept"]
            experiment_outcomes[isample][f"{method_name}_reject_below"] = \
                method_stats[method_name][isample]["reject_below"]
            experiment_outcomes[isample][f"{method_name}_reject_above"] = \
                method_stats[method_name][isample]["reject_above"]
            experiment_outcomes[isample][f"{method_name}_inconclusive"] = \
                method_stats[method_name][isample]["inconclusive"]

            # Extract parameter value (type-specific)
            if data_type == 'binomial':
                param_value = method_stats[method_name][isample]["successes"] / \
                             method_stats[method_name][isample]["decision_iteration"]
            else:  # continuous
                param_value = method_stats[method_name][isample]["sample_mean"]
            
            experiment_outcomes[isample][f"{method_name}_param_value"] = param_value

            if method_stats[method_name][isample]["inconclusive"]:
                # inconclusive - use expected value for decision making
                this_decision_accept = rope_min <= param_value <= rope_max
            else:  # conclusive case
                this_decision_accept = bool(method_stats[method_name][isample]["accept"]) \
                    if method_stats[method_name][isample]["accept"] is not None else None

            experiment_outcomes[isample][f"{method_name}_decision_correct"] = \
                this_decision_accept == accept_is_correct

    df_experiment_outcomes = pd.DataFrame(experiment_outcomes).T
    df_experiment_outcomes.index.name = "experiment_idx"

    return df_experiment_outcomes


def sims_hypo_dict_to_algo_stats_dfs(sims_hypo_dict):
    algo_stats_df = {}

    l_stats_viz = ["count","stop_iter_p25", 'stop_iter_median', "stop_iter_p75", "stop_iter_mean",
    "param_mean", "param_p25", "param_median", "param_p75",
    "conclusive_mean", "accept_mean", "reject_mean"]

    for subset_name in ["overall", "conclusive", "inconclusive"]:
        algo_stats_df[subset_name] = {}
        for algo_name in [ 'hdi_rope','pitg', 'epitg']:
            result_summary = {}
            for theta_true, experiment_result in sims_hypo_dict.items():
                if (algo_name, subset_name) in experiment_result['df_stats'].index:
                    result_summary[theta_true] = experiment_result['df_stats'].loc[(algo_name, subset_name), l_stats_viz]
                else:
                    result_summary[theta_true] = pd.Series({metric_:None for metric_ in l_stats_viz})
                    result_summary[theta_true]["count"] = 0


            algo_stats_df[subset_name][algo_name] = pd.DataFrame(result_summary)
            algo_stats_df[subset_name][algo_name].columns.name = 'theta_true'
            algo_stats_df[subset_name][algo_name].index.name = 'stat'
            algo_stats_df[subset_name][algo_name] = algo_stats_df[subset_name][algo_name].T

            algo_stats_df[subset_name][algo_name]  = algo_stats_df[subset_name][algo_name].sort_index()

    return algo_stats_df


def correctness_stats(df_experiment_correctness):
    q_pitg_inc = "pitg_inconclusive == True"
    cols_inconclusive = ['hdi_rope_inconclusive', 'pitg_inconclusive', 'epitg_inconclusive']
    cols_correctness = ['hdi_rope_decision_correct', 'pitg_decision_correct', 'epitg_decision_correct']

    return {
        "inconclusive_rates": df_experiment_correctness[cols_inconclusive].astype(int).mean(),
        "correctness_rates": df_experiment_correctness[cols_correctness].astype(int).mean(),
        "correctness_rates_pitg_inconclusive": df_experiment_correctness.query(q_pitg_inc)[cols_correctness].astype(int).mean()
    }

def sims_hypo_to_correctness_stats(sims_hypo_results):
    inconclusive_rates = {}
    correctness_rates = {}
    correctness_rates_pitg_inconclusive = {}

    for theta_true in sims_hypo_results.keys():
        correctness_ = correctness_stats(sims_hypo_results[theta_true]['hypothesis'].df_experiment_correctness)

        inconclusive_rates[theta_true] = correctness_['inconclusive_rates']
        correctness_rates[theta_true] = correctness_['correctness_rates']
        correctness_rates_pitg_inconclusive[theta_true] = correctness_['correctness_rates_pitg_inconclusive']


    df_inconclusive_rates = pd.DataFrame(inconclusive_rates).T.sort_index()
    df_correctness_rates = pd.DataFrame(correctness_rates).T.sort_index()
    df_correctness_rates_pitg_inconclusive = pd.DataFrame(correctness_rates_pitg_inconclusive).T.sort_index()

    return {"df_inconclusive_rates": df_inconclusive_rates,
            "df_correctness_rates": df_correctness_rates,
            "df_correctness_rates_pitg_inconclusive": df_correctness_rates_pitg_inconclusive}