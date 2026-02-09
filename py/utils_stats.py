from scipy.optimize import fmin
from scipy.stats import beta, binom_test # binom_test is binomtest (in later versions ...)
import numpy as np
import pandas as pd

CI_FRACTION = 0.95
MIN_COUNTS = 1.


def binomial_rate_ci_width_to_sample_size(p, credible_interval_width, z_star = 1.96):
    variance_ = (0.5 *  credible_interval_width / z_star) ** 2
    n_ = p * (1 - p) / variance_ - 1
    return n_

def test_value(x):
    if x == 0:
        return MIN_COUNTS
    else:
        return x

def HDIofICDF(dist_name, ci_fraction=CI_FRACTION, **args):
    """
    This program finds the HDI of a probability density function that is specified
    mathematically in Python.

    Example usage: HDIofICDF(beta, a=100, b=100)
    """
    # freeze distribution with given arguments
    distri = dist_name(**args)
    # initial guess for HDIlowTailPr
    incredMass =  1.0 - ci_fraction


    def intervalWidth(lowTailPr):
        return distri.ppf(ci_fraction + lowTailPr) - distri.ppf(lowTailPr)

    # find lowTailPr that minimizes intervalWidth
    HDIlowTailPr = fmin(intervalWidth, incredMass, ftol=1e-8, disp=False)[0]
    # return interval as array([low, high])
    return distri.ppf([HDIlowTailPr, ci_fraction + HDIlowTailPr])


def hdi_ci_limits(p, n, ci_fraction=CI_FRACTION):
    a = p * n
    b = n - a

    ci_min, ci_max = HDIofICDF(beta, a=a, b=b, ci_fraction=ci_fraction)
    return ci_min, ci_max

def successes_failures_to_hdi_ci_limits(a, b, ci_fraction=CI_FRACTION):
    return HDIofICDF(beta, a=a, b=b, ci_fraction=ci_fraction)


def get_success_rates(d_success = 0.00001, min_range=0., max_range=1., including_max=False):
    # d_success determines how accurate the audit_success_rate=99% results are. more zeros: more accurrate but slower

    max_range_ = max_range
    if including_max: # when max_range = 1. one must be cautious about beta going to infinity when failure = 0
        max_range_ += d_success

    success_rates = np.arange(min_range, max_range_, d_success)

    return d_success, success_rates

# TODO: move to utils_experiments and update
def sequence_to_sequential_pvalues(sequence, success_rate_null=0.5):
    p_values = []
    
    for idx, successes in enumerate(sequence.cumsum()):
        p_value = binom_test(successes, n=idx + 1, p=success_rate_null, alternative='two-sided') # alternative {‘two-sided’, ‘greater’, ‘less’},
        p_values.append(p_value)
    
    p_values = np.array(p_values)
    
    return p_values


# TODO: move to utils_experiments and update
def stop_decision_multiple_experiments(samples, nhst_details=None):
    n_samples = samples.shape[1]

    experiment_stop_results = {'successes': [], 'trials': [], 'p_value': []}
    iteration_stopping_on_or_prior = {iteration: 0 for iteration in range(1, n_samples + 1)}

    for sample in samples:
        successes = 0
        this_iteration = 0
        for toss in sample:
            successes += toss
            this_iteration += 1
            
            if nhst_details is not None:
                p_value = binom_test(successes, n=this_iteration, p=nhst_details['success_rate_null'], alternative=nhst_details['alternative'])
            
            if p_value < nhst_details['p_value_thresh']:
                for iteration in range(this_iteration, n_samples+1):
                    iteration_stopping_on_or_prior[iteration] += 1
                    
                break
        experiment_stop_results['successes'].append(successes)
        experiment_stop_results['trials'].append(this_iteration)
        if nhst_details is not None:
            experiment_stop_results['p_value'].append(p_value)

    return experiment_stop_results, iteration_stopping_on_or_prior


# TODO: move to utils_experiments and update
def stop_decision_multiple_experiments_bayesian(samples, bayes_details=None):

    n_experiments = samples.shape[0]
    n_samples = samples.shape[1]

    experiment_stop_results = {'successes': [], 'trials': [], 'within_rope': [], 'ci_min': [], 'ci_max': []}
    iteration_stopping_on_or_prior_within = {iteration: 0 for iteration in range(1, n_samples + 1)}
    iteration_stopping_on_or_prior_below = iteration_stopping_on_or_prior_within.copy()
    iteration_stopping_on_or_prior_above = iteration_stopping_on_or_prior_within.copy()


    for sample in samples:
        successes = 0
        this_iteration = 0
        for toss in sample:
            successes += toss
            this_iteration += 1
            
            failures = this_iteration - successes
        
            if this_iteration > 5: # cannot rely on below 5
                ci_min, ci_max = successes_failures_to_hdi_ci_limits(successes, failures)
                this_within_rope = (ci_min >= bayes_details['rope_min']) & (ci_max <= bayes_details['rope_max'])

                if this_within_rope:
                    for iteration in range(this_iteration, n_samples+1):
                        iteration_stopping_on_or_prior_within[iteration] += 1
                    break

                if (ci_max < bayes_details['rope_min']): 
                    for iteration in range(this_iteration, n_samples+1):
                        iteration_stopping_on_or_prior_below[iteration] += 1
                    break

                if (ci_min > bayes_details['rope_max']):
                    for iteration in range(this_iteration, n_samples+1):
                        iteration_stopping_on_or_prior_above[iteration] += 1
                    break
                
                
        experiment_stop_results['successes'].append(successes)
        experiment_stop_results['trials'].append(this_iteration)
        experiment_stop_results['within_rope'].append(this_within_rope)
        experiment_stop_results['ci_min'].append(ci_min)
        experiment_stop_results['ci_max'].append(ci_max)

    
    df_decision_counts = pd.DataFrame({'within': iteration_stopping_on_or_prior_within,
               'below': iteration_stopping_on_or_prior_below, 
              'above': iteration_stopping_on_or_prior_above,
              })

    df_decision_counts.index.name = "iteration_number"
    df_decision_counts['reject'] = df_decision_counts['above'] + df_decision_counts['below']
    df_decision_counts['inconclusive'] = n_experiments - df_decision_counts['within'] - df_decision_counts['reject']
    
    return experiment_stop_results, df_decision_counts

# TODO: move to utils_experiments and update
def stop_decision_multiple_experiments_pitg(samples, precision_goal, bayes_details=None):
    n_experiments = samples.shape[0]
    n_samples = samples.shape[1]

    experiment_stop_results = {'successes': [], 'trials': [], 'ci_min': [], 'ci_max': []}
    iteration_stopping_on_or_prior_accept = {iteration: 0 for iteration in range(1, n_samples + 1)}
    iteration_stopping_on_or_prior_reject_below = iteration_stopping_on_or_prior_accept.copy()
    iteration_stopping_on_or_prior_reject_above = iteration_stopping_on_or_prior_accept.copy()

    for sample in samples:
        successes = 0
        this_iteration = 0
        for toss in sample:
            successes += toss
            this_iteration += 1
            
            failures = this_iteration - successes
            
            aa = int(successes)
            bb = int(failures)
            
            if not failures:
                aa += 1
                bb += 1
                
            if not successes:
                aa += 1
                bb += 1
                
            ci_min, ci_max = successes_failures_to_hdi_ci_limits(aa, bb)
            
            this_precision_goal_achieved = (ci_max - ci_min) < precision_goal
        
            if this_precision_goal_achieved:
                break
                
        this_accept_within = (ci_min >= bayes_details['rope_min']) & (ci_max <= bayes_details['rope_max'])

        if this_accept_within & this_precision_goal_achieved:
            for iteration in range(this_iteration, n_samples+1):
                iteration_stopping_on_or_prior_accept[iteration] += 1

        if (ci_max < bayes_details['rope_min']) & this_precision_goal_achieved: 
            for iteration in range(this_iteration, n_samples+1):
                iteration_stopping_on_or_prior_reject_below[iteration] += 1

        if (ci_min > bayes_details['rope_max']) & this_precision_goal_achieved:
            for iteration in range(this_iteration, n_samples+1):
                iteration_stopping_on_or_prior_reject_above[iteration] += 1

        experiment_stop_results['successes'].append(successes)
        experiment_stop_results['trials'].append(this_iteration)
        experiment_stop_results['ci_min'].append(ci_min)
        experiment_stop_results['ci_max'].append(ci_max)

    
    df_decision_counts = pd.DataFrame({'accept': iteration_stopping_on_or_prior_accept,
                'reject_below': iteration_stopping_on_or_prior_reject_below, 
                'reject_above': iteration_stopping_on_or_prior_reject_above,
                })

    df_decision_counts.index.name = "iteration_number"
    df_decision_counts['reject'] = df_decision_counts['reject_above'] + df_decision_counts['reject_below']
    df_decision_counts['inconclusive'] = n_experiments - df_decision_counts['accept'] - df_decision_counts['reject']

    return experiment_stop_results, df_decision_counts

from utils_experiments import (
    BinaryAccounting,
    BinomialSimulation,
    BinomialHypothesis,
)

def report_success_rates(df_stats):
    """
    Computes summary statistics for success_rate across different decision subgroups.
    """
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
            
        sr_success_rate = df_group['success_rate']
        sr_stop_iter = df_group['decision_iteration']
        sr_conclusive = df_group['conclusive'].astype(int) # convert boolean to int for stats
        
        records.append({
            "group": group_name,
            "count": int(sr_success_rate.count()),
            "success_frac": sr_success_rate.count() / len(df_stats),
            # success rate statistics
            "success_mean": sr_success_rate.mean(),
            "success_std": sr_success_rate.std(),
            "success_p25": sr_success_rate.quantile(0.25),
            "success_median": sr_success_rate.median(),
            "success_p75": sr_success_rate.quantile(0.75),
            # stop iteration statistics
            "stop_iter_mean": sr_stop_iter.mean(),
            "stop_iter_std": sr_stop_iter.std(),
            "stop_iter_p25": sr_stop_iter.quantile(0.25),
            "stop_iter_median": sr_stop_iter.median(),
            "stop_iter_p75": sr_stop_iter.quantile(0.75),
            # conclusive statistics
            "conclusive_mean": sr_conclusive.mean(),
        })
        
    return pd.DataFrame(records).set_index("group")

def report_success_rates_multiple_algos(method_df_stats, viz=True):
    """
    Aggregates success rate statistics for multiple algorithms into a single DataFrame.
    """
    all_reports = []
    
    for algo_name, df_stats in method_df_stats.items():
        # Get stats for this algorithm
        df_report = report_success_rates(df_stats)
        
        # We generally care most about the 'overall' statistics for comparison, 
        # or we might want a multi-index (Algo, Group). 
        # Based on the user request "each row is a different algo_name", 
        # it implies comparing apples-to-apples (likely 'overall' or weighted stats).
        # However, information about 'conclusive' vs 'inconclusive' is vital.
        # Let's create a MultiIndex DataFrame to capture everything cleanly.
        
        df_report["algorithm"] = algo_name
        all_reports.append(df_report)
        
    if not all_reports:
        return pd.DataFrame()

    df_combined = pd.concat(all_reports).reset_index().set_index(["algorithm", "group"])
    
    from IPython.display import display
    
    if viz:
        display(df_combined)
    
    return df_combined

def run_simulations_and_analysis_report(binary_accounting: BinaryAccounting,
                                        success_rate_true: float=0.5,
                                        success_rate_null: float=0.5,
                                        dsuccess_rate: float=0.05,
                                        n_samples: int=1500,
                                        n_experiments: int=2000,
                                        seed: int=42,
                                        rope_precision_fraction: float=0.08,
                                        viz=True,
                                        ):
    synth = BinomialSimulation(success_rate=success_rate_true, n_experiments=n_experiments, n_samples=n_samples,seed=seed)
    hypothesis = BinomialHypothesis(success_rate_null=success_rate_null,dsuccess_rate=dsuccess_rate, rope_precision_fraction=rope_precision_fraction)
    hypothesis.run_hypothesis_on_experiments(synth.experiments, binary_accounting)
    if viz:
        hypothesis.plot_decision_rates(synth.success_rate)
        hypothesis.plot_stop_iter_sample_rates(success_rate=synth.success_rate, title=None)
    df_stats = report_success_rates_multiple_algos(hypothesis.method_df_stats.copy(), viz=viz)

    return {"synth": synth, "hypothesis": hypothesis, "df_stats": df_stats}

