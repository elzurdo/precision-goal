from typing import Dict, List, Union

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import binomtest  # used to be binom_test but that is deprecated in scipy 1.7.0 and will be removed in scipy 1.12.0

from utils_stats import (
    successes_failures_to_hdi_ci_limits
)

from utils_experiments_shared import (
    stats_dict_to_df,
    iteration_counts_to_df,
    report_success_rates,
    report_success_rates_multiple_algos,
    create_decision_correctness_df,
)

from utils_viz import (
    plot_multiple_decision_rates_separate,
    scatter_stop_iter_sample_rate,
    viz_one_sample_results,
    plot_sample_pdf_methods,
)

theta_true_str = r"$\theta_{\rm true}$"
SEQUENCE_HANDPICKED = "101101000110010000101111111110010101101110001111110010100110111111110111001111001110011110001010001011110101111110001111111111100000101001001100000001101000100010000000010010111001110100111000010010110011010000101011110011111111011100101011011100100101010011110101001111011100101110010011001010010001001011010101010100111100110011011011101110010100010110011001100101111001111101110101010001101110111100010110101010101010111100001000111011001010101100100110010001101101111100111000010011001000001010110010101101000001100101000110101110010101101000100110100100100110110100101011100001101000111111001001111100100011100011000101001010101110010000110111101111011100111011010010001001001111011100100000100011100000010010111111011110101000110110010001100101011110000001001101111100000001010011001001110001010100000101111100101110011011010111001000011110010011111110011111111100111011010000101110110001100111001000010011101100111000110010100000001101110000110011100111011100101001101010011001010100011000000011001100101100101000001101100111000000101010000110100100111110101101110010000100011101011011001110011100111011101010100101100001101100010111010010101000011000100111111010010111001100001001000110111011001011100100001001011111010011111101111001010000110011010101111001011110100001000100000010000011001110100110100100101000001100110111011011111010100111101111101010001010110010001000110111000101000010001011000100001101111011000000111010011000101001011110111101111010011101010111001111010101111011000110"

class BinomialHypothesis():
    def __init__(self, success_rate_null=0.5, dsuccess_rate=0.05, rope_precision_fraction=0.8):
        self.success_rate_null =  success_rate_null  # null hypothesis
        self.dsuccess_rate = dsuccess_rate  # - ROPE half width
        self.rope_precision_fraction = rope_precision_fraction # - precision must past this fractionu of ROPE

        self.set_hypothesis_params()

    def set_hypothesis_params(self):
        self.rope_min = self.success_rate_null - self.dsuccess_rate
        self.rope_max = self.success_rate_null + self.dsuccess_rate

        # hypothesis: if precision_goal is lower, then PitG has less of
        # an inconclusiveness problem but at the expense of more trials.
        self.precision_goal = (2 * self.dsuccess_rate) * self.rope_precision_fraction
        #precision_goal = (dsuccess_rate) * rope_precision_fraction # 1500 was not enough for 0.04
        #precision_goal = (1.5 * dsuccess_rate) * rope_precision_fraction # 1500 was not enough for 0.04


        print(f"{self.success_rate_null:0.5}: null hypothesis")
        print(f"{self.rope_min:0.2}: ROPE min")
        print(f"{self.rope_max:0.2}: ROPE max")
        print("-" * 20)
        print(f"{self.precision_goal:0.2}: Precision Goal")

    def run_hypothesis_on_experiments(self, experiments, binary_accounting):
        self.experiments = experiments
        self.n_experiments = experiments.shape[0]
        self.method_stats, self.method_roperesult_iteration  = stop_decision_multiple_experiments_multiple_methods(experiments, self.rope_min, self.rope_max, self.precision_goal, binary_accounting=binary_accounting)

        self.method_df_stats = {method_name: stats_dict_to_df(self.method_stats[method_name]) for method_name in self.method_stats}
        self.method_df_iteration_counts = {method_name: iteration_counts_to_df(self.method_roperesult_iteration[method_name], self.n_experiments) for method_name in self.method_roperesult_iteration}

        # creating table to summarize results
        self.experiments_summary()

    def one_experiment_all_iterations(self, iexperiment, binary_accounting=None, viz=True, success_rate=None, xlim = (0.4,0.8), method_names=None):
        print(self.experiments[iexperiment, :])
        df_experiment_results = sample_all_iterations_results(self.experiments[iexperiment, :], self.precision_goal, self.rope_min, self.rope_max, binary_accounting=binary_accounting, iteration_number=None)

        if viz:
            self.viz_one_experiment_all_iterations(df_experiment_results, success_rate=success_rate)
            self.plot_experiment_pdf_methods(iexperiment, xlim=xlim, method_names=method_names)

        return df_experiment_results

    def plot_decision_rates(self, success_rate=None, viz_epitg="separate"):
        plot_multiple_decision_rates_separate(self.method_df_iteration_counts, success_rate, self.n_experiments, viz_epitg=viz_epitg, iteration_values=None)

    def plot_stop_iter_sample_rates(self, success_rate=None, title=None):
        scatter_stop_iter_sample_rate(self.method_df_stats, rope_min=self.rope_min, rope_max=self.rope_max, success_rate_true=success_rate, success_rate_hypothesis=self.success_rate_null, precision_goal=self.precision_goal, title=title)

    def viz_one_experiment_all_iterations(self, df_sample_results, success_rate=None):
        viz_one_sample_results(df_sample_results, self.precision_goal, self.rope_min, self.rope_max, success_rate=success_rate)

    def plot_experiment_pdf_methods(self, iexperiment, xlim=(0.4,0.8), method_names=None):
        plot_sample_pdf_methods(self.method_df_stats, iexperiment, self.rope_min, self.rope_max, xlim=xlim, method_names=method_names)

    # TODO: update conditional stats. E.g, only conclusive, only inconclusive, etc.
    def experiments_summary(self, verbose=1):
        method_names = ["hdi_rope", "pitg", "epitg"]

        stat_queries = { "accept": "accept",
                        "reject": "reject",
                        "conclusive": "conclusive",
                        "inconclusive": "inconclusive",
                        "stop_iter_mean": None,
                        "stop_iter_std": None,
                        "success_rate_mean": None,
                        "success_rate_std": None,
                        }

        stat_results = {}

        for method_name in method_names:
            stat_results[method_name] = {}

            sr_stop_iter = self.method_df_stats[method_name]["decision_iteration"].copy()
            sr_success_rate = self.method_df_stats[method_name]["success_rate"].copy()
            for stat_name, stat_query in stat_queries.items():
                if ("_mean" not in stat_name) & ("_std" not in stat_name):
                    value_ = self.method_df_stats[method_name].query(stat_query).shape[0]
                    stat_results[method_name][stat_name] = value_ / self.n_experiments
                else:
                    if "stop_iter" in stat_name:
                        sr_aux = sr_stop_iter.copy()
                    elif "success_rate" in stat_name:
                        sr_aux = sr_success_rate.copy()
                    else:
                        sr_aux = None
                    
                    if "_mean" in stat_name:
                        stat_results[method_name][stat_name]  = sr_aux.mean()
                    elif "_std" in stat_name:
                        stat_results[method_name][stat_name]  = sr_aux.std()

        self.df_experiments_summary = pd.DataFrame(stat_results).T

        if verbose:
            display(self.df_experiments_summary)

    def decision_correctness(self, true_rate):
        self.df_experiment_correctness = create_decision_correctness_df(
            self.method_stats, true_rate, self.rope_min, self.rope_max, data_type='binomial'
        )


class BinomialSimulation():
    def __init__(self, success_rate=0.5, n_samples = 1500,  n_experiments = 500, seed=42):
        self.success_rate = success_rate  #0.65  ## the true value # 0.5 + 0.5 * dsuccess_rate
        self.n_samples = n_samples
        self.n_experiments = n_experiments
        self.seed= seed

        self.generate_experiments()


    def generate_experiments(self):

        print("Generating synthetic data with parameter values:")
        print(f"{self.success_rate:0.3}: true success rate")
        print(f"{self.n_experiments}: experiments")
        print(f"{self.n_samples}: sample size per experiment")
        
        np.random.seed(self.seed)
        # `experiments` was called `samples` in the original code
        self.experiments = np.random.binomial(1, self.success_rate, [self.n_experiments, self.n_samples])


class BinaryAccounting():
    def __init__(self):
        self.dict_successes_failures_counter = {}
        self.dict_successes_failures_hdi_limits = {}
    
    def successes_failures_to_hdi_limits(self, successes, failures):
        pair = (successes, failures)
        if pair not in self.dict_successes_failures_hdi_limits:
            self.dict_successes_failures_hdi_limits[pair] =\
                successes_failures_caculate_hdi_limits(successes, failures)
            self.dict_successes_failures_counter[pair] = 1
        else:
            self.dict_successes_failures_counter[pair] += 1

        return self.dict_successes_failures_hdi_limits[pair]


class BinaryPvalueAccounting():
    """Memoised binomial p-value calculator.

    Caches p-values keyed by (successes, n) so that repeated calls with the
    same counts — common when running many experiments of the same length —
    avoid redundant `binomtest` evaluations.

    Parameters
    ----------
    success_rate_null : float
        Null-hypothesis success rate (fixed for the lifetime of the object).
    alternative : str
        'two-sided', 'greater', or 'less' (fixed for the lifetime of the object).
    """

    def __init__(self, success_rate_null: float = 0.5, alternative: str = 'two-sided'):
        self.success_rate_null = success_rate_null
        self.alternative = alternative
        self.dict_successes_n_pvalue = {}   # type: Dict[tuple, float]
        self.dict_successes_n_counter = {}  # type: Dict[tuple, int]

    def successes_n_to_pvalue(self, successes: int, n: int) -> float:
        pair = (successes, n)
        if pair not in self.dict_successes_n_pvalue:
            self.dict_successes_n_pvalue[pair] = binomtest(
                successes, n=n, p=self.success_rate_null, alternative=self.alternative
            ).pvalue
            self.dict_successes_n_counter[pair] = 1
        else:
            self.dict_successes_n_counter[pair] += 1

        return self.dict_successes_n_pvalue[pair]


def _update_iteration_tally(iteration_dict, iteration):
    for this_iteration in range(iteration, len(iteration_dict)+1):
        iteration_dict[this_iteration] += 1

def booleans_to_rope_result(decision_accept, decision_reject_below, decision_reject_above):
    if decision_accept:
        return "within"
    elif decision_reject_below:
        return "below"
    elif decision_reject_above:
        return "above"

def successes_failures_caculate_hdi_limits(successes, failures):
    aa = int(successes)
    bb = int(failures)
    
    if not failures:
        aa += 1
        bb += 1
        
    if not successes:
        aa += 1
        bb += 1

    hdi_min, hdi_max = successes_failures_to_hdi_ci_limits(aa, bb)

    return hdi_min, hdi_max


def stop_decision_multiple_experiments_multiple_methods(samples, rope_min, rope_max, precision_goal, binary_accounting=None, min_iter=30, viz=False):
    # For each method and rope result type creating tally of outcomes
    method_names = ["pitg", "epitg", "hdi_rope"]
    n_samples = samples.shape[1]

    # stats at sample stop iteration
    method_stats = {method_name: {} for method_name in method_names}

    # for each iteration the number of sample that stopped there or before
    # and the decision outcome
    method_roperesult_iteration = {}
    
    rope_results = ["within", "below", "above"]
    for method in method_names:
        method_roperesult_iteration[method] = {}
        for rope_result in rope_results:
            method_roperesult_iteration[method][rope_result] = {iteration: 0 for iteration in range(1, n_samples + 1)}


    iteration_number = np.arange(1, n_samples + 1)

    for isample, sample in enumerate(samples):
        pitg_stopped = False
        hdi_rope_stopped = False

        # the number of successes at each iteration for this sample
        iteration_successes = sample.cumsum()
        iteration_failures = iteration_number - iteration_successes

        # examining the specifics of each iteration
        for iteration, successes, failures in zip(iteration_number, iteration_successes, iteration_failures):
            final_iteration = iteration == iteration_number[-1]

            if binary_accounting is not None:
                hdi_min, hdi_max = binary_accounting.successes_failures_to_hdi_limits(successes, failures)
            else:
                hdi_min, hdi_max = successes_failures_to_hdi_ci_limits(successes, failures)

            # has the precision goal been achieved?
            precision_goal_achieved = (hdi_max - hdi_min) < precision_goal

            # is the HDI conclusively within or outside the ROPE?
            decision_accept = (hdi_min >= rope_min) & (hdi_max <= rope_max)
            decision_reject_below = hdi_max < rope_min  
            decision_reject_above = rope_max < hdi_min
            conclusive = decision_accept | decision_reject_above | decision_reject_below

            if min_iter is not None:
                if iteration < min_iter:
                    #continue
                    decision_accept = False
                    decision_reject_below = False
                    decision_reject_above = False
                    conclusive = False
                    precision_goal_achieved = False


            iteration_results = {"decision_iteration": iteration,
                                                    "accept": decision_accept,
                                                        "reject_below": decision_reject_below,
                                                        "reject_above": decision_reject_above,
                                                        "conclusive": conclusive,
                                                        "inconclusive": not conclusive,
                                                        "successes": successes,
                                                        "failures": failures,
                                                        "hdi_min": hdi_min,
                                                        "hdi_max": hdi_max,
                                                        "precision_goal_achieved": precision_goal_achieved,
                                                    }   

            if precision_goal_achieved:

                # update Precision Is The Goal Stop
                if pitg_stopped is False:
                    # not applying `break` because we continue for ePiTG
                    if conclusive:
                        rope_result = booleans_to_rope_result(decision_accept, decision_reject_below, decision_reject_above)
                        _update_iteration_tally(method_roperesult_iteration["pitg"][rope_result], iteration)
                    method_stats["pitg"][isample] = iteration_results
                    pitg_stopped = True  # sample does not continue with PITG (only ePiTG) 

                # continue with Enhance Precision Is The Goal
                if conclusive:
                    rope_result = booleans_to_rope_result(decision_accept, decision_reject_below, decision_reject_above)
                    _update_iteration_tally(method_roperesult_iteration["epitg"][rope_result], iteration)

                    if hdi_rope_stopped is False:
                        _update_iteration_tally(method_roperesult_iteration["hdi_rope"][rope_result], iteration)
                        
                if conclusive | final_iteration:
                    method_stats["epitg"][isample] = iteration_results

                    if hdi_rope_stopped is False:
                        method_stats["hdi_rope"][isample] = iteration_results
                        hdi_rope_stopped = True

                    if final_iteration:
                        if viz:
                            print(f"Sample {isample} at final iteration")
                            print(method_stats["epitg"][isample])
                    break
            
            elif conclusive & (hdi_rope_stopped is False):
                # case in which precision not achieved yet but conclusive.
                # this is the HDI+ROPE approach which disregards precision
                rope_result = booleans_to_rope_result(decision_accept, decision_reject_below, decision_reject_above)
                _update_iteration_tally(method_roperesult_iteration["hdi_rope"][rope_result], iteration)
                method_stats["hdi_rope"][isample] = iteration_results

                hdi_rope_stopped = True

            elif final_iteration:
                # ensures that if reached final iteration and not conclusive that
                # results are still recorded as inconclusive
                if isample not in method_stats["hdi_rope"]:
                    method_stats["hdi_rope"][isample] = iteration_results
                if isample not in method_stats["pitg"]:
                    method_stats["pitg"][isample] = iteration_results
                if isample not in method_stats["epitg"]:
                    method_stats["epitg"][isample] = iteration_results
                break

    return method_stats, method_roperesult_iteration

def print_decision_rates(df_stats):
    for decision_ in ["accept", "reject", "inconclusive"]:
        print(f"{100. * df_stats.query(decision_).shape[0] / len(df_stats):06.3f}% {decision_}")

    print(f"\n{df_stats['success_rate'].mean():06.3%} mean success rate")
    print(f"{df_stats['success_rate'].median():06.3%} median success rate")
    print(f"{df_stats['success_rate'].std():06.3%} std success rate")

def print_methods_decision_rates(method_df_stats):
    for method_name in method_df_stats:
        print(f"{method_name}")
        print_decision_rates(method_df_stats[method_name])
        print("-" * 20)

def sample_all_iterations_results(sample, precision_goal, rope_min, rope_max, iteration_number=None, binary_accounting=None):
    # By all iterations it means that it doesn't stop, but does flag
    # when objectives are met: conclusiveness, percision goal
    if iteration_number is None:
        iteration_number = np.arange(1, sample.shape[0] + 1)

    iteration_successes = sample.cumsum()
    iteration_failures = iteration_number - iteration_successes

    sample_results = {}
    for iteration, successes, failures in zip(iteration_number, iteration_successes, iteration_failures):
        # final_iteration = iteration == iteration_number[-1]
        
        # TODO: turn this part into a function (if works out well with other part)
        if binary_accounting is not None:
            hdi_min, hdi_max = binary_accounting.successes_failures_to_hdi_limits(successes, failures)
        else:
            hdi_min, hdi_max = successes_failures_to_hdi_ci_limits(successes, failures)
        # has the precision goal been achieved?
        precision_goal_achieved = (hdi_max - hdi_min) < precision_goal

        # is the HDI conclusively within or outside the ROPE?
        decision_accept = (hdi_min >= rope_min) & (hdi_max <= rope_max)
        decision_reject_below = hdi_max < rope_min  
        decision_reject_above = rope_max < hdi_min
        conclusive = decision_accept | decision_reject_above | decision_reject_below


        iteration_results = {"decision_iteration": iteration,
                                                    "accept": decision_accept,
                                                    "reject_below": decision_reject_below,
                                                    "reject_above": decision_reject_above,
                                                    "conclusive": conclusive,
                                                    "inconclusive": not conclusive,
                                                    "successes": successes,
                                                    "failures": failures,
                                                    "hdi_min": hdi_min,
                                                    "hdi_max": hdi_max,
                                                    "goal_achieved": precision_goal_achieved,
                                                    }  
        # END TODO 

        sample_results[iteration] = iteration_results

    df_sample_results = stats_dict_to_df(sample_results)    

    return df_sample_results


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
    hypothesis.decision_correctness(success_rate_true)
    if viz:
        hypothesis.plot_decision_rates(synth.success_rate)
        hypothesis.plot_stop_iter_sample_rates(success_rate=synth.success_rate, title=None)
    df_stats = report_success_rates_multiple_algos(hypothesis.method_df_stats.copy(), data_type='binomial', viz=viz)

    return {"synth": synth, "hypothesis": hypothesis, "df_stats": df_stats}


def sequence_to_sequential_pvalues(sequence, success_rate_null=0.5, alternative='two-sided'):
    assert alternative in ['two-sided', 'greater', 'less'], "alternative must be one of 'two-sided', 'greater', or 'less'"

    p_values = []

    for idx, successes in enumerate(sequence.cumsum()):
        p_value = binomtest(successes, n=idx + 1, p=success_rate_null, alternative=alternative).pvalue
        p_values.append(p_value)

    p_values = np.array(p_values)

    return p_values


def stop_decision_multiple_experiments_nhst__slow(samples, p_value_thresh=0.05, success_rate_null=0.5):
    # TEST THIS! GEnerate by autocomplete based on function name!
    n_samples = samples.shape[1]
    n_experiments = samples.shape[0]

    iteration_stopping_on_or_prior = np.zeros(n_experiments)

    for isample, sample in enumerate(samples):
        p_values = sequence_to_sequential_pvalues(sample, success_rate_null=success_rate_null)

        stopping_iterations = np.where(p_values < p_value_thresh)[0]
        if len(stopping_iterations) > 0:
            iteration_stopping_on_or_prior[isample] = stopping_iterations[0] + 1  # +1 because iterations are 1-indexed

    return iteration_stopping_on_or_prior


# TODO: raname: 'samples' --> 'sequences' or 'experiments'
# TODO: unit test
def stop_decision_multiple_experiments_nhst(
    experiments: np.ndarray,
    p_value_thresh: float = 0.05,
    success_rate_null: float = 0.5,
    alternative: str = 'two-sided',
    binary_pvalue_accounting: 'BinaryPvalueAccounting' = None,
) -> Dict[str, Union[Dict[int, int], Dict[str, List]]]:
    """Run sequential NHST (optional stopping) on multiple binary experiments.

    For each experiment (row), iterates through observations computing a binomial
    test p-value at each step. Stops early if p-value <= p_value_thresh.

    Parameters
    ----------
    experiments : np.ndarray
        2D array of shape (num_experiments, sequence_length) with binary (0/1) values.
        Each row is one experiment; each column is one observation.
    p_value_thresh : float
        Significance threshold for early stopping (default 0.05).
    success_rate_null : float
        Null hypothesis success rate (default 0.5).
    alternative : str
        Direction of the test: 'two-sided', 'greater', or 'less'.
    binary_pvalue_accounting : BinaryPvalueAccounting, optional
        If provided, p-values are looked up from its cache before computing.
        Must be constructed with matching success_rate_null and alternative.
        Speeds up runs with many experiments by avoiding redundant binomtest calls.

    Returns
    -------
    dict with keys:
        "iteration_stopping_on_or_prior" : dict[int, int]
            Maps iteration (1-indexed) to count of experiments that stopped
            at or before that iteration.
        "experiment_stop_results" : dict[str, list]
            Lists of 'successes', 'trials', and 'p_value' per experiment
            (at the stopping point, or at the final iteration if no early stop).
    """
    n_observations = experiments.shape[1]

    experiment_stop_results = {'successes': [], 'trials': [], 'p_value': []}  # type: Dict[str, List]
    iteration_stopping_on_or_prior = {iteration: 0 for iteration in range(1, n_observations + 1)}  # type: Dict[int, int]

    for sequence in experiments:
        successes = 0
        this_iteration = 0
        for toss in sequence:
            successes += toss
            this_iteration += 1

            if binary_pvalue_accounting is not None:
                p_value = binary_pvalue_accounting.successes_n_to_pvalue(successes, this_iteration)
            else:
                p_value = binomtest(successes, n=this_iteration, p=success_rate_null, alternative=alternative).pvalue

            if p_value <= p_value_thresh:
                for iteration in range(this_iteration, n_observations+1):
                    iteration_stopping_on_or_prior[iteration] += 1
                    
                break
        experiment_stop_results['successes'].append(successes)
        experiment_stop_results['trials'].append(this_iteration)
        experiment_stop_results['p_value'].append(p_value)


    return {
        "iteration_stopping_on_or_prior": iteration_stopping_on_or_prior,
        "experiment_stop_results": experiment_stop_results,
    }