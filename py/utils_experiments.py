import numpy as np
import pandas as pd

from utils_stats import (
    successes_failures_to_hdi_ci_limits
)

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


def stats_dict_to_df(method_stats):
    df = pd.DataFrame(method_stats).T
    df.index.name = "experiment_number"
    df["precision"] = df["hdi_max"] - df["hdi_min"]
    df["success_rate"] = df["successes"] / (df["successes"] + df["failures"])
    return df


def iteration_counts_to_df(roperesult_iteration, experiments):
    df = pd.DataFrame({
        "iteration": list(roperesult_iteration["within"].keys()),
        "accept": list(roperesult_iteration["within"].values()),
        "reject_below": list(roperesult_iteration["below"].values()),
        "reject_above": list(roperesult_iteration["above"].values())
    })

    df['reject'] = df['reject_above'] + df['reject_below']
    df['inconclusive'] = experiments - df['accept'] - df['reject']

    return df


def stop_decision_multiple_experiments_multiple_methods(samples, rope_min, rope_max, precision_goal, binary_accounting=None):
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