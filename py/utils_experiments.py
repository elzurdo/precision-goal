import numpy as np
import pandas as pd
from IPython.display import display

from utils_stats import (
    successes_failures_to_hdi_ci_limits
)

from utils_viz import (
    plot_multiple_decision_rates_separate,
    scatter_stop_iter_sample_rate,
    viz_one_sample_results,
    plot_sample_pdf_methods,
)

theta_true_str = r"$\theta_{\rm true}$"

# TODO: this solves for aboslute decision correctness of accept/reject
# but not for the direction of rejection (e.g, higher or lower)
# This most likely should not impace PitG or ePiTG but it might impact HDI+ROPE
# which is likely to decide on the wrong side of the ROPE.
# This might be worth vislalising to exmaine prevelance.
def create_decision_correctness_df(method_stats, true_rate, rope_min, rope_max):
    accept_is_correct = rope_min <= true_rate <= rope_max

    experiment_outcomes = {}

    method_names = ["hdi_rope", "pitg","epitg"]

    for isample in range(len(method_stats[method_names[0]])):
        experiment_outcomes[isample] = {}
        for method_name in method_names:
            None
            experiment_outcomes[isample][f"{method_name}_decision_iteration"] = method_stats[method_name][isample]["decision_iteration"]
            experiment_outcomes[isample][f"{method_name}_accept"] = method_stats[method_name][isample]["accept"]
            experiment_outcomes[isample][f"{method_name}_reject_below"] = method_stats[method_name][isample]["reject_below"]
            experiment_outcomes[isample][f"{method_name}_reject_above"] = method_stats[method_name][isample]["reject_above"]
            experiment_outcomes[isample][f"{method_name}_inconclusive"] = method_stats[method_name][isample]["inconclusive"]

            experiment_outcomes[isample][f"{method_name}_success_rate"] = method_stats[method_name][isample]["successes"] / method_stats[method_name][isample]["decision_iteration"]

            if method_stats[method_name][isample]["inconclusive"]:
                # inconclusive - use expected rate for decision making
                this_decision_accept = rope_min <= experiment_outcomes[isample][f"{method_name}_success_rate"]  <= rope_max
            else: # conclusive case
                this_decision_accept = bool(method_stats[method_name][isample]["accept"]) if method_stats[method_name][isample]["accept"] is not None else None

            experiment_outcomes[isample][f"{method_name}_decision_correct"] = this_decision_accept == accept_is_correct

            
    df_experiment_outcomes = pd.DataFrame(experiment_outcomes).T
    df_experiment_outcomes.index.name = "experiment_idx"

    return df_experiment_outcomes

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
        self.df_experiment_correctness = create_decision_correctness_df(self.method_stats, true_rate, self.rope_min, self.rope_max)


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
    df["hdi_max"] = df["hdi_max"].astype(float)
    df["hdi_min"] = df["hdi_min"].astype(float)
    df["decision_iteration"] = df["decision_iteration"].astype(float)
    df["reject"] = df["reject_below"] + df["reject_above"]
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
        sr_accept = df_group['accept'].astype(int) # convert boolean to int for stats
        sr_reject = df_group['reject'].astype(int) # convert boolean to int for stats

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
            # accept/reject statistics
            "accept_mean": sr_accept.mean(),
            "reject_mean": sr_reject.mean()
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
