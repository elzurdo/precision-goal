import numpy as np
import pandas as pd
from IPython.display import display

from utils_stats import continuous_hdi_ci_limits
from utils_experiments_shared import (
    stats_dict_to_df, 
    iteration_counts_to_df,
    report_success_rates_multiple_algos,
)
from utils_viz_continuous import viz_one_sample_results_continuous


class ContinuousHypothesis():
    """
    Hypothesis testing for continuous data using (e)PitG methods.
    Minimal bare-bones implementation focusing on core sequential testing.
    """
    def __init__(self, mean_null=0.0, dmean=0.5, rope_precision_fraction=0.8):
        """
        Parameters:
        -----------
        mean_null : float
            Null hypothesis mean
        dmean : float
            ROPE half-width
        rope_precision_fraction : float
            Precision goal as fraction of ROPE width
        """
        self.mean_null = mean_null  # null hypothesis
        self.dmean = dmean  # ROPE half width
        self.rope_precision_fraction = rope_precision_fraction
        
        self.set_hypothesis_params()
    
    def set_hypothesis_params(self):
        """Set ROPE boundaries and precision goal from initialization parameters."""
        # TODO: Examine for generalisability - very similar to BinomialHypothesis
        self.rope_min = self.mean_null - self.dmean
        self.rope_max = self.mean_null + self.dmean
        
        # Precision goal: fraction of ROPE width
        self.precision_goal = (2 * self.dmean) * self.rope_precision_fraction
        
        print(f"{self.mean_null:0.5}: null hypothesis mean")
        print(f"{self.rope_min:0.2}: ROPE min")
        print(f"{self.rope_max:0.2}: ROPE max")
        print("-" * 20)
        print(f"{self.precision_goal:0.2}: Precision Goal")
    
    def run_hypothesis_on_experiments(self, experiments):
        """
        Run hypothesis testing on all experiments using PitG, ePitG, and HDI+ROPE methods.
        
        Parameters:
        -----------
        experiments : np.ndarray
            Shape (n_experiments, n_samples) continuous data
        """
        self.experiments = experiments
        self.n_experiments = experiments.shape[0]
        
        # Core sequential testing function for continuous data
        self.method_stats, self.method_roperesult_iteration = \
            stop_decision_continuous_multiple_methods(
                experiments, self.rope_min, self.rope_max, self.precision_goal
            )
        
        # Convert to DataFrames using shared generic function with data_type='continuous'
        self.method_df_stats = {
            method_name: stats_dict_to_df(self.method_stats[method_name], data_type='continuous')
            for method_name in self.method_stats
        }
        self.method_df_iteration_counts = {
            method_name: iteration_counts_to_df(self.method_roperesult_iteration[method_name], self.n_experiments)
            for method_name in self.method_roperesult_iteration
        }
        
        # Summary statistics
        self.experiments_summary()
    
    # TODO: Examine for generalisability
    def experiments_summary(self, verbose=1):
        """Summarize experiment results across methods."""
        method_names = ["hdi_rope", "pitg", "epitg"]
        
        stat_queries = {
            "accept": "accept",
            "reject": "reject",
            "conclusive": "conclusive",
            "inconclusive": "inconclusive",
            "stop_iter_mean": None,
            "stop_iter_std": None,
            "mean_estimate_mean": None,  # continuous-specific
            "mean_estimate_std": None,   # continuous-specific
        }
        
        stat_results = {}
        
        for method_name in method_names:
            stat_results[method_name] = {}
            
            sr_stop_iter = self.method_df_stats[method_name]["decision_iteration"].copy()
            sr_mean_estimate = self.method_df_stats[method_name]["sample_mean"].copy()
            
            for stat_name, stat_query in stat_queries.items():
                if ("_mean" not in stat_name) & ("_std" not in stat_name):
                    value_ = self.method_df_stats[method_name].query(stat_query).shape[0]
                    stat_results[method_name][stat_name] = value_ / self.n_experiments
                else:
                    if "stop_iter" in stat_name:
                        sr_aux = sr_stop_iter.copy()
                    elif "mean_estimate" in stat_name:
                        sr_aux = sr_mean_estimate.copy()
                    else:
                        sr_aux = None
                    
                    if "_mean" in stat_name:
                        stat_results[method_name][stat_name] = sr_aux.mean()
                    elif "_std" in stat_name:
                        stat_results[method_name][stat_name] = sr_aux.std()
        
        self.df_experiments_summary = pd.DataFrame(stat_results).T
        
        if verbose:
            display(self.df_experiments_summary)
    
    def decision_correctness(self, true_mean):
        """
        Evaluate decision correctness across experiments.
        
        Parameters:
        -----------
        true_mean : float
            True population mean
        """
        # TODO: Implement continuous version of create_decision_correctness_df
        # For now, placeholder
        print(f"TODO: Implement decision_correctness for continuous (true_mean={true_mean})")
    
    # TODO: Plotting methods - implement later
    def plot_decision_rates(self, true_mean=None, viz_epitg="separate"):
        """TODO: Implement continuous version of plot_decision_rates"""
        print("TODO: Implement plot_decision_rates for continuous data")
    
    def plot_stop_iter_sample_rates(self, true_mean=None, title=None):
        """TODO: Implement continuous version of plot_stop_iter_sample_rates"""
        print("TODO: Implement plot_stop_iter_sample_rates for continuous data")
    
    def one_experiment_all_iterations(self, iexperiment, viz=True, true_mean=None, 
                                      xlim=None, method_names=None, ylim=None):
        """
        Process one experiment through all iterations without stopping.
        
        Parameters:
        -----------
        iexperiment : int
            Experiment index
        viz : bool
            Whether to visualize results (currently plots are TODO)
        true_mean : float, optional
            True population mean for visualization
        xlim : tuple, optional
            X-axis limits for PDF plotting (mean value range)
        method_names : list, optional
            Specific methods to visualize
        
        Returns:
        --------
        df_experiment_results : pd.DataFrame
            Results at each iteration
        """
        # TODO: Examine for generalisability - very similar to BinomialHypothesis
        print(self.experiments[iexperiment, :])
        df_experiment_results = continuous_sample_all_iterations_results(
            self.experiments[iexperiment, :], 
            self.precision_goal, 
            self.rope_min, 
            self.rope_max, 
            iteration_number=None
        )
        
        if viz:
            # TODO: Implement visualization methods
            self.viz_one_experiment_all_iterations(df_experiment_results, true_mean=true_mean, ylim=ylim)
            self.plot_experiment_pdf_methods(iexperiment, xlim=xlim, method_names=method_names)
        
        return df_experiment_results
    
    def viz_one_experiment_all_iterations(self, df_sample_results, true_mean=None, show_sample_mean=True,ylim=None):
        """Visualize one experiment's evolution through all iterations."""
        viz_one_sample_results_continuous(
            df_sample_results, 
            self.precision_goal, 
            self.rope_min, 
            self.rope_max, 
            true_mean=true_mean,
            show_sample_mean=show_sample_mean,
            ylim=ylim
        )
    
    def plot_experiment_pdf_methods(self, iexperiment, xlim=None, method_names=None):
        """TODO: Implement continuous version of plot_experiment_pdf_methods"""
        print("TODO: Implement plot_experiment_pdf_methods for continuous data")


def _update_iteration_tally(iteration_dict, iteration):
    """Helper to update iteration tallies for stopping criteria."""
    # TODO: Examine for generalisability - already generic!
    for this_iteration in range(iteration, len(iteration_dict)+1):
        iteration_dict[this_iteration] += 1


def booleans_to_rope_result(decision_accept, decision_reject_below, decision_reject_above):
    """Convert decision booleans to ROPE result string."""
    # TODO: Examine for generalisability - already generic!
    if decision_accept:
        return "within"
    elif decision_reject_below:
        return "below"
    elif decision_reject_above:
        return "above"


def stop_decision_continuous_multiple_methods(samples, rope_min, rope_max, precision_goal, 
                                              min_iter=30, viz=False):
    """
    Sequential hypothesis testing for continuous data using PitG, ePitG, and HDI+ROPE.
    
    Uses Student-t distribution for HDI calculation via CLT.
    
    Parameters:
    -----------
    samples : np.ndarray
        Shape (n_experiments, n_samples) continuous data
    rope_min : float
        Lower ROPE boundary
    rope_max : float
        Upper ROPE boundary
    precision_goal : float
        Target precision (HDI width) for stopping
    min_iter : int
        Minimum iterations before decisions allowed (default 30)
    viz : bool
        Print debug information
    
    Returns:
    --------
    method_stats : dict
        Statistics at stopping iteration for each method and experiment
    method_roperesult_iteration : dict
        Cumulative decision counts by iteration for each method
    """
    method_names = ["pitg", "epitg", "hdi_rope"]
    n_samples = samples.shape[1]
    
    # Stats at sample stop iteration
    method_stats = {method_name: {} for method_name in method_names}
    
    # For each iteration the number of samples that stopped there or before
    method_roperesult_iteration = {}
    rope_results = ["within", "below", "above"]
    for method in method_names:
        method_roperesult_iteration[method] = {}
        for rope_result in rope_results:
            method_roperesult_iteration[method][rope_result] = {
                iteration: 0 for iteration in range(1, n_samples + 1)
            }
    
    iteration_number = np.arange(1, n_samples + 1)
    
    for isample, sample in enumerate(samples):
        pitg_stopped = False
        hdi_rope_stopped = False
        
        # For continuous data: track cumulative statistics
        for iteration in iteration_number:
            final_iteration = iteration == iteration_number[-1]
            
            # Get data up to current iteration
            data_so_far = sample[:iteration]
            n = iteration
            sample_mean = np.mean(data_so_far)
            sample_std = np.std(data_so_far, ddof=1)  # ddof=1 for sample std
            
            # Calculate HDI using Student-t distribution
            # Requires n >= 2
            if n >= 2:
                hdi_min, hdi_max = continuous_hdi_ci_limits(sample_mean, sample_std, n)
            else:
                # Skip iteration if n < 2 (cannot compute t-distribution HDI)
                continue
            
            # Has the precision goal been achieved?
            precision_goal_achieved = (hdi_max - hdi_min) < precision_goal
            
            # Is the HDI conclusively within or outside the ROPE?
            decision_accept = (hdi_min >= rope_min) & (hdi_max <= rope_max)
            decision_reject_below = hdi_max < rope_min
            decision_reject_above = rope_max < hdi_min
            conclusive = decision_accept | decision_reject_above | decision_reject_below
            
            # Enforce minimum iteration requirement
            if min_iter is not None:
                if iteration < min_iter:
                    decision_accept = False
                    decision_reject_below = False
                    decision_reject_above = False
                    conclusive = False
                    precision_goal_achieved = False
            
            iteration_results = {
                "decision_iteration": iteration,
                "accept": decision_accept,
                "reject_below": decision_reject_below,
                "reject_above": decision_reject_above,
                "conclusive": conclusive,
                "inconclusive": not conclusive,
                "sample_mean": sample_mean,
                "sample_std": sample_std,
                "n": n,
                "hdi_min": hdi_min,
                "hdi_max": hdi_max,
                "precision_goal_achieved": precision_goal_achieved,
            }
            
            # PitG: Stop when precision goal achieved
            if precision_goal_achieved:
                
                # Update Precision Is The Goal Stop
                if pitg_stopped is False:
                    if conclusive:
                        rope_result = booleans_to_rope_result(decision_accept, decision_reject_below, decision_reject_above)
                        _update_iteration_tally(method_roperesult_iteration["pitg"][rope_result], iteration)
                    method_stats["pitg"][isample] = iteration_results
                    pitg_stopped = True
                
                # ePitG: Continue until conclusive after precision achieved
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
            
            # HDI+ROPE: Stop when conclusive (regardless of precision)
            elif conclusive & (hdi_rope_stopped is False):
                rope_result = booleans_to_rope_result(decision_accept, decision_reject_below, decision_reject_above)
                _update_iteration_tally(method_roperesult_iteration["hdi_rope"][rope_result], iteration)
                method_stats["hdi_rope"][isample] = iteration_results
                hdi_rope_stopped = True
            
            # Final iteration: record inconclusive results
            elif final_iteration:
                if isample not in method_stats["hdi_rope"]:
                    method_stats["hdi_rope"][isample] = iteration_results
                if isample not in method_stats["pitg"]:
                    method_stats["pitg"][isample] = iteration_results
                if isample not in method_stats["epitg"]:
                    method_stats["epitg"][isample] = iteration_results
                break
    
    return method_stats, method_roperesult_iteration


def continuous_sample_all_iterations_results(sample, precision_goal, rope_min, rope_max, 
                                            iteration_number=None):
    """
    Process one continuous sample through ALL iterations without stopping.
    
    Unlike stop_decision_continuous_multiple_methods, this does not stop when 
    criteria are met - it flags when objectives are achieved but continues.
    
    Parameters:
    -----------
    sample : np.ndarray
        1D array of continuous observations (one experiment)
    precision_goal : float
        Target precision (HDI width)
    rope_min : float
        Lower ROPE boundary
    rope_max : float
        Upper ROPE boundary
    iteration_number : np.ndarray, optional
        Custom iteration numbers. If None, uses 1 to len(sample)
    
    Returns:
    --------
    df_sample_results : pd.DataFrame
        Results at each iteration (where n >= 2)
    """
    # TODO: Examine for generalisability - similar structure to binomial version
    if iteration_number is None:
        iteration_number = np.arange(1, sample.shape[0] + 1)
    
    sample_results = {}
    for iteration in iteration_number:
        # Skip iterations where n < 2 (cannot compute t-distribution HDI)
        if iteration < 2:
            continue
        
        # Get data up to current iteration
        data_so_far = sample[:iteration]
        n = iteration
        sample_mean = np.mean(data_so_far)
        sample_std = np.std(data_so_far, ddof=1)  # ddof=1 for sample std
        
        # Calculate HDI using Student-t distribution
        hdi_min, hdi_max = continuous_hdi_ci_limits(sample_mean, sample_std, n)
        
        # Has the precision goal been achieved?
        precision_goal_achieved = (hdi_max - hdi_min) < precision_goal
        
        # Is the HDI conclusively within or outside the ROPE?
        decision_accept = (hdi_min >= rope_min) & (hdi_max <= rope_max)
        decision_reject_below = hdi_max < rope_min
        decision_reject_above = rope_max < hdi_min
        conclusive = decision_accept | decision_reject_above | decision_reject_below
        
        iteration_results = {
            "decision_iteration": iteration,
            "accept": decision_accept,
            "reject_below": decision_reject_below,
            "reject_above": decision_reject_above,
            "conclusive": conclusive,
            "inconclusive": not conclusive,
            "sample_mean": sample_mean,
            "sample_std": sample_std,
            "n": n,
            "hdi_min": hdi_min,
            "hdi_max": hdi_max,
            "goal_achieved": precision_goal_achieved,
        }
        
        sample_results[iteration] = iteration_results
    
    df_sample_results = stats_dict_to_df(sample_results, data_type='continuous')
    
    return df_sample_results


class ContinuousSimulation():
    """
    Generate synthetic continuous data from various distributions.
    Uses CLT-based approach where posterior of mean is Student-t distributed.
    """
    def __init__(self, distribution='normal', mean=0.0, variance=1.0,
                 n_samples=1500, n_experiments=500, seed=42):
        """
        Parameters:
        -----------
        distribution : str
            Distribution type: 'normal', 'exponential', 'uniform', 'lognormal'
        mean : float
            Target mean of the distribution
        variance : float
            Target variance of the distribution (note: for exponential, variance = mean²)
        n_samples : int
            Number of samples per experiment (sequential observations)
        n_experiments : int
            Number of independent experiments to generate
        seed : int
            Random seed for reproducibility
        """
        self.distribution = distribution.lower()
        self.mean = mean
        self.variance = variance
        self.n_samples = n_samples
        self.n_experiments = n_experiments
        self.seed = seed

        self.generate_experiments()

    def generate_experiments(self):
        print("Generating synthetic continuous data with parameter values:")
        print(f"{self.distribution}: distribution type")
        print(f"{self.mean:0.3f}: target mean")
        print(f"{self.variance:0.3f}: target variance")
        print(f"{self.n_experiments}: experiments")
        print(f"{self.n_samples}: sample size per experiment")

        np.random.seed(self.seed)

        if self.distribution == 'normal':
            # Normal(μ, σ²)
            std = np.sqrt(self.variance)
            self.experiments = np.random.normal(
                self.mean, std, [self.n_experiments, self.n_samples]
            )

        elif self.distribution == 'exponential':
            # Exponential: mean = 1/λ, variance = 1/λ²
            # So for exponential: variance = mean²
            scale = self.mean  # scale parameter = 1/λ = mean
            self.experiments = np.random.exponential(
                scale, [self.n_experiments, self.n_samples]
            )
            self.actual_variance = self.mean ** 2
            print(f"  Note: For exponential, variance = mean² = {self.actual_variance:0.3f}")

        elif self.distribution == 'uniform':
            # Uniform(a, b): mean = (a+b)/2, variance = (b-a)²/12
            # Solving for a and b given mean and variance:
            # width = sqrt(12 * variance)
            width = np.sqrt(12 * self.variance)
            a = self.mean - width / 2
            b = self.mean + width / 2
            self.experiments = np.random.uniform(
                a, b, [self.n_experiments, self.n_samples]
            )

        elif self.distribution == 'lognormal':
            # Log-Normal: If Y ~ LogNormal(μ, σ²), then ln(Y) ~ Normal(μ, σ²)
            # E[Y] = exp(μ + σ²/2)
            # Var[Y] = exp(2μ + σ²) * (exp(σ²) - 1)
            # Given target mean and variance, solve for μ and σ:
            m = self.mean
            v = self.variance
            sigma_squared = np.log(1 + v / (m ** 2))
            mu = np.log(m) - sigma_squared / 2
            sigma = np.sqrt(sigma_squared)

            self.experiments = np.random.lognormal(
                mu, sigma, [self.n_experiments, self.n_samples]
            )

        else:
            raise ValueError(
                f"Unknown distribution '{self.distribution}'. "
                f"Choose from: 'normal', 'exponential', 'uniform', 'lognormal'"
            )


def run_simulations_and_analysis_report(mean_true: float = 0.0,
                                        mean_null: float = 0.0,
                                        dmean: float = 0.5,
                                        distribution: str = 'normal',
                                        variance: float = 1.0,
                                        n_samples: int = 1500,
                                        n_experiments: int = 2000,
                                        seed: int = 42,
                                        rope_precision_fraction: float = 0.8,
                                        viz: bool = True):
    """
    Run complete simulation and analysis pipeline for continuous data.
    
    Generates synthetic continuous data, runs hypothesis testing with PitG/ePitG/HDI+ROPE,
    computes decision correctness, and optionally visualizes results.
    
    Parameters:
    -----------
    mean_true : float
        True population mean for data generation
    mean_null : float
        Null hypothesis mean
    dmean : float
        ROPE half-width
    distribution : str
        Distribution type: 'normal', 'exponential', 'uniform', 'lognormal'
    variance : float
        Target variance of the distribution
    n_samples : int
        Number of samples per experiment (sequential observations)
    n_experiments : int
        Number of independent experiments to generate
    seed : int
        Random seed for reproducibility
    rope_precision_fraction : float
        Precision goal as fraction of ROPE width
    viz : bool
        Whether to display plots and summary tables
    
    Returns:
    --------
    dict
        Dictionary with keys:
        - 'synth': ContinuousSimulation object
        - 'hypothesis': ContinuousHypothesis object
        - 'df_stats': DataFrame with aggregated statistics across methods
    """
    synth = ContinuousSimulation(
        distribution=distribution,
        mean=mean_true,
        variance=variance,
        n_experiments=n_experiments,
        n_samples=n_samples,
        seed=seed
    )
    
    hypothesis = ContinuousHypothesis(
        mean_null=mean_null,
        dmean=dmean,
        rope_precision_fraction=rope_precision_fraction
    )
    
    hypothesis.run_hypothesis_on_experiments(synth.experiments)
    hypothesis.decision_correctness(mean_true)
    
    if viz:
        hypothesis.plot_decision_rates(synth.mean)
        hypothesis.plot_stop_iter_sample_rates(true_mean=synth.mean, title=None)
    
    df_stats = report_success_rates_multiple_algos(
        hypothesis.method_df_stats.copy(),
        data_type='continuous',
        viz=viz
    )

    return {"synth": synth, "hypothesis": hypothesis, "df_stats": df_stats}