import sys
import os
import numpy as np
import pytest
from scipy.stats import binomtest

# Ensure the py/ directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))

from utils_experiments import (
    stop_decision_multiple_experiments_nhst,
    SEQUENCE_HANDPICKED,
)


def _str_sequence_to_array(s):
    """Convert a string of '0'/'1' characters to a 1D numpy int array."""
    return np.array([int(c) for c in s])


def test_handpicked_sequence_consistency():
    """The returned successes, trials, and p_value should be mutually consistent."""
    sequence_array = _str_sequence_to_array(SEQUENCE_HANDPICKED)
    experiments = sequence_array.reshape(1, -1)
    n = len(SEQUENCE_HANDPICKED)

    result = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5, alternative='two-sided',
    )

    exp = result["experiment_stop_results"]
    iter_counts = result["iteration_stopping_on_or_prior"]

    # Structural checks
    assert len(exp["successes"]) == 1
    assert len(exp["trials"]) == 1
    assert len(exp["p_value"]) == 1
    assert 1 <= exp["trials"][0] <= n

    stop_trial = exp["trials"][0]
    recorded_successes = exp["successes"][0]
    recorded_pvalue = exp["p_value"][0]

    # Gold-master: known outcome for SEQUENCE_HANDPICKED must not change
    assert recorded_successes == 45
    assert stop_trial == 72
    assert recorded_pvalue == pytest.approx(0.04437092000802569)

    # Successes must match the cumulative sum up to stop_trial
    expected_successes = int(sequence_array[:stop_trial].sum())
    assert recorded_successes == expected_successes

    # p-value must match an independent binom_test call at the stop point
    expected_pvalue = binomtest(expected_successes, n=stop_trial, p=0.5, alternative='two-sided').pvalue
    assert abs(recorded_pvalue - expected_pvalue) < 1e-12

    # If stopped early, p-value must be at or below threshold
    if stop_trial < n:
        assert recorded_pvalue <= 0.05
        # iteration_stopping_on_or_prior should be 0 before stop, 1 from stop onward
        for it in range(1, stop_trial):
            assert iter_counts[it] == 0
        for it in range(stop_trial, n + 1):
            assert iter_counts[it] == 1
    else:
        # Did not stop: p-value >= threshold, counts all zero
        assert recorded_pvalue >= 0.05
        for it in range(1, n + 1):
            assert iter_counts[it] == 0


def test_all_successes_rejects_early():
    """A sequence of all 1s should reject H0: p=0.5 well before the end."""
    n = 50
    experiments = np.ones((1, n), dtype=int)

    result = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5,
    )

    exp = result["experiment_stop_results"]
    assert exp["trials"][0] < n
    assert exp["p_value"][0] < 0.05
    assert exp["successes"][0] == exp["trials"][0]  # every toss was a success


def test_all_failures_rejects_early():
    """A sequence of all 0s should also reject H0: p=0.5 early (two-sided)."""
    n = 50
    experiments = np.zeros((1, n), dtype=int)

    result = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5,
    )

    exp = result["experiment_stop_results"]
    assert exp["trials"][0] < n
    assert exp["p_value"][0] < 0.05
    assert exp["successes"][0] == 0


def test_balanced_sequence_does_not_reject():
    """An alternating 0-1 sequence (~50% rate) should not reject H0: p=0.5."""
    n = 200
    experiments = np.array([[i % 2 for i in range(n)]])

    result = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5,
    )

    exp = result["experiment_stop_results"]
    assert exp["trials"][0] == n  # went through all observations
    assert exp["p_value"][0] >= 0.05

    # All iteration counts should be zero
    iter_counts = result["iteration_stopping_on_or_prior"]
    assert all(v == 0 for v in iter_counts.values())


def test_multiple_experiments_counts_accumulate():
    """Two experiments: one rejects early, one doesn't. Counts should reflect this."""
    n = 50
    exp_rejects = np.ones(n, dtype=int)               # all 1s — rejects quickly
    exp_survives = np.array([i % 2 for i in range(n)]) # balanced — won't reject
    experiments = np.vstack([exp_rejects, exp_survives])

    result = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5,
    )

    exp = result["experiment_stop_results"]
    iter_counts = result["iteration_stopping_on_or_prior"]

    # First experiment should stop early, second should not
    assert exp["trials"][0] < n
    assert exp["trials"][1] == n

    stop_iter = exp["trials"][0]

    # Before the first experiment stops, count is 0; after, it's 1 (second never stops)
    for it in range(1, stop_iter):
        assert iter_counts[it] == 0
    for it in range(stop_iter, n + 1):
        assert iter_counts[it] == 1


def test_return_structure():
    """Verify the return dict has the expected keys and list lengths."""
    n_exp, n_obs = 3, 40
    np.random.seed(99)
    experiments = np.random.binomial(1, 0.5, (n_exp, n_obs))

    result = stop_decision_multiple_experiments_nhst(experiments)

    assert set(result.keys()) == {"iteration_stopping_on_or_prior", "experiment_stop_results"}

    iter_counts = result["iteration_stopping_on_or_prior"]
    assert set(iter_counts.keys()) == set(range(1, n_obs + 1))

    exp = result["experiment_stop_results"]
    for key in ("successes", "trials", "p_value"):
        assert len(exp[key]) == n_exp


def test_iteration_counts_monotonically_nondecreasing():
    """Cumulative stopping counts should never decrease as iteration increases."""
    np.random.seed(123)
    experiments = np.random.binomial(1, 0.7, (20, 100))

    result = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5,
    )

    counts = [result["iteration_stopping_on_or_prior"][it] for it in range(1, 101)]
    for i in range(1, len(counts)):
        assert counts[i] >= counts[i - 1]


def test_threshold_zero_stops_nothing():
    """With p_value_thresh=0, no experiment should ever stop early."""
    np.random.seed(77)
    n_obs = 60
    experiments = np.random.binomial(1, 0.9, (5, n_obs))

    result = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.0, success_rate_null=0.5,
    )

    for trial_count in result["experiment_stop_results"]["trials"]:
        assert trial_count == n_obs

    assert all(v == 0 for v in result["iteration_stopping_on_or_prior"].values())


def test_threshold_one_stops_at_first_iteration():
    """With p_value_thresh=1.0, every experiment should stop at iteration 1."""
    np.random.seed(42)
    experiments = np.random.binomial(1, 0.5, (10, 50))

    result = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=1.0, success_rate_null=0.5,
    )

    for trial_count in result["experiment_stop_results"]["trials"]:
        assert trial_count == 1


def test_alternative_greater_all_ones_rejects():
    """With alternative='greater', all 1s should reject quickly."""
    experiments = np.ones((1, 30), dtype=int)

    result = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5, alternative='greater',
    )

    assert result["experiment_stop_results"]["trials"][0] < 30
    assert result["experiment_stop_results"]["p_value"][0] < 0.05


def test_alternative_greater_all_zeros_does_not_reject():
    """With alternative='greater', all 0s should NOT reject (wrong tail)."""
    experiments = np.zeros((1, 30), dtype=int)

    result = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5, alternative='greater',
    )

    assert result["experiment_stop_results"]["trials"][0] == 30


def test_identical_experiments_produce_identical_results():
    """Three copies of the same sequence should yield identical per-experiment results."""
    sequence_array = _str_sequence_to_array(SEQUENCE_HANDPICKED)
    experiments = np.vstack([sequence_array] * 3)

    result = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5,
    )

    exp = result["experiment_stop_results"]
    assert exp["trials"][0] == exp["trials"][1] == exp["trials"][2]
    assert exp["successes"][0] == exp["successes"][1] == exp["successes"][2]
    assert exp["p_value"][0] == exp["p_value"][1] == exp["p_value"][2]

    # Iteration counts should be exactly 3× what a single experiment gives
    single = stop_decision_multiple_experiments_nhst(
        sequence_array.reshape(1, -1), p_value_thresh=0.05, success_rate_null=0.5,
    )
    n = len(SEQUENCE_HANDPICKED)
    for it in range(1, n + 1):
        assert result["iteration_stopping_on_or_prior"][it] == 3 * single["iteration_stopping_on_or_prior"][it]
