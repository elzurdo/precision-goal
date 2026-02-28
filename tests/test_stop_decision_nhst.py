import sys
import os
import numpy as np
import pytest
from scipy.stats import binomtest

# Ensure the py/ directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))

from utils_experiments import (
    stop_decision_multiple_experiments_nhst,
    BinaryPvalueAccounting,
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


# ---------------------------------------------------------------------------
# BinaryPvalueAccounting tests
# ---------------------------------------------------------------------------

def test_bpva_pvalue_matches_binomtest():
    """successes_n_to_pvalue should return the same value as a direct binomtest call."""
    acct = BinaryPvalueAccounting(success_rate_null=0.5, alternative='two-sided')
    successes, n = 45, 72
    expected = binomtest(successes, n=n, p=0.5, alternative='two-sided').pvalue
    assert acct.successes_n_to_pvalue(successes, n) == pytest.approx(expected)


def test_bpva_first_call_populates_cache():
    """After the first call the pair should be in the cache with counter == 1."""
    acct = BinaryPvalueAccounting(success_rate_null=0.5, alternative='two-sided')
    pair = (10, 20)
    assert pair not in acct.dict_successes_n_pvalue

    acct.successes_n_to_pvalue(*pair)

    assert pair in acct.dict_successes_n_pvalue
    assert acct.dict_successes_n_counter[pair] == 1


def test_bpva_repeated_call_increments_counter():
    """Calling with the same pair twice should increment the counter, not recompute."""
    acct = BinaryPvalueAccounting(success_rate_null=0.5, alternative='two-sided')
    pair = (10, 20)

    acct.successes_n_to_pvalue(*pair)
    acct.successes_n_to_pvalue(*pair)
    acct.successes_n_to_pvalue(*pair)

    assert acct.dict_successes_n_counter[pair] == 3


def test_bpva_repeated_call_returns_same_value():
    """The cached value must equal the freshly computed one."""
    acct = BinaryPvalueAccounting(success_rate_null=0.5, alternative='two-sided')
    pair = (15, 30)

    first = acct.successes_n_to_pvalue(*pair)
    second = acct.successes_n_to_pvalue(*pair)

    assert first == second


def test_bpva_distinct_pairs_cached_independently():
    """Different (successes, n) pairs must have independent cache entries."""
    acct = BinaryPvalueAccounting(success_rate_null=0.5, alternative='two-sided')
    pair_a = (10, 20)
    pair_b = (11, 20)

    pv_a = acct.successes_n_to_pvalue(*pair_a)
    pv_b = acct.successes_n_to_pvalue(*pair_b)

    assert pair_a in acct.dict_successes_n_pvalue
    assert pair_b in acct.dict_successes_n_pvalue
    assert pv_a != pv_b  # different counts → different p-values
    assert acct.dict_successes_n_counter[pair_a] == 1
    assert acct.dict_successes_n_counter[pair_b] == 1


def test_bpva_alternative_greater():
    """alternative='greater' should produce a different p-value than 'two-sided'."""
    pair = (15, 20)
    acct_two = BinaryPvalueAccounting(success_rate_null=0.5, alternative='two-sided')
    acct_gt = BinaryPvalueAccounting(success_rate_null=0.5, alternative='greater')

    pv_two = acct_two.successes_n_to_pvalue(*pair)
    pv_gt = acct_gt.successes_n_to_pvalue(*pair)

    expected_two = binomtest(*pair, p=0.5, alternative='two-sided').pvalue
    expected_gt = binomtest(*pair, p=0.5, alternative='greater').pvalue

    assert pv_two == pytest.approx(expected_two)
    assert pv_gt == pytest.approx(expected_gt)
    assert pv_two != pv_gt


def test_bpva_integration_same_results_as_without():
    """stop_decision_multiple_experiments_nhst with BinaryPvalueAccounting must
    produce identical results to calling without it."""
    sequence_array = _str_sequence_to_array(SEQUENCE_HANDPICKED)
    experiments = sequence_array.reshape(1, -1)

    result_plain = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5,
    )

    acct = BinaryPvalueAccounting(success_rate_null=0.5, alternative='two-sided')
    result_cached = stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5,
        binary_pvalue_accounting=acct,
    )

    assert result_plain["experiment_stop_results"] == result_cached["experiment_stop_results"]
    assert result_plain["iteration_stopping_on_or_prior"] == result_cached["iteration_stopping_on_or_prior"]


def test_bpva_integration_cache_populated_after_run():
    """After a run the cache should contain entries for every (successes, n) pair seen."""
    sequence_array = _str_sequence_to_array(SEQUENCE_HANDPICKED)
    experiments = sequence_array.reshape(1, -1)

    acct = BinaryPvalueAccounting(success_rate_null=0.5, alternative='two-sided')
    stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5,
        binary_pvalue_accounting=acct,
    )

    # Cache should be non-empty
    assert len(acct.dict_successes_n_pvalue) > 0

    # Every cached p-value must match a fresh binomtest call
    for (successes, n), cached_pv in acct.dict_successes_n_pvalue.items():
        expected = binomtest(successes, n=n, p=0.5, alternative='two-sided').pvalue
        assert cached_pv == pytest.approx(expected)


def test_bpva_integration_second_run_uses_cache():
    """Running the same experiments twice with one accounting object should
    result in hit counts > 0 for at least some pairs on the second run."""
    sequence_array = _str_sequence_to_array(SEQUENCE_HANDPICKED)
    experiments = sequence_array.reshape(1, -1)

    acct = BinaryPvalueAccounting(success_rate_null=0.5, alternative='two-sided')

    stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5,
        binary_pvalue_accounting=acct,
    )
    counts_after_first = dict(acct.dict_successes_n_counter)

    stop_decision_multiple_experiments_nhst(
        experiments, p_value_thresh=0.05, success_rate_null=0.5,
        binary_pvalue_accounting=acct,
    )

    # Every pair seen in the first run should have its counter incremented in the second
    for pair, count in counts_after_first.items():
        assert acct.dict_successes_n_counter[pair] > count
