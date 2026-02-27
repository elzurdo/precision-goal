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

    # If stopped early, p-value must be below threshold
    if stop_trial < n:
        assert recorded_pvalue < 0.05
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
