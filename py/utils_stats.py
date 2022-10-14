from scipy.optimize import fmin
from scipy.stats import beta, binom_test # binom_test is binomtest (in later versions ...)
import numpy as np

CI_FRACTION = 0.95
MIN_COUNTS = 1.

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


def sequence_to_sequential_pvalues(sequence, success_rate_null=0.5):
    p_values = []
    
    for idx, successes in enumerate(sequence.cumsum()):
        p_value = binom_test(successes, n=idx + 1, p=success_rate_null, alternative='two-sided') # alternative {‘two-sided’, ‘greater’, ‘less’},
        p_values.append(p_value)
    
    p_values = np.array(p_values)
    
    return p_values