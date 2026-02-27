# Notebooks

## `notebooks` folder

* `notebooks/binomial_experiment_analysis.ipynb` - binomial results for paper are here
* `notebooks/continuous_experiment_analysis.ipynb` - same but for continuous

## py folder

* `binomial_pvalue.py` - used to conduct p-value tests of NHST hypothesis testing
of binomial data. It is used to show the lack of power of the p-value to accept
the null hypothesis when it is correct.

* `notebook_coin_toss.py` - used to conduct hypothesis testing for binomial testing
using different types of stop criterion: nhst, hdi+rope, percision is the goal
and the first attempt at enahnced percision is the goal
(then called ultimate or conservative).
(I suspect that `notebook_coin_toss_experiments.py` precedes it and should be
considered legacy.)

* `beta_uncertainties.py` - analytic solutions for uncertainties of beta functions.

**Legacy Notebooks**

* `notebook_explore.py` - visualising beta functions of two different success/failure
pairings.

* `notebook_nhst_explore.py` - appears to be legacy of `binomial_pvalue.py`

* `precision_goal_numerical.py` - first attempt at BEST for binomial data.
I'm not sure why I called it "precision goal".
