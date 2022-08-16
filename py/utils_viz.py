import matplotlib.pyplot as plt
import numpy as np

from utils_stats import (CI_FRACTION,
                         successes_failures_to_hdi_ci_limits,
                         get_success_rates,
                         beta,
                         test_value
                         )

def plot_success_rates(success, failure, ci_fraction=CI_FRACTION,
                       min_psuccess=0.85, max_psucess=1.,d_psuccess=0.0001,
                       color="purple", format='-', label=None, fill=False, display_ci=True,
                       alpha=1., factor=1., ci_label=None,
                       xlabel="success rate",
                       ylabel="probability distribution function", xmin=None, xmax=None):

    ci_min, ci_max = successes_failures_to_hdi_ci_limits(success, failure, ci_fraction=ci_fraction)

    _, p_success = get_success_rates(d_success = d_psuccess, min_range=0., max_range=1., including_max=True)
    beta_pdf = beta.pdf(p_success, test_value(success), test_value(failure))

    plt.plot(p_success, beta_pdf * factor, format, linewidth=3, color=color, label=label, alpha=alpha)

    xmin_ = np.max([p_success.min(), ci_min * 0.95])
    if xmin is not None:
        xmin_ = np.min([xmin_, xmin])
        xmin_ = np.max([xmin_, 0.])
    xmax_ = np.min([p_success.max(), ci_max * 1.05])
    if xmax is not None:
        xmax_ = np.max([xmax_, xmax])
        xmax_ = np.min([xmax_, 1.])



    plt.xlim(xmin_, xmax_)

    if display_ci == True:
        plt.plot([ci_min, ci_min],
                 [0, beta.pdf(ci_min, success, failure) * factor],
                 "--", color=color, alpha=alpha, label=ci_label)
        plt.plot([ci_max, ci_max],
                 [0, beta.pdf(ci_max, success, failure) * factor],
                 "--", color=color, alpha=alpha)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)