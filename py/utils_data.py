import numpy as np

def generate_biomial_sequence(success_rate=0.5, n_samples=1500, n_experiments=1, seed=None, ):
    if seed:
        np.random.seed(seed)

    if n_experiments == 1:
        n_samples_shape = n_samples
    else:
        n_samples_shape = [n_experiments, n_samples]

    return np.random.binomial(1, success_rate, n_samples_shape)