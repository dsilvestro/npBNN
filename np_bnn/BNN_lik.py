import numpy as np
import scipy.stats

# lik functions
def poi_likelihood(prediction, # 2D array: inst x (prms)
                   true_values, # 2D array: ints x (successes, counts, time, sample_IDs)
                   sample_id=None, class_weight=None,
                   instance_weight=None,
                   lik_temp=1,
                   sig2=0
                   ):
    poi_rate = np.exp(prediction)
    lik =  np.sum(scipy.stats.poisson.logpmf(true_values, poi_rate))
    return lik

def negbin_likelihood(prediction, # 2D array: inst x (prms)
                      true_values, # 2D array: ints x (successes, counts, time, sample_IDs)
                      sample_id=None, class_weight=None,
                      instance_weight=None,
                      lik_temp=1,
                      sig2=0
                      ):
    n = np.exp(prediction[:, 0])
    p = 1 / (1 + np.exp(-prediction[:, 1]))
    lik =  np.sum(scipy.stats.nbinom.logpmf(true_values, n=n, p=p))
    return lik

def gamma_likelihood(prediction, # 2D array: inst x (prms)
                     true_values, # 2D array: ints x (successes, counts, time, sample_IDs)
                     sample_id=None, class_weight=None,
                     instance_weight=None,
                     lik_temp=1,
                     sig2=0
                     ):
    a = np.exp(prediction[:, 0])
    b = np.exp(prediction[:, 1])
    lik =  np.sum(scipy.stats.gamma.logpdf(true_values, a, b))
    return lik

