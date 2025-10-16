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
    poi_rate = np.exp(prediction[:,0])
    lik =  np.sum(scipy.stats.poisson.logpmf(true_values[:,0], poi_rate))
    return lik

def negbin_likelihood(prediction, # 2D array: inst x (prms)
                      true_values, # 2D array: ints x (successes, counts, time, sample_IDs)
                      sample_id=None, class_weight=None,
                      instance_weight=None,
                      lik_temp=1,
                      sig2=0
                      ):
    # mean
    # nb_mean = np.exp(prediction[:, 0])
    nb_mean = np.exp(prediction[:, 0])
    p = 1 / (1 + np.exp(-prediction[:, 1]))
    n = p * nb_mean / (1 - p)
    # "n = np.exp(prediction[:, 0])"
    lik =  np.sum(scipy.stats.nbinom.logpmf(true_values[:,0], n=n, p=p))
    return lik


def negbin_likelihood_base10(prediction, # 2D array: inst x (prms)
                      true_values, # 2D array: ints x (successes, counts, time, sample_IDs)
                      sample_id=None, class_weight=None,
                      instance_weight=None,
                      lik_temp=1,
                      sig2=0
                      ):
    nb_mean = 10**(prediction[:, 0])
    p = 1 / (1 + 10**(-prediction[:, 1]))
    n = p * nb_mean / (1 - p)
    lik =  np.sum(scipy.stats.nbinom.logpmf(true_values[:,0], n=n, p=p))
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


def negbin_acc(y, lab):
    acc = np.mean( (np.exp(y[: ,0]) - lab[:, 0])**2 )
    return acc

def poi_acc(y, lab):
    acc = np.mean( (np.exp(y[:, 0]) - lab[:, 0])**2 )
    return acc

def gamma_acc(y, lab):
    acc = np.mean( (np.exp(y[:,0]) - lab.flatten())**2 )

