import numpy as np
import scipy.stats
import scipy.special
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
import pickle
small_number = 1e-10
import random, sys
from .BNN_files import *
from .BNN_lib import *
import os


def get_feature_summary(data, focal_features):
    """Get whether features are ohe/binary/ordinal/continuous and their min and max values"""
    num_features = len(focal_features)
    feature_summary = np.zeros((3, num_features))

    for i in range(num_features):
        ff = data[:, focal_features[i]]
        values, counts = np.unique(ff, return_counts=True)
        feature_summary[1, i] = np.nanmin(values)
        feature_summary[2, i] = np.nanmax(values)
        values_range = np.arange(feature_summary[1, i], feature_summary[2, i] + 1)
        feature_summary[0, i] = np.all(np.isin(values, values_range)) # 1 for binary/ordinal, 0 for continuous

    return feature_summary


def make_pdp_features(data, focal_features, steps_continuous=100):
    """Get the feature gradient along which we calculate the PDP"""
    feature_summary = get_feature_summary(data, focal_features)

    if np.sum(feature_summary[0, :] == 0) and len(focal_features) == 1:
        # Single continuous feature
        pdp_feat = np.linspace(feature_summary[1, 0], feature_summary[2, 0], num=steps_continuous).reshape(steps_continuous, 1)
    elif feature_summary[0, 0] == 1 and len(focal_features) == 1:
        # ordinal or binary
        M = int(feature_summary[2, 0])
        pdp_feat = np.linspace(feature_summary[1, 0], M, num=M + 1).reshape((M + 1, 1))
    else:
        # One-hot-encoded
        pdp_feat = np.eye(feature_summary.shape[1])

    return pdp_feat


def get_pdp(data,
            focal_features,
            estimation_mode,
            size_output,
            actFun,
            output_act_fun,
            weights,
            alphas,
            data_transform):
    """Get partial dependence output for a given focal feature"""
    pdp_features = make_pdp_features(data, focal_features)
    num_pdp_steps = pdp_features.shape[0]

    pdp = np.zeros((num_pdp_steps, size_output, 3))

    for n in range(num_pdp_steps):
        feat = np.copy(data)
        feat[:, focal_features] = pdp_features[n, :]
        pred = np.zeros((len(weights), data.shape[0], size_output))
        for i in range(len(weights)):
            actFun_i = actFun
            actFun_i.reset_prm(alphas[i])
            pred[i, :, :] = RunPredict(feat, weights[i],
                                       actFun=actFun_i,
                                       output_act_fun=output_act_fun,
                                       data_transform=data_transform)

        if estimation_mode == 'classification':
            pred = np.cumsum(pred, axis=2)

        pdp[n, :, 0] = np.mean(pred, axis=(0,1))
        # Should this be an HPD?
        probs_quantiles = np.quantile(np.mean(pred, axis=0), q=(0.025, 0.975), axis=0)
        pdp[n, :, 1] = probs_quantiles[0, :]
        pdp[n, :, 2] = probs_quantiles[1, :]

    return {'feature': pdp_features, 'pdp': pdp}


def pdp(pickle_file, pdp_features):
    bnn_obj, mcmc_obj, logger_obj = load_obj(pickle_file)
    post_samples = logger_obj._post_weight_samples

    # load posterior weights
    post_weights = [post_samples[i]['weights'] for i in range(len(post_samples))]
    post_alphas = [post_samples[i]['alphas'] for i in range(len(post_samples))]

    if bnn_obj._feature_indicators is not None:
        data_transform = data_transform_obj(bnn_obj._feature_indicators, bnn_obj._feature_means)
    else:
        data_transform = None

    pdp = []
    for p in pdp_features:
        p = get_pdp(bnn_obj._data, p,
                    bnn_obj._estimation_mode, bnn_obj._size_output,
                    bnn_obj._act_fun, bnn_obj._output_act_fun,
                    post_weights, post_alphas, data_transform)
        pdp.append(p)

    return pdp
