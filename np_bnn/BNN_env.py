import pickle
import sys
import numpy as np
import scipy.stats
from  np_bnn.BNN_lib import *
import np_bnn.BNN_files
import random, numpy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


class npBNN():
    def __init__(self, dat, n_nodes=[50, 5],
                 use_bias_node=1, init_std=0.1, p_scale=1, prior_ind1=0.5,
                 prior_f=1, hyper_p=0, freq_indicator=0, w_bound=np.infty,
                 pickle_file="", seed=1234, use_class_weights=0, actFun=genReLU()):
        # prior_f: 0) uniform 1) normal 2) cauchy
        # to change the boundaries of a uniform prior use -p_scale
        # hyper_p: 0) no hyperpriors, 1) 1 per layer, 2) 1 per input node, 3) 1 per node
        # freq_indicator -> update freq indicators: 0) no indicators, 0-1
        data, labels, test_data, test_labels = dat['data'], dat['labels'], dat['test_data'], dat['test_labels']
        self._seed = seed
        np.random.seed(self._seed)
        self._data = data
        self._labels = labels.astype(int)
        self._test_data = test_data
        if len(test_labels) > 0:
            self._test_labels = test_labels.astype(int)
        else:
            self._test_labels = []
        self._size_output = len(np.unique(self._labels))
        self._init_std = init_std
        self._n_layers = len(n_nodes) + 1
        self._n_nodes = n_nodes
        self._use_bias_node = use_bias_node
        if use_bias_node:
            self._data = np.c_[np.ones(self._data.shape[0]), self._data]
            if len(test_labels) > 0:
                self._test_data = np.c_[np.ones(self._test_data.shape[0]), self._test_data]
            else:
                self._test_data = []

        self._n_samples = self._data.shape[0]
        self._n_features = self._data.shape[1]
        self._sample_id = np.arange(self._n_samples)
        self._w_bound = w_bound
        self._freq_indicator = freq_indicator
        self._hyper_p = hyper_p
        self._sample_id = np.arange(self._n_samples)
        self._prior = prior_f
        self._p_scale = p_scale
        self._prior_ind1 = prior_ind1

        # reset labels
        self._labels_reset = np.zeros(len(self._labels)).astype(int)
        self._test_labels_reset = np.zeros(len(self._test_labels)).astype(int)
        j = 0
        for i in np.unique(self._labels):
            self._labels_reset[self._labels == i] = j
            self._test_labels_reset[self._test_labels == i] = j
            j += 1

        if use_class_weights:
            class_counts = np.unique(self._labels_reset, return_counts=True)[1]
            self._class_w = 1 / (class_counts / np.max(class_counts))
            print("Using class weights:", self._class_w)
        else:
            self._class_w = []

        # init weights
        if pickle_file == "":
            # 1st layer
            w_layers = [np.random.normal(0, self._init_std, (self._n_nodes[0], self._n_features))]
            # add hidden layers
            for i in range(1, self._n_layers - 1):
                w_layers.append(np.random.normal(0, self._init_std, (self._n_nodes[i], self._n_nodes[i - 1])))
            # last layer
            w_layers.append(np.random.normal(0, self._init_std, (self._size_output, self._n_nodes[-1])))
        else:
            post_samples = np_bnn.BNN_files.load_obj(pickle_file)
            post_weights = [post_samples[i]['weights'] for i in range(len(post_samples))]
            w_layers = post_weights[-1]
        self._w_layers = w_layers

        self._indicators = np.ones(self._w_layers[0].shape)
        if pickle_file == "":
            self._act_fun = actFun
        else:
            post_alphas = [post_samples[i]['alphas'] for i in range(len(post_samples))]
            self._act_fun = genReLU(prm=post_alphas[-1])


        # init prior function
        if self._prior == 0:
            'Uniform'
            self._w_bound = self._p_scale
        else:
            if self._prior == 1:
                'Normal'
                self._prior_f = scipy.stats.norm.logpdf
            if self._prior == 2:
                'Cauchy'
                self._prior_f = scipy.stats.cauchy.logpdf
            elif self._prior == 3:
                'Laplace'
                self._prior_f = scipy.stats.laplace.logpdf
            else:
                print('Using default prior N(0,s)')
                self._prior_f = scipy.stats.norm.logpdf
        # init prior scales: will be updated if hyper-priors
        self._prior_scale = np.ones(self._n_layers) * self._p_scale

        if len(self._test_data) > 0:
            print("\nTraining set:", self._n_samples, "test set:", self._test_data.shape[0])
        else:
            print("\nTraining set:", self._n_samples, "test set:", None)
        print("Number of features:", self._n_features)
        
        n_params = np.sum(np.array([np.size(i) for i in self._w_layers]))
        if self._act_fun._trainable:
            n_params += self._n_layers
        print("N. of parameters:", n_params)
        for w in self._w_layers: print(w.shape)

    # init prior functions
    def calc_prior(self, w=0, ind=[]):
        if w == 0:
            w = self._w_layers
        if len(ind) == 0:
            ind = self._indicators
        if self._prior == 0:
            logPrior = 0
        else:
            logPrior = 0
            for i in range(self._n_layers):
                logPrior += np.sum(self._prior_f(w[i], 0, scale=self._prior_scale[i]))
        if self._freq_indicator:
            logPrior += np.sum(ind) * np.log(self._prior_ind1) + \
                        (self._indicators.size - np.sum(ind)) * np.log(1 - self._prior_ind1)
        return logPrior

    def sample_prior_scale(self):
        if self._prior != 1:
            print("Hyper-priors available only for Normal priors.")
            quit()
        if self._hyper_p == 1:
            '1 Hyp / layer'
            prior_scale = list()
            for x in self._w_layers:
                prior_scale.append(GibbsSampleNormStdGammaVector(x.flatten()))
            self._prior_scale = prior_scale
        elif self._hyper_p == 2:
            '1 Hyp / input node / layer'
            self._prior_scale = [np.ones(w.shape[1]) for w in self._w_layers]
            prior_scale = list()
            for x in self._w_layers:
                prior_scale.append(GibbsSampleNormStdGamma2D(x))
            self._prior_scale = prior_scale
        elif self._hyper_p == 3:
            '1 Hyp / weight / layer'
            self._prior_scale = [np.ones(w.shape) for w in self._w_layers]
            prior_scale = list()
            for x in self._w_layers:
                prior_scale.append(GibbsSampleNormStdGammaONE(x))
            self._prior_scale = prior_scale
        else:
            pass

    def reset_weights(self, w):
        self._w_layers = w

    def reset_indicators(self, ind):
        self._indicators = ind

    def update_data(self, data_dict):
        self._data = data_dict['data']
        self._labels = data_dict['labels']
        self._test_data = data_dict['test_data']
        self._test_labels = data_dict['test_labels']


class MCMC():
    def __init__(self, bnn_obj, update_f=None, update_ws=None,
                 temperature=1, n_iteration=100000, sampling_f=100, print_f=1000, n_post_samples=1000,
                 update_function=UpdateNormal, sample_from_prior=0, run_ID="", init_additional_prob=0,
                 likelihood_tempering=1, mcmc_id=0, randomize_seed=False):
        if update_ws is None:
            update_ws = [0.075] * bnn_obj._n_layers
        if update_f is None:
            update_f = [0.05] * bnn_obj._n_layers
        if run_ID == "":
            self._runID = bnn_obj._seed
        else:
            self._runID = run_ID
        self._update_f = update_f
        self._update_ws = [np.ones(bnn_obj._w_layers[i].shape) * update_ws[i] for i in range(bnn_obj._n_layers)]
        self._update_n = [np.round(bnn_obj._w_layers[i].size * update_f[i]).astype(int) for i in
                          range(bnn_obj._n_layers)]
        self._temperature = temperature
        self._n_iterations = n_iteration
        self._sampling_f = sampling_f
        self._print_f = print_f
        self._current_iteration = 0
        self._y = RunPredict(bnn_obj._data, bnn_obj._w_layers, bnn_obj._act_fun)
        if sample_from_prior:
            self._logLik = 0
        else:
            self._logLik = calc_likelihood(self._y,
                                           bnn_obj._labels_reset,
                                           bnn_obj._sample_id,
                                           bnn_obj._class_w,
                                           likelihood_tempering)
        self._logPrior = bnn_obj.calc_prior() + init_additional_prob
        self._logPost = self._logLik + self._logPrior
        self._accuracy = CalcAccuracy(self._y, bnn_obj._labels)
        if len(bnn_obj._test_data) > 0:
            self._y_test = RunPredictInd(bnn_obj._test_data, bnn_obj._w_layers, bnn_obj._indicators,bnn_obj._act_fun)
            self._test_accuracy = CalcAccuracy(self._y_test, bnn_obj._test_labels)
        else:
            self._y_test = []
            self._test_accuracy = 0
        self._label_freq = CalcLabelFreq(self._y)
        self.update_function = update_function
        self._sample_from_prior = sample_from_prior
        self._last_accepted = 1
        self._lik_temp = likelihood_tempering
        self._mcmc_id = mcmc_id
        self._randomize_seed = randomize_seed
        self._rs = RandomState(MT19937(SeedSequence(1234)))
        self._counter = 0
        self._n_post_samples = n_post_samples
        self._accepted_states = 0

    def mh_step(self, bnn_obj, additional_prob=0, return_bnn=False):
        if self._randomize_seed:
            self._rs = RandomState(MT19937(SeedSequence(self._current_iteration + self._mcmc_id)))

        hastings = 0
        w_layers_prime = []
        tmp = bnn_obj._data + 0
        indicators_prime = bnn_obj._indicators + 0
        
        # if trainable prm in activation function
        if bnn_obj._act_fun._trainable:
            prm_tmp, _, h = UpdateNormal1D(bnn_obj._act_fun._acc_prm, d=0.05, n=1, Mb=1, mb=0, rs=self._rs)
            r = 10
            additional_prob += np.log(r) * -np.sum(prm_tmp)*r # aka exponential Exp(r)
            hastings += h
            bnn_obj._act_fun.reset_prm(prm_tmp)
            
        for i in range(bnn_obj._n_layers):
            if np.random.random() > bnn_obj._freq_indicator or i > 0:
                update, indx, h = self.update_function(bnn_obj._w_layers[i], d=self._update_ws[i], n=self._update_n[i],
                                                    Mb=bnn_obj._w_bound, mb=-bnn_obj._w_bound, rs=self._rs)
                w_layers_prime.append(update)
                hastings += h
            else:
                w_layers_prime.append(bnn_obj._w_layers[i] + 0)
                indicators_prime = UpdateBinomial(bnn_obj._indicators, self._update_f[3], bnn_obj._indicators.shape)
            if i == 0:
                w_layers_prime_temp = w_layers_prime[i] * indicators_prime
            else:
                w_layers_prime_temp = w_layers_prime[i]
            if i < bnn_obj._n_layers-1:
                tmp = RunHiddenLayer(tmp, w_layers_prime_temp,bnn_obj._act_fun, i)
            else:
                tmp = RunHiddenLayer(tmp, w_layers_prime_temp, False, i)
        y_prime = SoftMax(tmp)

        logPrior_prime = bnn_obj.calc_prior(w=w_layers_prime, ind=indicators_prime) + additional_prob
        if self._sample_from_prior:
            logLik_prime = 0
        else:
            # TODO: expose self._lik_temp as parameter that can be estimated by MCMC
            logLik_prime = calc_likelihood(y_prime,
                                           bnn_obj._labels_reset,
                                           bnn_obj._sample_id,
                                           bnn_obj._class_w,
                                           self._lik_temp)
        logPost_prime = logLik_prime + logPrior_prime
        rrr = np.log(self._rs.random())
        if (logPost_prime - self._logPost) * self._temperature + hastings >= rrr:
            # print(logPost_prime, self._logPost)
            bnn_obj.reset_weights(w_layers_prime)
            bnn_obj.reset_indicators(indicators_prime)
            bnn_obj._act_fun.reset_accepted_prm()
            self._logPost = logPost_prime
            self._logLik = logLik_prime
            self._logPrior = logPrior_prime
            self._y = y_prime
            self._accuracy = CalcAccuracy(self._y, bnn_obj._labels_reset)
            self._label_freq = CalcLabelFreq(self._y)
            if len(bnn_obj._test_data) > 0:
                self._y_test = RunPredictInd(bnn_obj._test_data, bnn_obj._w_layers,
                                             bnn_obj._indicators, bnn_obj._act_fun)
                self._test_accuracy = CalcAccuracy(self._y_test, bnn_obj._test_labels_reset)
            else:
                self._y_test = []
                self._test_accuracy = 0
            self._last_accepted = 1
            self._accepted_states += 1
        else:
            self._last_accepted = 0

        self._current_iteration += 1
        if return_bnn:
            return bnn_obj, self

    def gibbs_step(self, bnn_obj):
        bnn_obj.sample_prior_scale()
        self._logPrior = bnn_obj.calc_prior()
        self._logPost = self._logLik + self._logPrior
        self._current_iteration += 1

    def reset_update_n(self, n):
        self._update_n = n
    
    def reset_temperature(self,temp):
        self._temperature = temp
    

class postLogger():
    def __init__(self,
                 bnn_obj,
                 filename="BNN",
                 wdir="",
                 sample_from_prior=0,
                 add_prms=None,
                 continue_logfile=False,
                 log_all_weights=0):
        
        wlog, logfile, w_file, wweight = np_bnn.BNN_files.init_output_files(bnn_obj, filename, sample_from_prior,
                                                                            outpath=wdir, add_prms=add_prms,
                                                                            continue_logfile=continue_logfile,
                                                                            log_all_weights=log_all_weights)
        self._wlog = wlog
        self._logfile = logfile
        self._w_file = w_file
        self._wweight = wweight
        self.log_all_weights = log_all_weights
        self._counter = 0
        self._list_post_weights = list()

    def reset_counter(self):
        self._counter = 0

    def update_counter(self):
        self._counter += 1

    def log_sample(self, bnn_obj, mcmc_obj, add_prms=None):
        row = [mcmc_obj._current_iteration, mcmc_obj._logPost, mcmc_obj._logLik, mcmc_obj._logPrior,
               mcmc_obj._accuracy, mcmc_obj._test_accuracy] + list(mcmc_obj._label_freq)
        for i in range(bnn_obj._n_layers):
            row = row + [np.mean(bnn_obj._w_layers[i]), np.std(bnn_obj._w_layers[i])]
            if bnn_obj._hyper_p:
                if bnn_obj._hyper_p == 1:
                    row.append(bnn_obj._prior_scale[i])
                else:
                    row.append(np.mean(bnn_obj._prior_scale[i]))
        if bnn_obj._freq_indicator > 0:
            row.append(np.mean(bnn_obj._indicators))
        if add_prms:
            row = row + add_prms
        if bnn_obj._act_fun._trainable:
            row = row + list(bnn_obj._act_fun._acc_prm)
        row.append(mcmc_obj._accepted_states / mcmc_obj._current_iteration)
        row.append(mcmc_obj._mcmc_id)
        self._wlog.writerow(row)
        self._logfile.flush()

    def log_weights(self, bnn_obj, mcmc_obj, add_prms=None):
        # print(mcmc_obj._current_iteration, self._counter, len(self._list_post_weights))
        if not self.log_all_weights:
            if bnn_obj._freq_indicator:
                tmp = list()
                tmp.append(bnn_obj._w_layers[0] * bnn_obj._indicators)
                for i in range(1, bnn_obj._n_layers):
                    tmp.append(bnn_obj._w_layers[i])
            else:
                tmp = bnn_obj._w_layers
            post_prm = {'weights': tmp}
            # a ReLU prms
            post_prm['alphas'] = list(bnn_obj._act_fun._acc_prm)

            if add_prms:
                post_prm['additional_prm'] = list(add_prms)

            # print(mcmc_obj._current_iteration, self._counter, len(self._list_post_weights))
            if len(self._list_post_weights) < mcmc_obj._n_post_samples:
                self._list_post_weights.append(post_prm)
            else:
                self._list_post_weights[self._counter] = post_prm
            self.update_counter()
            if self._counter == mcmc_obj._n_post_samples:
                self.reset_counter()
            SaveObject(self._list_post_weights, self._w_file)
        else:
            row = [mcmc_obj._current_iteration]
            tmp = bnn_obj._w_layers[0] * bnn_obj._indicators[0]
            row = row + [j for j in list(tmp.flatten())]
            for i in range(1, bnn_obj._n_layers):
                row = row + [j for j in list(bnn_obj._w_layers[i].flatten())]
            self._wweight.writerow(row)
            self._w_file.flush()

