import pickle
import sys
import numpy as np
import scipy.stats
from np_bnn.BNN_lib import *
import np_bnn.BNN_files
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit

class npBNN():
    def __init__(self, data, labels, test_data, test_labels, n_nodes = [50, 5],
                 n_layers=3, use_bias_node=1, init_std=0.1, p_scale=1,
                 prior_f=1, hyper_p=0, freq_indicator=0, w_bound = np.infty,
                 pickle_file=""):
        # prior_f: 0) uniform 1) normal 2) cauchy
        # to change the boundaries of a uniform prior use -p_scale
        # hyper_p: 0) no hyperpriors, 1) 1 per layer, 2) 1 per input node, 3) 1 per node
        # freq_indicator -> update freq indicators: 0) no indicators, 0-1
        self._data = data
        self._labels = labels.astype(int)
        self._test_data = test_data
        if len(test_labels) > 0:
            self._test_labels = test_labels.astype(int)
        else:
            self._test_labels = []
        self._size_output = len(np.unique(self._labels))
        self._init_std = init_std
        self._n_layers = n_layers
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
        self._prior_ind1 = 0.5

        # reset labels
        self._labels_reset = np.zeros(len(self._labels)).astype(int)
        self._test_labels_reset = np.zeros(len(self._test_labels)).astype(int)
        j = 0
        for i in np.unique(self._labels):
            self._labels_reset[self._labels==i] = j
            self._test_labels_reset[self._test_labels==i] = j
            j += 1

        # init weights
        if pickle_file == "":
            # 1st layer
            w_layers = [np.random.normal(0, self._init_std, (self._n_nodes[0], self._n_features))]
            # add hidden layers
            for i in range(1, n_layers - 1):
                w_layers.append(np.random.normal(0, self._init_std, (self._n_nodes[i], self._n_nodes[i - 1])))
            # last layer
            w_layers.append(np.random.normal(0, self._init_std, (self._size_output, self._n_nodes[-1])))
        else:
            post_w = np_bnn.BNN_files.load_obj(pickle_file)
            w_layers = post_w[-1]
        self._w_layers = w_layers

        self._indicators = np.ones(self._w_layers[0].shape)

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
        # init prior scales: will be updated if hyper-priors
        self._prior_scale = np.ones(self._n_layers) * self._p_scale

    # init prior functions
    def calc_prior(self, w=0):
        if w == 0:
            w = self._w_layers
        if self._prior == 0:
            logPrior = 0
        else:
            logPrior = 0
            for i in range(self._n_layers):
                logPrior += np.sum(self._prior_f(w[i], 0, scale=self._prior_scale[i]))
        if self._prior_ind1 != 0.5:
            logPrior += np.sum(self._indicators) * np.log(self._prior_ind1) + \
            (self._indicators.size-np.sum(self._indicators)) * np.log(1 - self._prior_ind1)
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


class MCMC():
    def __init__(self, bnn_obj,update_f=[0.05, 0.05, 0.8, 0.01], update_ws=[0.05, 0.075, 0.05],
                 temperature = 1, n_iteration=100000, sampling_f=100, print_f=1000, n_post_samples=1000,
                 data_set=0, wdir="",update_function=UpdateNormal, sample_from_prior=0):
        self._update_f = update_f
        self._update_ws = [np.ones(bnn_obj._w_layers[i].shape)*update_ws[i] for i in range(bnn_obj._n_layers)]
        self._update_n = [np.round(bnn_obj._w_layers[i].size*update_f[i]).astype(int) for i in range(bnn_obj._n_layers)]
        self._temperature = temperature
        self._n_iterations = n_iteration
        self._sampling_f = sampling_f
        self._print_f = print_f
        self._current_iteration = 0
        self._y = RunPredict(bnn_obj._data, bnn_obj._w_layers)
        if sample_from_prior:
            self._logLik = 0
        else:
            self._logLik = calc_likelihood(self._y, bnn_obj._labels_reset, bnn_obj._sample_id)
        self._logPrior = bnn_obj.calc_prior()
        self._logPost = self._logLik + self._logPrior
        self._accuracy = CalcAccuracy(self._y, bnn_obj._labels)
        if len(bnn_obj._test_data) > 0:
            self._y_test = RunPredictInd(bnn_obj._test_data, bnn_obj._w_layers, bnn_obj._indicators)
            self._test_accuracy = CalcAccuracy(self._y_test, bnn_obj._test_labels)
        else:
            self._y_test = []
            self._test_accuracy = 0
        self._label_freq = CalcLabelFreq(self._y)
        self._list_post_weights = list()
        self._n_post_samples = n_post_samples
        wlog, logfile, w_file = np_bnn.BNN_files.init_output_files(bnn_obj, data_set, sample_from_prior, outpath=wdir)
        self._wlog = wlog
        self._logfile = logfile
        self._w_file = w_file
        self.update_function = update_function
        self._sample_from_prior = sample_from_prior


    def mh_step(self, bnn_obj):
        w_layers_prime = []
        tmp = bnn_obj._data + 0
        indicators_prime = bnn_obj._indicators + 0
        for i in range(bnn_obj._n_layers):
            if np.random.random() > bnn_obj._freq_indicator or i > 0:
                update, indx = self.update_function(bnn_obj._w_layers[i], d=self._update_ws[i], n=self._update_n[i],
                                            Mb=bnn_obj._w_bound, mb= -bnn_obj._w_bound)
                w_layers_prime.append(update)
            else:
                w_layers_prime.append(bnn_obj._w_layers[i] + 0)
                indicators_prime = UpdateBinomial(bnn_obj._indicators, self._update_f[3], bnn_obj._indicators.shape)
            if i == 0:
                w_layers_prime_temp = w_layers_prime[i] * indicators_prime
            else:
                w_layers_prime_temp = w_layers_prime[i]
            tmp = RunHiddenLayer(tmp, w_layers_prime_temp)
        y_prime = SoftMax(tmp)

        logPrior_prime = bnn_obj.calc_prior(w=w_layers_prime)
        if self._sample_from_prior:
            logLik_prime = 0
        else:
            logLik_prime = calc_likelihood(y_prime, bnn_obj._labels_reset, bnn_obj._sample_id)
        logPost_prime = logLik_prime + logPrior_prime

        if (logPost_prime - self._logPost) / self._temperature >= np.log(np.random.random()):
            #print(logPost_prime, self._logPost)
            bnn_obj._w_layers = w_layers_prime
            bnn_obj._indicators = indicators_prime
            self._logPost = logPost_prime
            self._logLik = logLik_prime
            self._logPrior = logPrior_prime
            self._y = y_prime
            self._accuracy = CalcAccuracy(self._y, bnn_obj._labels_reset)
            self._label_freq = CalcLabelFreq(self._y)
            if len(bnn_obj._test_data) > 0:
                self._y_test = RunPredictInd(bnn_obj._test_data, bnn_obj._w_layers, bnn_obj._indicators)
                self._test_accuracy = CalcAccuracy(self._y_test, bnn_obj._test_labels_reset)
            else:
                self._y_test = []
                self._test_accuracy = 0

        self._current_iteration += 1

    def gibbs_step(self, bnn_obj):
        bnn_obj.sample_prior_scale()
        self._logPrior = bnn_obj.calc_prior()
        self._logPost = self._logLik + self._logPrior
        self._current_iteration += 1

    def log_sample(self, bnn_obj):
        row = [self._current_iteration, self._logPost, self._logLik, self._logPrior,
               self._accuracy, self._test_accuracy] + list(self._label_freq)
        for i in range(bnn_obj._n_layers):
            row = row + [np.mean(bnn_obj._w_layers[i]), np.std(bnn_obj._w_layers[i])]
            if bnn_obj._hyper_p:
                if bnn_obj._hyper_p == 1:
                    row.append(bnn_obj._prior_scale[i])
                else:
                    row.append(np.mean(bnn_obj._prior_scale[i]))
        if bnn_obj._freq_indicator > 0:
            row.append(np.mean(bnn_obj._indicators))
        self._wlog.writerow(row)
        self._logfile.flush()

    def log_weights(self, bnn_obj):
        if len(self._list_post_weights) < self._n_post_samples:
            self._counter = 0
            if bnn_obj._freq_indicator:
                tmp = list()
                tmp.append(bnn_obj._w_layers[0] *  bnn_obj._indicators)
                for i in range(1, bnn_obj._n_layers):
                    tmp.append(bnn_obj._w_layers[i])
                self._list_post_weights.append(tmp)
            else:
                self._list_post_weights.append(bnn_obj._w_layers)
        else:
            if bnn_obj._freq_indicator:
                tmp = list()
                tmp.append(bnn_obj._w_layers[0] *  bnn_obj._indicators)
                for i in range(1, bnn_obj._n_layers):
                    tmp.append(bnn_obj._w_layers[i])
                self._list_post_weights[self._counter] = tmp
            else:
                self._list_post_weights[self._counter] = bnn_obj._w_layers
            self._counter += 1
            if self._counter == len(self._list_post_weights):
                self._counter = 0

        if len(self._list_post_weights) == self._n_post_samples:
            SaveObject(self._list_post_weights, self._w_file)
            self._list_post_weights = list()
