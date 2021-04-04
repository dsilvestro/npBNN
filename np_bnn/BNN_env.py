import scipy.stats
from numpy.random import MT19937

from .BNN_lib import *
from .BNN_mcmc import *
from .BNN_files import *


class npBNN():
    def __init__(self, dat, n_nodes=[50, 5],
                 use_bias_node=1, init_std=0.1, p_scale=1, prior_ind1=0.5,
                 prior_f=1, hyper_p=0, freq_indicator=0, w_bound=np.infty,
                 pickle_file="", seed=1234, use_class_weights=0, actFun=ActFun(),init_weights=None,
                 estimation_mode="classification"):
        # prior_f: 0) uniform 1) normal 2) cauchy
        # to change the boundaries of a uniform prior use -p_scale
        # hyper_p: 0) no hyperpriors, 1) 1 per layer, 2) 1 per input node, 3) 1 per node
        # freq_indicator -> update freq indicators: 0) no indicators, 0-1
        data, labels, test_data, test_labels = dat['data'], dat['labels'], dat['test_data'], dat['test_labels']
        self._seed = seed
        np.random.seed(self._seed)
        self._data = data
        if estimation_mode == "classification":
            self._labels = labels.astype(int)
        else:
            self._labels = labels
        self._test_data = test_data
        if len(test_labels) > 0:
            if estimation_mode == "classification":
                self._test_labels = test_labels.astype(int)
            else:
                self._test_labels = test_labels
        else:
            self._test_labels = []
        if estimation_mode == "classification":
            self._size_output = len(np.unique(self._labels))
            self._n_output_prm = self._size_output
            self._output_act_fun = SoftMax
        elif estimation_mode == "regression":
            self._size_output = self._labels.shape[1] * 2 # mus, sigs
            self._n_output_prm = self._labels.shape[1]
            self._output_act_fun = RegressTransform
        self._init_std = init_std
        try: # see if we have an actual list or single element
            n_nodes = list(n_nodes)
        except TypeError:
            n_nodes = [n_nodes]
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
        # self._sample_id = np.arange(self._n_samples)
        self._w_bound = w_bound
        self._freq_indicator = freq_indicator
        self._hyper_p = hyper_p
        self._sample_id = np.arange(self._n_samples)
        self._prior = prior_f
        self._p_scale = p_scale
        self._prior_ind1 = prior_ind1
        self._estimation_mode = estimation_mode

        if use_class_weights:
            class_counts = np.unique(self._labels, return_counts=True)[1]
            self._class_w = 1 / (class_counts / np.max(class_counts))
            self._class_w = self._class_w / np.mean(self._class_w)
            print("Using class weights:", self._class_w)
        else:
            self._class_w = []

        # init weights
        if init_weights is None:
            if pickle_file == "":
                w_layers = init_weight_prm(self._n_nodes,
                                           self._n_features,
                                           self._size_output,
                                           init_std=0.1,
                                           bias_node=use_bias_node-1)
            else:
                bnn_obj,mcmc_obj,logger_obj = load_obj(pickle_file)
                post_samples = logger_obj._post_weight_samples
                post_weights = [post_samples[i]['weights'] for i in range(len(post_samples))]
                w_layers = post_weights[-1]
        else:
            w_layers = init_weights
            #self._n_layers -= 1
        self._w_layers = w_layers

        self._indicators = np.ones(self._w_layers[0].shape)

        self._act_fun = actFun
        if pickle_file != "" and actFun._trainable:
            post_alphas = [post_samples[i]['alphas'] for i in range(len(post_samples))]
            self._act_fun.reset_prm(post_alphas[-1])


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
                 likelihood_tempering=1, mcmc_id=0, randomize_seed=False, adapt_f=0):
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
        self._update_n = [np.max([1, np.round(bnn_obj._w_layers[i].size * update_f[i]).astype(int)]) for i in
                          range(bnn_obj._n_layers)]
        self._temperature = temperature
        self._n_iterations = n_iteration
        self._sampling_f = sampling_f
        self._print_f = print_f
        self._current_iteration = 0
        self._y = RunPredict(bnn_obj._data, bnn_obj._w_layers, bnn_obj._act_fun, bnn_obj._output_act_fun)

        if bnn_obj._estimation_mode == "classification":
            self._likelihood_f = calc_likelihood
        else:
            self._likelihood_f = calc_likelihood_regression
        if sample_from_prior:
            self._logLik = 0
        else:
            self._logLik = self._likelihood_f(self._y,
                                              bnn_obj._labels,
                                              bnn_obj._sample_id,
                                              bnn_obj._class_w,
                                              likelihood_tempering)
        self._logPrior = bnn_obj.calc_prior() + init_additional_prob
        self._logPost = self._logLik + self._logPrior
        if bnn_obj._estimation_mode == "classification":
            self._accuracy_f = CalcAccuracy
            self._accuracy_lab_f = CalcLabelAccuracy
        else:
            self._accuracy_f = CalcAccuracyRegression
            self._accuracy_lab_f = CalcLabelAccuracyRegression

        self._accuracy = self._accuracy_f(self._y, bnn_obj._labels)
        self._label_acc = self._accuracy_lab_f(self._y, bnn_obj._labels)
        if len(bnn_obj._test_data) > 0:
            self._y_test = RunPredictInd(bnn_obj._test_data, bnn_obj._w_layers, bnn_obj._indicators,
                                         bnn_obj._act_fun, bnn_obj._output_act_fun)
            self._test_accuracy = self._accuracy_f(self._y_test, bnn_obj._test_labels)
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
        self._freq_layer_update = np.ones(bnn_obj._n_layers)
        self._adapt_f = adapt_f

    def mh_step(self, bnn_obj, additional_prob=0, return_bnn=False):
        if self._randomize_seed:
            self._rs = RandomState(MT19937(SeedSequence(self._current_iteration + self._mcmc_id)))

        hastings = 0
        w_layers_prime = []
        tmp = bnn_obj._data + 0
        indicators_prime = bnn_obj._indicators + 0
        
        if self._current_iteration % 100 == 0:
            acc_rate = (1 + self._accepted_states) / (1 + self._current_iteration)
            if acc_rate < self._adapt_f:
                self._freq_layer_update = self._freq_layer_update * 0.8
                print(self._freq_layer_update)
        
        # if trainable prm in activation function
        if bnn_obj._act_fun._trainable:
            prm_tmp, _, h = UpdateNormal1D(bnn_obj._act_fun._acc_prm, d=0.05, n=1, Mb=1, mb=0, rs=self._rs)
            r = 10
            additional_prob += np.log(r) * -np.sum(prm_tmp)*r # aka exponential Exp(r)
            hastings += h
            bnn_obj._act_fun.reset_prm(prm_tmp)
        
        rr = self._rs.random(bnn_obj._n_layers)
        # rr[self._rs.randint(0, bnn_obj._n_layers)] = 1 # make sure to update one layer
        for i in range(bnn_obj._n_layers):
            if rr[i] > bnn_obj._freq_indicator or i > 0:
                if rr[i] < self._freq_layer_update[i]:
                    update, indx, h = self.update_function(bnn_obj._w_layers[i], d=self._update_ws[i], n=self._update_n[i],
                                                           Mb=bnn_obj._w_bound, mb=-bnn_obj._w_bound, rs=self._rs)
                    w_layers_prime.append(update)
                    hastings += h
                else:
                    w_layers_prime.append(bnn_obj._w_layers[i] + 0)
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
        y_prime = bnn_obj._output_act_fun(tmp)

        logPrior_prime = bnn_obj.calc_prior(w=w_layers_prime, ind=indicators_prime) + additional_prob
        if self._sample_from_prior:
            logLik_prime = 0
        else:
            logLik_prime = self._likelihood_f(y_prime,
                                              bnn_obj._labels,
                                              bnn_obj._sample_id,
                                              bnn_obj._class_w,
                                              self._lik_temp)
        logPost_prime = logLik_prime + logPrior_prime
        rrr = np.log(self._rs.random())
        if (logPost_prime - self._logPost) * self._temperature + hastings >= rrr:
            # print(logPost_prime, self._logPost)
            bnn_obj.reset_weights(w_layers_prime)
            bnn_obj.reset_indicators(indicators_prime)
            if bnn_obj._act_fun._trainable:
                bnn_obj._act_fun.reset_accepted_prm()
            self._logPost = logPost_prime
            self._logLik = logLik_prime
            self._logPrior = logPrior_prime
            self._y = y_prime
            self._accuracy = self._accuracy_f(self._y, bnn_obj._labels)
            self._label_acc = self._accuracy_lab_f(self._y, bnn_obj._labels)
            self._label_freq = CalcLabelFreq(self._y)
            if len(bnn_obj._test_data) > 0:
                self._y_test = RunPredictInd(bnn_obj._test_data, bnn_obj._w_layers,
                                             bnn_obj._indicators, bnn_obj._act_fun,
                                             bnn_obj._output_act_fun)
                self._test_accuracy = self._accuracy_f(self._y_test, bnn_obj._test_labels)
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
        
        logfile, w_file, pklfile = init_output_files(bnn_obj, filename, sample_from_prior,
                                                                            outpath=wdir, add_prms=add_prms,
                                                                            continue_logfile=continue_logfile,
                                                                            log_all_weights=log_all_weights)

        self._logfile = logfile
        self._w_file = w_file
        self._pklfile = pklfile
        self._log_all_weights = log_all_weights
        self._post_weight_samples = []
        self._estimation_mode = bnn_obj._estimation_mode

    def update_post_weight_samples(self,row):
        self._post_weight_samples += [row]

    def replace_post_weight_samples(self,post_weight_samples):
        self._post_weight_samples = post_weight_samples

    def control_weight_sample_length(self,maxlength):
        if len(self._post_weight_samples)>maxlength:
            self._post_weight_samples = self._post_weight_samples[-maxlength:]

    def log_sample(self, bnn_obj, mcmc_obj, add_prms=None):
        row = [mcmc_obj._current_iteration, mcmc_obj._logPost, mcmc_obj._logLik, mcmc_obj._logPrior,
               mcmc_obj._accuracy, mcmc_obj._test_accuracy] + list(mcmc_obj._label_acc)
        #list(mcmc_obj._label_freq)
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
        logfile_IO = open(self._logfile, "a")
        wlog = csv.writer(logfile_IO, delimiter='\t')
        wlog.writerow(row)
        logfile_IO.flush()

    def log_weights(self, bnn_obj, mcmc_obj, add_prms=None):
        # print(mcmc_obj._current_iteration, self._counter, len(self._list_post_weights))
        if self._log_all_weights:
            row = [mcmc_obj._current_iteration]
            tmp = bnn_obj._w_layers[0] * bnn_obj._indicators[0]
            row = row + [j for j in list(tmp.flatten())]
            for i in range(1, bnn_obj._n_layers):
                row = row + [j for j in list(bnn_obj._w_layers[i].flatten())]
            w_file_IO = open(self._w_file, "a")
            wweights = csv.writer(w_file_IO, delimiter='\t')
            wweights.writerow(row)
            w_file_IO.flush()
        else:
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
            post_prm['mcmc_it'] = mcmc_obj._current_iteration
            if add_prms:
                post_prm['additional_prm'] = list(add_prms)

            self.update_post_weight_samples(post_prm)
            self.control_weight_sample_length(mcmc_obj._n_post_samples)

        SaveObject([bnn_obj,mcmc_obj,self],self._pklfile)
