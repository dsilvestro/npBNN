import os
from copy import deepcopy
import numpy as np
import multiprocessing
from .BNN_env import npBNN, MCMC, postLogger


class MC3():
    def __init__(self,
                 data: npBNN,
                 logger: postLogger,
                 n_post_samples=100,
                 sampling_f=100,
                 n_chains=4,
                 swap_frequency=100,
                 verbose=1,
                 print_f=100,
                 temperatures=None,
                 min_temperature=0.8,
                 likelihood_f=None,
                 accuracy_f=None,
                 adapt_freq=50,
                 adapt_f=0.1,
                 adapt_fM=0.6,
                 adapt_stop=1000,
                 n_iteration=100000

                 ):
        self.n_chains = n_chains
        self.swap_frequency = swap_frequency
        self.verbose = verbose
        self.print_f = print_f / swap_frequency
        self.n_post_samples = n_post_samples
        self.sampling_f = sampling_f
        self.adapt_freq = adapt_freq
        self.adapt_f = adapt_f
        self.adapt_fM = adapt_fM
        self.adapt_stop = adapt_stop
        self.likelihood_f = likelihood_f
        self.n_mc3_iteration = np.round(n_iteration / swap_frequency).astype(int)
        self.accuracy_f = accuracy_f

        # init chains seeds
        self.rseeds = np.random.choice(range(1000, 9999), n_chains, replace=False)

        if temperatures is None:
            if n_chains == 1:
                temperatures = [1]
            else:
                temperatures = np.linspace(min_temperature, 1, n_chains)
        self.temperatures = temperatures

        # replicate data
        bnnList = []
        for i in range(n_chains):
            data_tmp = deepcopy(data)
            data_tmp.reset_seed(self.rseeds[i])
            bnnList.append(data_tmp)

        # setup MCMCs
        mcmcList = [MCMC(bnnList[i],
                         temperature=self.temperatures[i],
                         n_iteration=self.swap_frequency,
                         sampling_f=self.sampling_f,
                         print_f=self.swap_frequency * 10,
                         n_post_samples=self.n_post_samples,
                         mcmc_id=i,
                         randomize_seed=True,
                         adapt_freq=self.adapt_freq,
                         adapt_f=self.adapt_f,
                         adapt_fM=self.adapt_fM,
                         adapt_stop=self.adapt_stop,
                         likelihood_f=self.likelihood_f,
                         accuracy_f=self.accuracy_f)
                    for i in range(n_chains)]

        self.logger = logger
        self.singleChainArgs = [[bnnList[i], mcmcList[i]] for i in range(self.n_chains)]

    def run_single_mcmc(self, arg_list):
        [bnn_obj, mcmc_obj] = arg_list
        for i in range(self.swap_frequency - 1):
            mcmc_obj.mh_step(bnn_obj)
        bnn_obj_new, mcmc_obj_new = mcmc_obj.mh_step(bnn_obj, return_bnn=True)
        return [bnn_obj_new, mcmc_obj_new]

    def run_mcmc(self):
        # Choose the appropriate multiprocessing method based on the OS
        if os.name == 'posix':  # For Unix-based systems (macOS, Linux)
            ctx = multiprocessing.get_context('fork')
        else:  # For Windows
            ctx = multiprocessing.get_context('spawn')

        with ctx.Pool(self.n_chains) as pool:
            for mc3_it in range(self.n_mc3_iteration):
                self.singleChainArgs = list(pool.map(self.run_single_mcmc, self.singleChainArgs))
                # singleChainArgs = [i for i in tmp]
                if self.n_chains > 1:
                    n1 = np.random.choice(range(self.n_chains), 2, replace=False)
                    [j, k] = n1
                    temp_j = self.singleChainArgs[j][1]._temperature + 0
                    temp_k = self.singleChainArgs[k][1]._temperature + 0
                    r = (self.singleChainArgs[k][1]._logPost - self.singleChainArgs[j][1]._logPost) * temp_j + \
                        (self.singleChainArgs[j][1]._logPost - self.singleChainArgs[k][1]._logPost) * temp_k

                    # print(mc3_it, r, singleChainArgs[j][1]._logPost, singleChainArgs[k][1]._logPost, temp_j, temp_k)
                    # if mc3_it % self.print_f == 0:
                    #     print(mc3_it, self.singleChainArgs[0][1]._logPost, self.singleChainArgs[1][1]._logPost,
                    #           self.singleChainArgs[0][0]._w_layers[0][0][0:5])
                    if r >= np.log(np.random.random()):
                        self.singleChainArgs[j][1].reset_temperature(temp_k)
                        self.singleChainArgs[k][1].reset_temperature(temp_j)
                        if self.verbose > 0:
                            print(mc3_it, "SWAPPED", self.singleChainArgs[j][1]._logPost,
                                  self.singleChainArgs[k][1]._logPost, temp_j,
                                  temp_k)

                for i in range(self.n_chains):
                    if self.singleChainArgs[i][1]._temperature == 1:
                        # print( singleChainArgs[i][0]._w_layers[0][0][0:10] )
                        self.logger.log_sample(self.singleChainArgs[i][0], self.singleChainArgs[i][1])
                        self.logger.log_weights(self.singleChainArgs[i][0], self.singleChainArgs[i][1])

                else:
                    if mc3_it % self.print_f == 0:
                        print(mc3_it, self.singleChainArgs[0][1]._logPost, self.singleChainArgs[0][0]._w_layers[0][0][0:5])
