import os
import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor

np.set_printoptions(suppress=True, precision=3)

import np_bnn as bn

# set random seed
rseed = 1234
np.random.seed(rseed)

# load data (2 files: features and labels)
f = "./example_files/data_features.txt"
l = "./example_files/data_labels.txt"
dat = bn.get_data(f, l,
                  seed=rseed,
                  testsize=0.1,
                  all_class_in_testset=1,
                  header=1,  # input data has a header
                  instance_id=1)  # input data includes names of instances

# set up model architecture and priors
n_nodes_list = [5, 5]  # 2 hidden layers with 5 nodes each

# set up the BNN model
data_obj = bn.npBNN(dat,
                    n_nodes=n_nodes_list,
                    use_bias_node=-1,
                    seed=1,
                    init_std=0.1)


# initialize output files
logger = bn.postLogger(data_obj, filename="BNNMC3", log_all_weights=0)

mc3 = bn.MC3(data_obj,logger=logger,
             n_post_samples=100,
             sampling_f=100,
             n_iteration=20000,
             n_chains=4,
             swap_frequency=100,
             verbose=1,  # print successful swaps
             )

# run MCMCMC
mc3.run_mcmc()


# make predictions based on MCMC's estimated weights
post_pr_test = bn.predictBNN(dat['test_data'],
                             pickle_file=logger._pklfile,
                             test_labels=dat['test_labels'],
                             instance_id=dat['id_test_data'])

# train+test data
dat = bn.get_data(f, l,
                  testsize=0,
                  header=1,  # input data has a header
                  instance_id=1)  # input data includes names of instances

post_pr_all = bn.predictBNN(dat['data'],
                            pickle_file=logger._pklfile,
                            test_labels=dat['labels'],
                            instance_id=dat['id_data'])

# predict new unlabeled data
dat = bn.get_data(f="./example_files/unlabeled_data.txt",
                  testsize=0,
                  header=1,  # input data has a header
                  instance_id=1)  # input data includes names of instances

post_pr_new = bn.predictBNN(dat['data'],
                            pickle_file=logger._pklfile,
                            instance_id=dat['id_data'])



