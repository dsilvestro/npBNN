import pickle
import numpy as np
import scipy.stats
import glob,os,sys,argparse
import pickle
np.set_printoptions(suppress= 1)
np.set_printoptions(precision=3)

# load BNN modules
from np_bnn import BNN_env
from np_bnn import BNN_files
from np_bnn import BNN_plot

# set random seed
rseed = 1234
np.random.seed(rseed)

# load data (2 files: features and labels)
f= "./example_files/features_subtypes_0_1_4_6_9.txt"
l= "./example_files/labels_subtypes_0_1_4_6_9.txt"
data, labels, test_data, test_labels = BNN_files.get_data(f,l,seed=rseed)


# set up model architecture and priors
n_nodes_list = [5, 5] # 2 hidden layers with 5 nodes each
prior = 1 # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
p_scale = 1 # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform


# set up the BNN model
bnn = BNN_env.npBNN(data, labels, test_data, test_labels, n_nodes = n_nodes_list,
                 use_bias_node = 1, prior_f = prior, p_scale = p_scale, seed=rseed)


# set up the MCMC environment
sample_from_prior = 0 # set to 1 to run an MCMC sampling the parameters from the prior only

mcmc = BNN_env.MCMC(bnn,update_f=[0.05, 0.04, 0.07, 0.02], update_ws=[0.075, 0.075, 0.075],
                 temperature = 1, n_iteration=100000, sampling_f=100, print_f=1000, n_post_samples=100,
                 sample_from_prior=sample_from_prior)


# initialize output files
logger = BNN_env.postLogger(bnn, filename="BNN")


# run MCMC
while True:
    mcmc.mh_step(bnn)
    # print some stats (iteration number, likelihood, training accuracy, test accuracy
    if mcmc._current_iteration % mcmc._print_f == 0 or mcmc._current_iteration == 1:
        print(mcmc._current_iteration, np.round([mcmc._logLik, mcmc._accuracy, mcmc._test_accuracy],2))
    # save to file
    if mcmc._current_iteration % mcmc._sampling_f == 0:
        logger.log_sample(bnn,mcmc)
        logger.log_weights(bnn,mcmc)
    # stop MCMC after running desired number of iterations
    if mcmc._current_iteration == mcmc._n_iterations:
        break

# make predictions based on MCMC's estimated weights
post_pkl = "./d13_l5_5/BNN_d13_p0_h0_l5_5_s5.0_b5.0_4321.pkl"
post_pr_in = BNN_lib.predictBNN(x_test_in_distribution, post_pkl,
                                labels_test_in_distribution, pickle_file_prior=prior_pkl)







# ADDITIONAL OPTIONS

# to restart a previous run you can provide the pickle file with the posterior parameters
# when initializing the BNN environment
pickle_file = "..."
bnn = BNN_env.npBNN(data, labels, test_data, test_labels, n_nodes = n_nodes_list,
                 use_bias_node = 1, prior_f = prior, p_scale = p_scale,
                 w_bound = w_bound, pickle_file=pickle_file, seed=rseed)
