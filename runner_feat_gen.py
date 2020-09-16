import numpy as np

np.set_printoptions(suppress= 1)
np.set_printoptions(precision=3)

# load BNN modules
from np_bnn import BNN_env, BNN_files, BNN_lib

# set random seed
rseed = 1234
np.random.seed(rseed)

# load data (2 files: features and labels)
f= "./example_files/data_features.txt"
l= "./example_files/data_labels.txt"
# with testsize=0.1, 10% of the data are randomly selected as test set
# if all_class_in_testset = 1: 10% of the samples and a minimum of 1 sample
# for each class are represented in the test set
dat = BNN_files.get_data(f,l,seed=rseed,testsize=0.1, all_class_in_testset=1) # 10% test set


# set up model architecture and priors
n_nodes_list = [5, 5] # 2 hidden layers with 5 nodes each
prior = 1 # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
p_scale = 1 # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
use_class_weight = 0 # set to 1 to use class weights for unbalanced classes

# set up the BNN model
bnn = BNN_env.npBNN(dat, n_nodes = n_nodes_list, use_class_weights=use_class_weight,
                 use_bias_node = 1, prior_f = prior, p_scale = p_scale, seed=rseed)


# set up the MCMC environment
sample_from_prior = 0 # set to 1 to run an MCMC sampling the parameters from the prior only

mcmc = BNN_env.MCMC(bnn,update_f=[0.05, 0.04, 0.07, 0.02], update_ws=[0.075, 0.075, 0.075],
                 temperature = 1, n_iteration=50000, sampling_f=100, print_f=1000, n_post_samples=100,
                 sample_from_prior=sample_from_prior)


# initialize output files
logger = BNN_env.postLogger(bnn, filename="BNN")

# update data
"""
while True:
    1. creat temporary weights (not in feat_gen class)
    2. calca new prior (temporary weights)
    3. update features (temporary weights)
    4. updated data in bnn
    5. mcmc.mh_step(updated_bnn, additional_prior)

"""



"temp_dat = feature_gen ..."
bnn.update_data(temp_dat)




# run MCMC
while True:
"""

feature_gen_prime = deepcopy(feature_gen)
bnn_prime = deepcopy(bnn)

    1. create temporary weights
            feature_gen_prime._w1 = BNN_lib.UpdateNormal(feature_gen_prime._w1, d=0.01, n=1, Mb=5, mb= -5) 
            feature_gen_prime._w2 = BNN_lib.UpdateNormal(feature_gen_prime._w1, d=0.01, n=1, Mb=5, mb= -5) 
    2. calc new prior (temporary weights)
            pdf_normal(temp_w1, temp_w2)
    3. update features (temporary weights)
            feature_gen.update_weights(temp_w1,temp_w2)
            feature_gen.update_features()
    4. updated data in bnn
            bnn_prime.update_data(feature_gen._features_dict)
    5.  do MCMC step
            mcmc.mh_step(bnn_prime, additional_prior)
            if mcmc._accepted:
                pass
            else:
                bnn = bnn_previous
                features_gen = feature_gen_previous


"""    
    mcmc.mh_step(bnn)
    # print some stats (iteration number, likelihood, training accuracy, test accuracy
    if mcmc._current_iteration % mcmc._print_f == 0 or mcmc._current_iteration == 1:
        print(mcmc._current_iteration, np.round([mcmc._logLik, mcmc._accuracy, mcmc._test_accuracy],3))
    '''
    if accept:
        update weights in feat_gen
        features = temp_features
    if reject:
        bnn = bnn_previous
        features_gen = feature_gen_previous
    '''
    # save to file
    if mcmc._current_iteration % mcmc._sampling_f == 0:
        "save additional weights"
        logger.log_sample(bnn,mcmc)
        logger.log_weights(bnn,mcmc)
    # stop MCMC after running desired number of iterations
    if mcmc._current_iteration == mcmc._n_iterations:
        break



# make predictions based on MCMC's estimated weights (test data)
post_pr = BNN_lib.predictBNN(dat['test_data'], pickle_file=logger._w_file, test_labels=dat['test_labels'])

# make predictions based on MCMC's estimated weights (train data)
post_pr = BNN_lib.predictBNN(dat['data'], pickle_file=logger._w_file, test_labels=dat['labels'])





# ADDITIONAL OPTIONS

# to restart a previous run you can provide the pickle file with the posterior parameters
# when initializing the BNN environment
pickle_file = "./BNN_p1_h0_l5_5_s1_binf_1234.pkl"
bnn = BNN_env.npBNN(dat, n_nodes = n_nodes_list,
                 use_bias_node = 1, prior_f = prior, p_scale = p_scale,
                 pickle_file=pickle_file, seed=rseed)

mcmc = BNN_env.MCMC(bnn,update_f=[0.05, 0.04, 0.07, 0.02], update_ws=[0.075, 0.075, 0.075],
                 temperature = 1, n_iteration=50000, sampling_f=100, print_f=1000, n_post_samples=100)
