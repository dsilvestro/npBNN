import numpy as np
np.set_printoptions(suppress=True, precision=3)
# load BNN package
import np_bnn as bn

# set random seed
rseed = 1234
np.random.seed(rseed)

# load data (2 files: features and labels)
f= "./example_files/data_features.txt"
l= "./example_files/data_labels.txt"
# with testsize=0.1, 10% of the data are randomly selected as test set
# if all_class_in_testset = 1: 10% of the samples and a minimum of 1 sample
# for each class are represented in the test set
dat = bn.get_data(f,l,
                  seed=rseed,
                  testsize=0.1, # 10% test set
                  all_class_in_testset=1,
                  header=1, # input data has a header
                  instance_id=1) # input data includes names of instances


# set up model architecture and priors
n_nodes_list = [5, 5] # 2 hidden layers with 5 nodes each
alphas = np.zeros(len(n_nodes_list))
activation_function = bn.genReLU(prm=alphas, trainable=True) # To use default ReLU: BNN_lib.genReLU()
prior = 1 # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
p_scale = 1 # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
use_class_weight = 0 # set to 1 to use class weights for unbalanced classes
init_std = 0.1 # st dev of the initial weights

# set up the BNN model
bnn_model = bn.npBNN(dat,
                     n_nodes = n_nodes_list,
                     use_class_weights=use_class_weight,
                     actFun=activation_function,
                     use_bias_node=1,
                     prior_f=prior,
                     p_scale=p_scale,
                     seed=rseed,
                     init_std=init_std)


# set up the MCMC environment
sample_from_prior = 0 # set to 1 to run an MCMC sampling the parameters from the prior only

mcmc = bn.MCMC(bnn_model,
               update_f=[0.05, 0.05, 0.07],
               update_ws=[0.075, 0.075, 0.075],
               temperature = 1,
               n_iteration=5000,
               sampling_f=10,
               print_f=1000,
               n_post_samples=100,
               sample_from_prior=sample_from_prior,
               likelihood_tempering=1)



# initialize output files
logger = bn.postLogger(bnn_model, filename="BNN", log_all_weights=0)

# run MCMC
bn.run_mcmc(bnn_model, mcmc, logger)

# make predictions based on MCMC's estimated weights
# test data
post_pr_test = bn.predictBNN(dat['test_data'],
                                  pickle_file=logger._w_file,
                                  test_labels=dat['test_labels'],
                                  instance_id=dat['id_test_data'],
                                  fname=dat['file_name'],
                                  post_summary_mode=0)

# determine feature importance with test data
feature_importance = bn.feature_importance(dat['test_data'],
                                           weights_pkl=logger._w_file,
                                           true_labels=dat['test_labels'],
                                           fname_stem=dat['file_name'],
                                           feature_names=dat['feature_names'],
                                           feature_blocks = [[0,1,2,3,4,5,6,7],[8,9,10],[11,12,13,14,15,16,17,18,19,20]],
                                           unlink_features_within_block = True)

# train+test data
dat_all = bn.get_data(f,l,
                      testsize=0, # no test set
                      header=1, # input data has a header
                      instance_id=1) # input data includes names of instances

post_pr_all = bn.predictBNN(dat_all['data'],
                            pickle_file=logger._w_file,
                            test_labels=dat_all['labels'],
                            instance_id=dat_all['id_data'],
                            fname="all_data")

# predict new unlabeled data
new_dat = bn.get_data(f="./example_files/unlabeled_data.txt",
                      header=1, # input data has a header
                      instance_id=1) # input data includes names of instances

post_pr_new = bn.predictBNN(new_dat['data'],
                            pickle_file=logger._w_file,
                            instance_id=new_dat['id_data'],
                            fname=new_dat['file_name'])



# ADDITIONAL OPTIONS
# to restart a previous run you can provide the pickle file with the posterior parameters
# when initializing the BNN environment
pickle_file = logger._w_file
bnn_model = bn.npBNN(dat,
                     n_nodes = n_nodes_list,
                     use_bias_node = 1,
                     prior_f = prior,
                     p_scale = p_scale,
                     pickle_file=pickle_file,
                     seed=rseed)

mcmc = bn.MCMC(bnn_model,
               update_f=[0.05, 0.04, 0.07],
               update_ws=[0.075, 0.075, 0.075],
               temperature = 1,
               n_iteration=50000,
               sampling_f=100,
               print_f=1000,
               n_post_samples=100)
