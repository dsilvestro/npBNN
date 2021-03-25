import numpy as np
np.set_printoptions(suppress=True, precision=3)
# load BNN package
import np_bnn as bn

# set random seed
rseed = 1234
np.random.seed(rseed)

# load data (2 files: features and labels)
f= "./ants/mp_features.txt"
l= "./ants/mp_labels.txt"
# with testsize=0.1, 10% of the data are randomly selected as test set
# if all_class_in_testset = 1: 10% of the samples and a minimum of 1 sample
# for each class are represented in the test set
cross_validation_batch = 0 # cross validation (0 = 1st batch; set to 1,2,... to use subsequent batches as test set)
dat = bn.get_data(f,l,
                  seed=rseed,
                  testsize=0.10,
                  all_class_in_testset=1,
                  header=1, # input data has a header
                  cv=cross_validation_batch,
                  instance_id=1) # input data includes names of instances


# set up model architecture and priors
n_nodes_list = [10, 5] # 2 hidden layers with 10 and 5 nodes, respectively
# default ReLU:
# activation_function = bn.ActFun(fun="ReLU")
# alternatively for swish function:
activation_function = bn.ActFun(fun="swish")
# finally for generalized ReLU:
# activation_function = bn.ActFun(fun="genReLU", trainable=True, prm=np.zeros(len(n_nodes_list)))
prior = 1 # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
p_scale = 1 # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
init_std = 0.1 # st dev of the initial weights

# set up the BNN model
bnn_model = bn.npBNN(dat,
                     n_nodes = n_nodes_list,
                     actFun=activation_function,
                     use_bias_node=1,
                     prior_f=prior,
                     p_scale=p_scale,
                     seed=rseed,
                     init_std=init_std)


# set up the MCMC environment
sample_from_prior = 0 # set to 1 to run an MCMC sampling the parameters from the prior only

mcmc = bn.MCMC(bnn_model,
               update_f=[0.05, 0.1, 0.2],
               update_ws=[0.07, 0.1, 0.1],
               n_iteration=250000,
               sampling_f=100,
               print_f=1000,
               n_post_samples=250)



# initialize output files
out_file_name = "./ants/BNN_cv%s" % cross_validation_batch
logger = bn.postLogger(bnn_model, filename=out_file_name, log_all_weights=0)

# run MCMC
bn.run_mcmc(bnn_model, mcmc, logger)

# test data
dat_all = bn.get_data(f,l,
                      testsize=0, # no test set
                      header=1, # input data has a header
                      instance_id=1) # input data includes names of instances

# find the posterior probability threshold that yields a target accuracy (here set to 0.90)
bn.get_posterior_threshold(pkl_file=logger._pklfile,
                           target_acc=0.9,
                           output_file="./ants/accuracy_thresholds.txt")


# predict new unlabeled data
new_dat = bn.get_data(f="your_unlabeled_feature_data",
                      header=1, # input data has a header
                      instance_id=1) # input data includes names of instances

post_pr_new = bn.predictBNN(new_dat['data'],
                            pickle_file=logger._pklfile,
                            instance_id=new_dat['id_data'],
                            fname=new_dat['file_name'],
                            post_summary_mode=1) # use this option to apply the threshold estimated above

