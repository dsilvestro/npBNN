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

dat = bn.get_data(f,l,
                  seed=rseed,
                  testsize=0.1, # 10% test set
                  all_class_in_testset=1,
                  header=1, # input data has a header
                  instance_id=1) # input data includes names of instances

# make up new lab
dat['labels'] = np.max(dat['data'], axis=1) / (np.max(dat['data']) - np.min(dat['data']))
u = np.zeros((2, len(dat['labels'])))
u[0,:] = dat['labels'] + 0
u[1,:] = dat['labels'] + 1
dat['labels'] = u.T
dat['test_labels'] = np.max(dat['test_data'], axis=1) / (np.max(dat['test_data']) - np.min(dat['test_data']))
u = np.zeros((2, len(dat['test_labels'])))
u[0,:] = dat['test_labels'] + 0
u[1,:] = dat['test_labels'] + 1
dat['test_labels'] = u.T

# set up the BNN model
bnn_model = bn.npBNN(dat,
                     n_nodes = [5],
                     estimation_mode="regression"
)


# set up the MCMC environment
sample_from_prior = 0 # set to 1 to run an MCMC sampling the parameters from the prior only

mcmc = bn.MCMC(bnn_model,
               update_f=[0.05, 0.05, 0.07],
               update_ws=[0.075, 0.075, 0.075],
               temperature = 1,
               n_iteration=10000,
               sampling_f=10,
               print_f=1000,
               n_post_samples=100,
               sample_from_prior=sample_from_prior,
               likelihood_tempering=1)



# initialize output files
logger = bn.postLogger(bnn_model, filename="BNN", log_all_weights=0)

# run MCMC
bn.run_mcmc(bnn_model, mcmc, logger)


import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5, 5))
sns.regplot(x=dat['labels'][:,0].flatten(),y=mcmc._y[:,0])
sns.regplot(x=dat['labels'][:,1].flatten(),y=mcmc._y[:,1])
fig.show()



