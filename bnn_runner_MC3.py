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
f= "./example_files/data_features.txt"
l= "./example_files/data_labels.txt"
dat = bn.get_data(f,l,
                  seed=rseed,
                  testsize=0.1,
                  all_class_in_testset=1,
                  header=1, # input data has a header
                  instance_id=1) # input data includes names of instances

# set up model architecture and priors
n_nodes_list = [5, 5] # 2 hidden layers with 5 nodes each
prior = 1 # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
p_scale = 1 # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
use_class_weight = 0 # set to 1 to use class weights for unbalanced classes

# set up the BNN model

n_chains = 4
rseeds = np.random.choice(range(1000,9999), n_chains, replace=False)


bnnList = [bn.npBNN(dat, n_nodes = n_nodes_list, use_class_weights=use_class_weight,
                    use_bias_node = 1, prior_f = prior, p_scale = p_scale,
                    seed=rseeds[i], init_std=0.1)
           for i in range(n_chains)]


if n_chains == 1:
    temperatures = [1]
else:
    temperatures = np.linspace(0.8, 1, n_chains)
mcmcList = [bn.MCMC(bnnList[i],
                    temperature=temperatures[i],
                    n_iteration=100, sampling_f=100, print_f=1000, n_post_samples=100,
                    mcmc_id=i, randomize_seed=True,
                    adapt_freq=50, adapt_f=0.1, adapt_fM=0.6, adapt_stop=1000)
            for i in range(n_chains)]


singleChainArgs = [[bnnList[i],mcmcList[i]] for i in range(n_chains)]
n_iterations = 100
# initialize output files
logger = bn.postLogger(bnnList[0], filename="BNNMC3", log_all_weights=0)

def run_single_mcmc(arg_list):
    [bnn_obj, mcmc_obj] = arg_list
    for i in range(n_iterations-1):
        mcmc_obj.mh_step(bnn_obj)
    bnn_obj_new, mcmc_obj_new = mcmc_obj.mh_step(bnn_obj, return_bnn=True)
    return [bnn_obj_new, mcmc_obj_new]

for mc3_it in range(500):

    # Choose the appropriate multiprocessing method based on the OS
    if os.name == 'posix':  # For Unix-based systems (macOS, Linux)
        ctx = multiprocessing.get_context('fork')
    else:  # For Windows
        ctx = multiprocessing.get_context('spawn')

    with ctx.Pool(n_chains) as pool:
        singleChainArgs = list(pool.map(run_single_mcmc, singleChainArgs))

    # with ProcessPoolExecutor(max_workers=n_chains) as pool:
    #     singleChainArgs = list(pool.map(run_single_mcmc, singleChainArgs))
        
    # singleChainArgs = [i for i in tmp]
    if n_chains > 1:
        n1 = np.random.choice(range(n_chains),2,replace=False)
        [j, k] = n1
        temp_j = singleChainArgs[j][1]._temperature + 0
        temp_k = singleChainArgs[k][1]._temperature + 0
        r = (singleChainArgs[k][1]._logPost - singleChainArgs[j][1]._logPost) * temp_j + \
            (singleChainArgs[j][1]._logPost - singleChainArgs[k][1]._logPost) * temp_k
    
        # print(mc3_it, r, singleChainArgs[j][1]._logPost, singleChainArgs[k][1]._logPost, temp_j, temp_k)
        if mc3_it % 100 == 0:
            print(mc3_it, singleChainArgs[0][1]._logPost, singleChainArgs[1][1]._logPost, singleChainArgs[0][0]._w_layers[0][0][0:5])
        if r >= np.log(np.random.random()):
            singleChainArgs[j][1].reset_temperature(temp_k)
            singleChainArgs[k][1].reset_temperature(temp_j)
            print(mc3_it,"SWAPPED", singleChainArgs[j][1]._logPost, singleChainArgs[k][1]._logPost, temp_j, temp_k)

    for i in range(n_chains):
        if singleChainArgs[i][1]._temperature == 1:
            #print( singleChainArgs[i][0]._w_layers[0][0][0:10] )
            logger.log_sample(singleChainArgs[i][0],singleChainArgs[i][1])
            logger.log_weights(singleChainArgs[i][0],singleChainArgs[i][1])
            
    else:
        if mc3_it % 10 == 0:
            print(mc3_it, singleChainArgs[0][1]._logPost, singleChainArgs[0][0]._w_layers[0][0][0:5])

# make predictions based on MCMC's estimated weights
# test data
post_pr_test = bn.predictBNN(dat['test_data'],
                             pickle_file=logger._pklfile,
                             test_labels=dat['test_labels'],
                             instance_id=dat['id_test_data'])

# train+test data
dat = bn.get_data(f, l,
                  testsize=0,  # 10% test set
                  header=1,  # input data has a header
                  instance_id=1)  # input data includes names of instances

post_pr_all = bn.predictBNN(dat['data'],
                            pickle_file=logger._pklfile,
                            test_labels=dat['labels'],
                            instance_id=dat['id_data'])

# predict new unlabeled data
dat = bn.get_data(f="./example_files/unlabeled_data.txt",
                  testsize=0,  # 10% test set
                  header=1,  # input data has a header
                  instance_id=1)  # input data includes names of instances

post_pr_new = bn.predictBNN(dat['data'],
                  pickle_file=logger._pklfile,
                  instance_id=dat['id_data'])



