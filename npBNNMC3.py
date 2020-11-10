import numpy as np
from concurrent.futures import ProcessPoolExecutor
np.set_printoptions(suppress=1)  # prints floats, no scientific notation
np.set_printoptions(precision=3)  # rounds all array elements to 3rd digit

from np_bnn import BNN_env, BNN_files, BNN_lib

# set random seed
rseed = 1234
np.random.seed(rseed)

# load data (2 files: features and labels)
f= "./example_files/data_features.txt"
l= "./example_files/data_labels.txt"
dat = BNN_files.get_data(f,l,
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

n_chains = 10
rseeds = np.random.choice(range(1000,9999), n_chains, replace=False)


bnnList = [BNN_env.npBNN(dat, n_nodes = n_nodes_list, use_class_weights=use_class_weight,
                        use_bias_node = 1, prior_f = prior, p_scale = p_scale,
                        seed=rseeds[i], init_std=0.1)
           for i in range(n_chains)]


if n_chains == 1:
    temperatures = [1]
else:
    temperatures = np.linspace(0.8, 1, n_chains)
mcmcList = [BNN_env.MCMC(bnnList[i],
                         update_f=[0.05, 0.04, 0.07, 0.02], update_ws=[0.075, 0.075, 0.075],
                         temperature=temperatures[i], likelihood_tempering=1.2,
                         n_iteration=100, sampling_f=100, print_f=1000, n_post_samples=100,
                         mcmc_id=i, randomize_seed=True)
            for i in range(n_chains)]


singleChainArgs = [[bnnList[i],mcmcList[i]] for i in range(n_chains)]
n_iterations = 100
# initialize output files
logger = BNN_env.postLogger(bnnList[0], filename="BNNMC3", log_all_weights=0)

def run_single_mcmc(arg_list):
    [bnn_obj, mcmc_obj] = arg_list
    for i in range(n_iterations-1):
        mcmc_obj.mh_step(bnn_obj)
    bnn_obj_new, mcmc_obj_new = mcmc_obj.mh_step(bnn_obj, return_bnn=True)
    return [bnn_obj_new, mcmc_obj_new]

for mc3_it in range(500):
    with ProcessPoolExecutor(max_workers=n_chains) as pool:
        singleChainArgs = list(pool.map(run_single_mcmc, singleChainArgs))
        
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
post_pr_test = BNN_lib.predictBNN(dat['test_data'],
                                  pickle_file=logger._w_file,
                                  test_labels=dat['test_labels'],
                                  instance_id=dat['id_test_data'])

# train+test data
dat = BNN_files.get_data(f, l,
                         testsize=0,  # 10% test set
                         header=1,  # input data has a header
                         instance_id=1)  # input data includes names of instances

post_pr_all = BNN_lib.predictBNN(dat['data'],
                                 pickle_file=logger._w_file,
                                 test_labels=dat['labels'],
                                 instance_id=dat['id_data'])

# predict new unlabeled data
dat = BNN_files.get_data(f="./example_files/unlabeled_data.txt",
                         testsize=0,  # 10% test set
                         header=1,  # input data has a header
                         instance_id=1)  # input data includes names of instances

post_pr_new = BNN_lib.predictBNN(dat['data'],
                                 pickle_file=logger._w_file,
                                 instance_id=dat['id_data'])



