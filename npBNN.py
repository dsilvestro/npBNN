import pickle
import numpy as np
import scipy.stats
import glob,os,sys,argparse
import pickle
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit

from np_bnn import BNN_env
from np_bnn import BNN_files
from np_bnn import BNN_lib
from np_bnn import BNN_plot
import time

run_cmd_line = 0
run_mcmc = 1
run_predict = 1




if run_cmd_line:
    p = argparse.ArgumentParser()  # description='<input file>')
    p.add_argument("-seed", type=int, help="", default=1234, metavar=1234)
    p.add_argument("-p", type=int, help="0) uniform, 1) normal, 2) cauchy", default=1, metavar=1)
    p.add_argument("-d", type=str, help="", default='999', metavar='999')
    p.add_argument("-hp", type=int, help="hyperprior: 1-3", default=0, metavar=0)
    p.add_argument("-i", type=int, help="1: use indicators", default=0, metavar=0)
    p.add_argument("-n", type=int, nargs="+", default = [20, 5])
    p.add_argument("-t", type=float, help="define test size", default=0.1, metavar=0.1)
    p.add_argument("-b", type=float, help="define bounds prior", default=np.infty, metavar=np.infty)
    p.add_argument("-scale", type=float, help="scale prior", default=1, metavar=1)
    p.add_argument("-f", type=str, help="feature array (.npy file)", default=0, metavar=0)
    p.add_argument("-l", type=str, help="label array (.txt file, integer labels only)", default=0, metavar=0)
    p.add_argument("-reload", type=str, help="load pickle file to restart MCMC", default="", metavar="")

    args = p.parse_args()
    rseed = args.seed
    np.random.seed(rseed)
    prior_mode = args.p
    data_set = args.d
    n_nodes_list = args.n
    h_prior = args.hp
    f_ind = args.i * 0.25
    prior = args.p
    test_size = args.t
    p_scale = args.scale
    w_bound = args.b
    pickle_file = args.reload
    f = args.f
    l = args.l
    if h_prior:
        init_sd = 1
    else:
        init_sd = 0.1

else:
    rseed = 1234
    np.random.seed(rseed)
    data_set = '11'
    n_nodes_list = [15, 10]
    h_prior = 0 # 1) '1 Hyp / layer' 2) '1 Hyp / input node / layer' 3) '1 Hyp / weight / layer'
    f_ind = 0 # set to >0 <1 to use indicators
    prior = 3 # 0) uniform, 1) normal, 2) cauchy, 3) Laplace
    w_bound = np.infty
    p_scale = 1
    test_size = 0.1
    f= 0
    l= 0
    if h_prior:
        init_sd = 1
    else:
        init_sd = 0.1
    pickle_file = "/Users/danielesilvestro/Software/BNNs/data/beta_sim_data/BNN_d10_p2_h0_l20_15_s1.0_b5.0.pkl"

if f and l:
    data, labels, test_data, test_labels = BNN_files.get_data(data_set,test_size,f,l,seed=rseed)
else:
    data, labels, test_data, test_labels = BNN_files.get_data(data_set,test_size,seed=rseed)
    BNN_files.save_data(data, labels, outname="./data", test_dat=test_data, test_lab=test_labels)

bnn = BNN_env.npBNN(data, labels, test_data, test_labels, n_nodes = n_nodes_list,
                 n_layers = len(n_nodes_list)+1, use_bias_node = 1, init_std = init_sd,
                 prior_f = prior, p_scale = p_scale, hyper_p = h_prior, freq_indicator = f_ind,
                 w_bound = w_bound, pickle_file=pickle_file)

if bnn._hyper_p:
    'init prior scales'
    bnn.sample_prior_scale()

if len(bnn._test_data) >0:
    print("\nTraining set:", bnn._n_samples, "test set:", bnn._test_data.shape[0])
else:
    print("\nTraining set:", bnn._n_samples, "test set:", None)
print("Number of features:", bnn._n_features)
print("N. of parameters:", np.sum(np.array([np.size(i) for i in bnn._w_layers])))
for w in bnn._w_layers: print(w.shape)

mcmc = BNN_env.MCMC(bnn,update_f=[0.04, 0.04, 0.07, 0.02], update_ws=[0.075, 0.075, 0.075],data_set=data_set,
                 temperature = 1, n_iteration=10000000, sampling_f=1000, print_f=1000, n_post_samples=1000,
                 update_function=BNN_lib.UpdateNormal, sample_from_prior=1)
# run MCMC
if run_mcmc:
    tStart = time.time()
    while True:
        rr = np.random.random()
        if rr < 0.005 and bnn._hyper_p > 0:
            mcmc.gibbs_step(bnn)
        else:
            mcmc.mh_step(bnn)

        if mcmc._current_iteration % mcmc._print_f == 0 or mcmc._current_iteration == 1:
            print(mcmc._current_iteration, np.array([mcmc._logPost, mcmc._logLik, mcmc._logPrior,
                    mcmc._accuracy, mcmc._test_accuracy, np.mean(bnn._indicators)]))

        if mcmc._current_iteration % mcmc._sampling_f == 0:
            mcmc.log_sample(bnn)
            mcmc.log_weights(bnn)

        if mcmc._current_iteration == mcmc._n_iterations:
            break

    print('Time: %.1f s' %(time.time() - tStart))

if run_predict:
    # load the weights sampled during BNN training
    use_bias_node = bnn._use_bias_node
    w_file = mcmc._w_file
    nn_predictions_file = ""
    pred_features = "./data/sklearn_wine/training_features.npy"
    predictions_outdir = "."
    BNN_plot.summarizeOutput(predictions_outdir, pred_features, w_file, nn_predictions_file, use_bias_node)

    # load the weights sampled during BNN training
    use_bias_node = bnn._use_bias_node
    w_file = mcmc._w_file
    nn_predictions_file = "./predicted_labels/NN_1_2/training_features_categories_1_2labelsPr_.txt"
    pred_features = './data/sklearn_wine/training_features_categories_1_2.npy'
    predictions_outdir = './predicted_labels/BNN_1_2'
    BNN_plot.summarizeOutput(predictions_outdir, pred_features, w_file, nn_predictions_file, use_bias_node)
    nn_predictions_file = "./predicted_labels/NN_3/training_features_category_3labelsPr_.txt"
    pred_features = './data/sklearn_wine/training_features_category_3.npy'
    predictions_outdir = './predicted_labels/BNN_3'
    BNN_plot.summarizeOutput(predictions_outdir, pred_features, w_file, nn_predictions_file, use_bias_node)