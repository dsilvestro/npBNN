import numpy as np
np.set_printoptions(suppress=True, precision=3)
# load BNN package
import np_bnn as bn
import scipy.stats
# set random seed
rseed = 1234
np.random.seed(rseed)

# generate data
make_up_data = 0
if make_up_data:
    a = np.random.gamma(2,2,1000)
    b = np.random.random(len(a))*a
    x = np.random.uniform(1,4,1000)
    y0 = scipy.stats.gamma.logpdf(x,a,scale=1/b)
    y1 = scipy.stats.gamma.logpdf(x-1,a,scale=1/(a+b)) + 5
    features = np.array([x,a,b]).T
    labels = np.array([y0,y1]).T
    np.savetxt("./example_files/data_features_reg.txt", features)
    np.savetxt("./example_files/data_lab_reg.txt", labels)

f="./example_files/data_features_reg.txt"
l="./example_files/data_lab_reg.txt"


dat = bn.get_data(f,
                  l,
                  seed=1234,
                  testsize=0.1, # 10% test set
                  all_class_in_testset=0,
                  cv=0, # cross validation (1st batch; set to 1,2,... to run on subsequent batches)
                  header=0, # input data has a header
                  from_file=True,
                  instance_id=0,
                  randomize_order=True,
                  label_mode="regression")


# set up the BNN model
bnn_model = bn.npBNN(dat,
                     n_nodes = [6,2],
                     estimation_mode="regression",
                     actFun = bn.ActFun(fun="tanh"),
                     p_scale=1,
                     use_bias_node=0)


# set up the MCMC environment
mcmc = bn.MCMC(bnn_model,
               update_ws=[0.025,0.025, 0.05],
               update_f=[0.005,0.005,0.05],
               n_iteration=200000,
               sampling_f=100,
               print_f=1000,
               n_post_samples=100,
               likelihood_tempering=1,
               adapt_f=0.3,
               estimate_error=True)

# mcmc._update_n = [1,1,1]
print(mcmc._update_n)

mcmc._accuracy_lab_f(mcmc._y, bnn_model._labels)
# initialize output files
logger = bn.postLogger(bnn_model, filename="err_est_MASK", log_all_weights=0)

# run MCMC
bn.run_mcmc(bnn_model, mcmc, logger)


import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5, 5))
# sns.regplot(x=dat['labels'][:,0].flatten(),y=mcmc._y[:,0])
sns.regplot(x=(dat['labels'][:,0].flatten()),y=(mcmc._y[:,0]))
sns.regplot(x=(dat['test_labels'][:,0].flatten()),y=(mcmc._y_test[:,0]))
sns.regplot(x=(dat['labels'][:,1].flatten()),y=(mcmc._y[:,1]))
sns.regplot(x=(dat['test_labels'][:,1].flatten()),y=(mcmc._y_test[:,1]))
# sns.regplot(x=dat['labels'][:,1].flatten(),y=mcmc._y[:,1])
plt.axline((0, 0), (1, 1), linewidth=2, color='k')
fig.show()

#### run predict
bnn_obj, mcmc_obj, logger_obj = bn.load_obj(logger._pklfile)
post_samples = logger_obj._post_weight_samples

# load posterior weights
post_weights = [post_samples[i]['weights'] for i in range(len(post_samples))]
post_alphas = [post_samples[i]['alphas'] for i in range(len(post_samples))]
actFun = bnn_obj._act_fun
output_act_fun = bnn_obj._output_act_fun

post_cat_probs = []
for i in range(len(post_weights)):
    actFun_i = actFun
    actFun_i.reset_prm(post_alphas[i])
    pred = bn.RunPredict(bnn_obj._data, post_weights[i], actFun=actFun_i, output_act_fun=output_act_fun)
    post_cat_probs.append(pred)


