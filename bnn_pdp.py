import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=3)
# load BNN package
import np_bnn as bn
import scipy.stats
# set random seed
rseed = 1234
np.random.seed(rseed)

# features
f = "./example_files/pdp_features.txt"
# labels for regression
l_reg = "./example_files/pdp_lab_reg.txt"
# labels for classification
l_class = "./example_files/pdp_lab_class.txt"


#--------------------#
# regression example #
#--------------------#

dat_reg = bn.get_data(f,
                      l_reg,
                      seed=1234,
                      testsize=0.1, # 10% test set
                      all_class_in_testset=0,
                      cv=0, # no cross validation
                      header=True, # input data has a header
                      from_file=True,
                      instance_id=1, # input data has instance names in first column
                      randomize_order=True,
                      label_mode="regression")


# set up the BNN model
bnn_reg_model = bn.npBNN(dat_reg,
                         n_nodes = [6,4],
                         estimation_mode="regression",
                         actFun = bn.ActFun(fun="tanh"),
                         p_scale=1,
                         use_bias_node=2,
                         empirical_error=True)


# set up the MCMC environment
mcmc_reg = bn.MCMC(bnn_reg_model,
                   update_ws=[0.025,0.025, 0.05],
                   update_f=[0.005,0.005,0.05],
                   n_iteration=20000,
                   sampling_f=100,
                   print_f=1000,
                   n_post_samples=100,
                   likelihood_tempering=1,
                   adapt_f=0.3,
                   estimate_error=False)


# initialize output files
logger_reg = bn.postLogger(bnn_reg_model, filename="reg")

# run MCMC
bn.run_mcmc(bnn_reg_model, mcmc_reg, logger_reg)

# partial dependence plots for the first feature (continuous), second feature (ordinal),
# and the one-hot encoded feature in the last three columns
pd_reg = bn.pdp(logger_reg._pklfile, [[0], [1], [4, 5, 6]])

# partial dependence plot for the continuous feature
pd0 = pd_reg[0]
x = pd0["feature"].flatten()
y_mean = pd0["pdp"][:, 0, 0]
y_lwr = pd0["pdp"][:, 0, 1]
y_upr = pd0["pdp"][:, 0, 2]

plt.figure(figsize=(7, 4))
plt.plot(x, y_mean, color="C0")
plt.fill_between(x, y_lwr, y_upr, color="C0", alpha=0.3)
plt.xlabel("x1")
plt.ylabel("y")
plt.show()

# partial dependence plot for the ordinal feature
pd1 = pd_reg[1]
categories = pd1["feature"].flatten().astype(str)
means = pd1["pdp"][:, 0, 0]
yerr = [means - pd1["pdp"][:, 0, 1], pd1["pdp"][:, 0, 2] - means]

plt.figure(figsize=(5, 4))
plt.errorbar(categories, means, yerr=yerr, fmt='o', capsize=5, color='C0', markersize=8)
plt.show()

# partial dependence plot for one-hot encoded feature
pd2 = pd_reg[2]
categories = np.arange(pd2["feature"].shape[0]).astype(str)
means = pd2["pdp"][:, 0, 0]
yerr = [means - pd2["pdp"][:, 0, 1], pd2["pdp"][:, 0, 2] - means]

plt.figure(figsize=(5, 4))
plt.errorbar(categories, means, yerr=yerr, fmt='o', capsize=5, color='C0', markersize=8)
plt.show()


#------------------------#
# classification example #
#------------------------#

dat_class = bn.get_data(f,
                        l_class,
                        seed=rseed,
                        testsize=0.1, # 10% test set
                        all_class_in_testset=1,
                        header=1, # input data has a header
                        cv=0,
                        instance_id=1) # input data includes names of instances

# set up the BNN model
bnn_class_model = bn.npBNN(dat_class,
                           n_nodes = [6, 4],
                           use_class_weights=0,
                           actFun=bn.ActFun(fun="tanh"),
                           use_bias_node=2,
                           prior_f=1,
                           p_scale=1,
                           seed=rseed,
                           init_std=0.1,
                           instance_weights=None)

mcmc_class = bn.MCMC(bnn_class_model,
                     update_f=[0.05, 0.05, 0.07],
                     update_ws=[0.075, 0.075, 0.075],
                     n_iteration=10000,
                     sampling_f=10,
                     print_f=1000,
                     n_post_samples=100,
                     adapt_f=0.3, # target acceptance probability (min)
                     adapt_fM=0.6 # target acceptance probability (max)
                     )

# initialize output files
logger_class = bn.postLogger(bnn_class_model, filename="class")

# run MCMC
bn.run_mcmc(bnn_class_model, mcmc_class, logger_class)

# partial dependence plots for the first feature (continuous), second feature (ordinal),
# and the one-hot encoded feature in the last three columns
pd_class = bn.pdp(logger_class._pklfile, [[0], [1], [4, 5, 6]])

# partial dependence plot for the continuous feature without uncertainty
pd0 = pd_class[0]
x = pd0["feature"].flatten()
y_mean = pd0["pdp"][:, :, 0]
y_lwr = pd0["pdp"][:, :, 1]
y_upr = pd0["pdp"][:, :, 2]
y_bottom = np.hstack([np.zeros((len(x), 1)), y_mean[:, :-1]])

plt.figure(figsize=(7, 4))
colors = plt.cm.viridis(np.linspace(0, 1, y_mean.shape[1]))
for i in range(y_mean.shape[1]):
    plt.fill_between(x, y_bottom[:, i], y_mean[:, i], color=colors[i])

plt.show()

# partial dependence plot for one-hot encoded feature without uncertainty
pd2 = pd_class[2]
categories = np.arange(pd2["feature"].shape[0])
y_mean = pd2["pdp"][:, :, 0]

y_bottom = np.hstack([np.zeros((y_mean.shape[0], 1)), y_mean[:, :-1]])
y_heights = y_mean - y_bottom

x = np.arange(y.shape[0])
colors = plt.cm.viridis(np.linspace(0, 1, y_mean.shape[1]))

plt.figure(figsize=(5, 4))

for i in range(y_mean.shape[1]):
    plt.bar(categories, y_heights[:, i], bottom=y_bottom[:, i],
            color=colors[i], edgecolor="black")

plt.xticks(categories, categories)
plt.tight_layout()
plt.show()

