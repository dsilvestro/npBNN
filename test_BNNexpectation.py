import os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=3)
import np_bnn as bn

# load data (2 files: features and labels)
f="./example_files/data_features_reg.txt"
l="./example_files/data_lab_reg.txt"

dat = bn.get_data(f,
                  l,
                  seed=1234,
                  testsize=0.,
                  all_class_in_testset=0,
                  cv=0, # cross validation (1st batch; set to 1,2,... to run on subsequent batches)
                  header=True, # input data has a header
                  from_file=True,
                  instance_id=0,
                  randomize_order=True,
                  label_mode="regression")

# examples output functions
def logistic_t(z, t=0.5, k=0.75, x0=0):
    return t / (1 + np.exp(-k * (z - x0)))

def fixed_sigmoid(z, lower=0, upper=0.5):
    return lower + (upper - lower) / (1 + np.exp(-z))

def exponential_t(z, t=2):
    return np.minimum(np.exp(z), t)

# set up the BNN model
bnn_obj = bn.npBNN(dat,
                   n_nodes = [6,4],
                   estimation_mode="regression",
                   actFun = bn.ActFun(fun="tanh"),
                   prior_f=1, #
                   p_scale=1,
                   use_bias_node=2,
                   output_act_fun=logistic_t,
                   empirical_error=True)


reps = 2
pred = np.zeros((1, bnn_obj._n_output_prm))
for i in range(reps):
    bnn_obj.sample_from_prior(reset_weights=True)
    x = bn.predict(bnn_obj, bnn_obj._data)
    pred = np.vstack((pred, x))


plt.hist(pred[1:,1])
plt.show()
