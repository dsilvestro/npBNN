import numpy as np
np.set_printoptions(suppress=True, precision=3)
# load BNN package
import np_bnn as bn
import scipy.stats
# set random seed
rseed = 1234
np.random.seed(rseed)

import numpy as np
np.set_printoptions(suppress=True, precision=3)

# generate data
a = np.random.gamma(2,2,1000)
b = np.random.random(len(a))*a
x = np.random.uniform(1,4,1000)
y0 = scipy.stats.gamma.logpdf(x,a,scale=1/b)
y1 = scipy.stats.gamma.logpdf(x-1,a,scale=1/(a+b)) + 5
features = np.array([x,a,b]).T
labels = np.array([y0,y1]).T
np.savetxt("/Users/dsilvestro/Software/npBNN/example_files/data_features_reg.txt", features)
np.savetxt("/Users/dsilvestro/Software/npBNN/example_files/data_lab_reg.txt", labels)

f="/Users/dsilvestro/Software/npBNN/example_files/data_features_reg.txt"
l="/Users/dsilvestro/Software/npBNN/example_files/data_lab_reg.txt"


dat = bn.get_data(f,
                  l,
                  seed=1234,
                  testsize=0.1, # 10% test set
                  all_class_in_testset=0,
                  cv=0, # cross validation (1st batch; set to 1,2,... to run on subsequent batches)
                  header=0, # input data has a header
                  from_file=True,
                  randomize_order=False,
                  label_mode="regression")

dat['labels'] = labels[:len(dat['labels']),:]
dat['test_labels'] = labels[-len(dat['test_labels']):,:]

# set up the BNN model
bnn_model = bn.npBNN(dat,
                     n_nodes = [32,8],
                     estimation_mode="regression"
)


# set up the MCMC environment
mcmc = bn.MCMC(bnn_model,
               update_ws=[0.025,0.025, 0.05],
               update_f=[0.005,0.005,0.05],
               n_iteration=250000,
               sampling_f=100,
               print_f=1000,
               n_post_samples=100,
               likelihood_tempering=1)



mcmc._accuracy_lab_f(mcmc._y, bnn_model._labels)
# initialize output files
logger = bn.postLogger(bnn_model, filename="BNN", log_all_weights=0)

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




