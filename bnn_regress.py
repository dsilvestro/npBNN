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
                     n_nodes = [32,8],
                     estimation_mode="regression",
                     p_scale=1,
                     use_bias_node=2
)


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
logger = bn.postLogger(bnn_model, filename="err_est", log_all_weights=0)

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


##

# TF regress
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model = keras.Sequential([
        layers.Dense(100, activation='relu', input_shape=[dat['data'].shape[1]]),
        layers.Dense(60, activation='relu'),
        layers.Dense(8, activation='relu',use_bias=False),
        layers.Dense(2)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()

EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(dat['data'], dat['labels'],
                          epochs=EPOCHS, validation_split=0.1, verbose=1,
                          callbacks=[early_stop])

y_train = model.predict(dat['data'], verbose=2)
y_test = model.predict(dat['test_data'], verbose=2)
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5, 5))
sns.regplot(x=(dat['labels'][:,0].flatten()), y=y_train[:,0].flatten())
sns.regplot(x=(dat['test_labels'][:,0].flatten()), y=y_test[:,0].flatten())
sns.regplot(x=(dat['labels'][:,1].flatten()), y=y_train[:,1].flatten())
sns.regplot(x=(dat['test_labels'][:,1].flatten()), y=y_test[:,1].flatten())
plt.axline((0, 0), (1, 1), linewidth=2, color='k')
fig.show()



