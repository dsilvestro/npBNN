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



# TF regress
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=[dat['data'].shape[1]]),
        # layers.Dense(60, activation='relu'),
        layers.Dense(8, activation='relu'),
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