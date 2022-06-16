import numpy as np
np.set_printoptions(suppress=True, precision=3)
# load BNN package
import np_bnn as bn
import scipy.stats
# set random seed
rseed = 1234
np.random.seed(rseed)

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

# Build sparse NNs

"""
Create network with sparse first layer. Each feature is connected to 2 nodes.
Second and third layers are fully connected
"""
bnn_model = bn.npBNN(dat,
                     n_nodes = [6,2],
                     estimation_mode="regression",
                     actFun = bn.ActFun(fun="tanh"),
                     p_scale=1,
                     use_bias_node=-1) # bias node on last layer

m = bn.create_mask(bnn_model._w_layers,
                   indx_input_list=[[0, 1, 2], [], []],
                   nodes_per_feature_list=[[2, 2, 2], [], []])

bnn_model.apply_mask(m)

"""
Create network with sparse first and second layer. Each feature is connected to a block of 3 nodes
in the first layer. The 3 nodes are connected to 2 nodes in the second layer.
The third layers is fully connected.
"""
bnn_model = bn.npBNN(dat,
                     n_nodes = [9,6],
                     estimation_mode="regression",
                     p_scale=1,
                     use_bias_node=-1
)

m = bn.create_mask(bnn_model._w_layers,
                   indx_input_list=[[0, 1, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2], []],
                   nodes_per_feature_list=[[3, 3, 3], [2, 2, 2], []])

bnn_model.apply_mask(m)



"""
Create network with sparse first and second layer. The first feature is connected to 3 nodes
in the first layer, which then connect to 2 nodes in the second layers. 
The second and third features are grouped in a network with 6 nodes, which then connect to 3 
nodes in the second layer.
The third layers is fully connected.
"""
bnn_model = bn.npBNN(dat,
                     n_nodes = [9,5],
                     estimation_mode="regression",
                     p_scale=1,
                     use_bias_node=-1
)

m = bn.create_mask(bnn_model._w_layers,
                   indx_input_list=[[0, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1], []],
                   nodes_per_feature_list=[[3, 6], [2, 3], []])

bnn_model.apply_mask(m)

