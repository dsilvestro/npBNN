from np_bnn import BNN_lib
import glob

# load test sets
#from mlxtend.data import loadlocal_mnist
x_test, labels_test = loadlocal_mnist(
    images_path='./data/mnist/t10k-images.idx3-ubyte',
    labels_path='./data/mnist/t10k-labels.idx1-ubyte')

x_test = x_test / 255
x_test_in_distribution = x_test[labels_test < 5]
labels_test_in_distribution = labels_test[labels_test < 5]
x_test_out_of_distribution = x_test[labels_test >= 5]
labels_test_out_of_distribution = labels_test[labels_test >= 5]

# calculate Bayes factors
post_pkl = "./d13_l5_5/BNN_d13_p0_h0_l5_5_s5.0_b5.0_4321.pkl"
prior_pkl = "./d13_l5_5_prior_samples/BNN_d13_p0_h0_l5_5_s5.0_b5.0_4321.pkl"
post_pr_in = BNN_lib.predictBNN(x_test_in_distribution, post_pkl,
                                labels_test_in_distribution, pickle_file_prior=prior_pkl)


post_pickle_files = sorted(glob.glob('./d13_l5_5/*h0*4321.pkl'))
prior_pickle_files = sorted(glob.glob('./d13_l5_5_prior_samples/*h0*4321.pkl'))

for i in range(len(post_pickle_files)):
    print("In distribution", post_pickle_files[i])
    post_pr_in = BNN_lib.predictBNN(x_test_in_distribution, post_pickle_files[i],
                                    labels_test_in_distribution,pickle_file_prior=prior_pickle_files[i])
    print("Out of distribution")
    post_pr_out = BNN_lib.predictBNN(x_test_out_of_distribution, post_pickle_files[i],
                                     labels_test_out_of_distribution,pickle_file_prior=prior_pickle_files[i])





#pickle_files = sorted(glob.glob('./d13_l5/*.pkl'))
pickle_files = sorted(glob.glob('./d13_l5_5/*.pkl'))


for pickle_file in pickle_files:
    print("In distribution", pickle_file)
    post_pr_in = BNN_lib.predictBNN(x_test_in_distribution, pickle_file, labels_test_in_distribution)
    print("Out of distribution")
    post_pr_out = BNN_lib.predictBNN(x_test_out_of_distribution, pickle_file, labels_test_out_of_distribution)



















