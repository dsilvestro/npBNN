import pickle
import numpy as np
import scipy.stats
from np_bnn.BNN_lib import *
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
import os
import csv

from np_bnn import BNN_env

# get data
def get_data(f,l,testsize=0.1, batch_training=0,seed=1234, all_class_in_testset=1):
    np.random.seed(seed)
    try: tot_x = np.load(f)
    except(ValueError): tot_x = np.loadtxt(f)
    tot_labels = np.loadtxt(l,dtype=str)
    tot_labels_numeric = turn_labels_to_numeric(tot_labels, l)
    x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels_numeric,testsize=testsize,
                                                    all_class_in_testset=all_class_in_testset)

    if batch_training:
        indx = np.random.randint(0,len(labels),batch_training)
        x = x[indx]
        labels = labels[indx]
        
    return {'data': x, 'labels': labels, 'label_dict': np.unique(tot_labels),
            'test_data': x_test, 'test_labels': labels_test}


def save_data(dat, lab, outname="data", test_dat=[], test_lab=[]):
    np.savetxt(outname+"_features.txt", dat, delimiter="\t")
    np.savetxt(outname+"_labeles.txt", lab.astype(int), delimiter="\t")
    if len(test_dat) > 0:
        np.savetxt(outname + "_test_features.txt", test_dat, delimiter="\t")
        np.savetxt(outname + "_test_labeles.txt", test_lab.astype(int), delimiter="\t")

def init_output_files(bnn_obj, filename="BNN", sample_from_prior=0, outpath=""):
    'prior_f = 0, p_scale = 1, hyper_p = 0, freq_indicator = 0'
    if bnn_obj._freq_indicator ==0:
        ind = ""
    else:
        ind = "_ind"
    if sample_from_prior:
        ind = ind + "_prior"
    ind =  ind + "_%s" % bnn_obj._seed
    outname = "%s_p%s_h%s_l%s_s%s_b%s%s" % (filename, bnn_obj._prior,bnn_obj._hyper_p, "_".join(map(str, \
                                            bnn_obj._n_nodes)), bnn_obj._p_scale, bnn_obj._w_bound, ind)
                                            
    logfile_name = os.path.join(outpath, outname + ".log")
    w_file_name = os.path.join(outpath, outname + ".pkl")
    logfile = open(logfile_name, "w")
    head = ["it", "posterior", "likelihood", "prior", "accuracy", "test_accuracy"]
    for i in range(bnn_obj._size_output):
        head.append("f_C%s" % i)
    for i in range(bnn_obj._n_layers):
        head.append("mean_w%s" % i)
        head.append("std_w%s" % i)
        if bnn_obj._hyper_p:
            if bnn_obj._hyper_p == 1:
                head.append("prior_std_w%s" % i)
            else:
                head.append("mean_prior_std_w%s" % i)
    if bnn_obj._freq_indicator:
        head.append("mean_ind")
    wlog = csv.writer(logfile, delimiter='\t')
    wlog.writerow(head)
    return wlog, logfile, w_file_name


def randomize_data(tot_x, tot_labels, testsize=0.1, all_class_in_testset=1):
    rnd_order = np.random.choice(range(len(tot_labels)), len(tot_labels), replace=False)
    tot_x = tot_x[rnd_order]
    tot_labels = tot_labels[rnd_order]
    test_set_ind = int(testsize * len(tot_labels))

    if all_class_in_testset:
        test_set_ind = []

        for i in np.unique(tot_labels):
            ind = np.where(tot_labels == i)[0]
            print(i, len(ind))
            test_set_ind = test_set_ind + list(np.random.choice(ind, np.max([1, int(testsize*len(ind))])))

        test_set_ind = np.array(test_set_ind)
        x_test = tot_x[test_set_ind]
        labels_test = tot_labels[test_set_ind]
        train_ind = np.array([z for z in range(tot_labels.size) if not z in test_set_ind])
        print(train_ind)

        x = tot_x[train_ind]
        labels = tot_labels[train_ind]
    elif test_set_ind == 0:
        x_test = []
        labels_test = []
        x = tot_x
        labels = tot_labels
    else:
        x_test = tot_x[0:test_set_ind, :]
        labels_test = tot_labels[0:test_set_ind]
        x = tot_x[test_set_ind:, :]
        labels = tot_labels[test_set_ind:]
    return x, labels, x_test, labels_test


def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def turn_labels_to_numeric(labels,label_file,save_to_file=False):
    numerical_labels = np.zeros(len(labels)).astype(int)
    c = 0
    for i in np.unique(labels):
        numerical_labels[labels == i] = c
        c += 1

    if save_to_file:
        outfile = label_file.replace('.txt','_numerical.txt')
        np.savetxt(outfile,numerical_labels,fmt='%i')
    return numerical_labels






