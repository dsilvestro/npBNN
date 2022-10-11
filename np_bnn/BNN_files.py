import csv
import glob
import os
import numpy as np
import pickle
from .BNN_lib import *

# get data
def get_data(f,l=None,testsize=0.1, batch_training=0,seed=1234, all_class_in_testset=1,
             instance_id=0, header=0,feature_indx=None,randomize_order=True,from_file=True,
             label_mode="classification", cv=-1):
    np.random.seed(seed)
    inst_id = []
    if from_file:
        fname = os.path.splitext(os.path.basename(f))[0]
        try:
            tot_x = np.load(f)
        except:
            if not instance_id:
                tot_x = np.loadtxt(f, skiprows=header)
            else:
                tmp = np.genfromtxt(f, skip_header=header, dtype=str)
                tot_x = tmp[:,1:].astype(float)
                inst_id = tmp[:,0].astype(str)
            
        if header:
            feature_names = np.array(next(open(f)).split()[1:])
        else:
            feature_names = np.array(["feature_%s" % i for i in range(tot_x.shape[1])])
    else:
        f = pd.DataFrame(f)
        fname = 'bnn'
        if not instance_id:
            feature_names = np.array(f.columns)
            tot_x = f.values
        else:
            feature_names = np.array(f.columns[1:])
            tot_x = f.values[:,1:]
            inst_id = f.values[:,0].astype(str)

    if feature_indx is not None:
        feature_indx = np.array(feature_indx)
        tot_x = tot_x[:,feature_indx]
        feature_names = feature_names[feature_indx]

    if l is None:
        out_dict = {'data': np.array(tot_x).astype(float), 'labels': [], 'label_dict': [],
                'test_data': [], 'test_labels': [],
                'id_data': inst_id, 'id_test_data': [],
                'file_name': fname, 'feature_names': feature_names}
    else:
        try:
            l = pd.DataFrame(l)
            if instance_id:
                tot_labels = l.values[:, 1]
            else:
                tot_labels = l.values.astype(str).flatten()  # if l already is a dataframe
        except:
            tot_labels = np.loadtxt(l,skiprows=header,dtype=str)

            if instance_id:
                tot_labels = tot_labels[:, 1]

        if label_mode == "classification":
            tot_labels_numeric = turn_labels_to_numeric(tot_labels, l)
        else:
            if len(tot_labels.shape) == 1:
                tot_labels_numeric = tot_labels.reshape((tot_labels.shape[0], 1))
            else:
                tot_labels_numeric = tot_labels
        x, labels, x_test, labels_test, inst_id_x, inst_id_x_test = randomize_data(tot_x, tot_labels_numeric,
                                                                                   testsize=testsize,
                                                                                   all_class_in_testset=all_class_in_testset,
                                                                                   inst_id=inst_id,
                                                                                   randomize=randomize_order,
                                                                                   seed=seed,
                                                                                   cv=cv)

        if batch_training:
            indx = np.random.randint(0,len(labels),batch_training)
            x = x[indx]
            labels = labels[indx]

        if label_mode == "regression":
            labels = labels.astype(float)
            if testsize:
                labels_test = labels_test.astype(float)

        out_dict = {'data': np.array(x).astype(float), 'labels': labels, 'label_dict': np.unique(tot_labels),
                'test_data': np.array(x_test).astype(float), 'test_labels': labels_test,
                'id_data': inst_id_x, 'id_test_data': inst_id_x_test,
                'file_name': fname, 'feature_names': feature_names}
    return out_dict


def save_data(dat, lab, outname="data", test_dat=[], test_lab=[]):
    np.savetxt(outname+"_features.txt", dat, delimiter="\t")
    np.savetxt(outname+"_labeles.txt", lab.astype(int), delimiter="\t")
    if len(test_dat) > 0:
        np.savetxt(outname + "_test_features.txt", test_dat, delimiter="\t")
        np.savetxt(outname + "_test_labeles.txt", test_lab.astype(int), delimiter="\t")

def init_output_files(bnn_obj, filename="BNN", sample_from_prior=0, outpath="",add_prms=None,
                      continue_logfile=False, log_all_weights=0):
    # create output dir
    outdir = os.path.dirname(filename)
    if len(outdir)>0:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    'prior_f = 0, p_scale = 1, hyper_p = 0, freq_indicator = 0'
    if bnn_obj._freq_indicator ==0:
        ind = ""
    else:
        ind = "_ind"
    if sample_from_prior:
        ind = ind + "_prior"
    ind =  ind + "_%s" % bnn_obj._seed
    outname = "%s_p%s_h%s_l%s_s%s_b%s%s" % (filename, bnn_obj._prior,bnn_obj._hyper_p, "_".join(map(str,
                                            bnn_obj._n_nodes)), bnn_obj._p_scale, bnn_obj._w_bound, ind)

    logfile_name = os.path.join(outpath, outname + ".log")
    if log_all_weights:
        w_file_name = os.path.join(outpath, outname + "_W.log")
        head_w = ["it"]
    else:
        w_file_name = None

    head = ["it", "posterior", "likelihood", "prior"]
    if bnn_obj._estimation_mode == "classification":
        head = head + ["accuracy", "test_accuracy"]
        for i in range(bnn_obj._n_output_prm):
            head.append("acc_C%s" % i)
    else:
        head = head + ["MSE", "test_MSE"]
        for i in range(bnn_obj._n_output_prm):
            head.append("MSE_prm%s" % i)

    for i in range(bnn_obj._n_layers):
        head.append("mean_w%s" % i)
        head.append("std_w%s" % i)
        if bnn_obj._hyper_p:
            if bnn_obj._hyper_p == 1:
                head.append("prior_std_w%s" % i)
            else:
                head.append("mean_prior_std_w%s" % i)
        if log_all_weights:
            head_w = head_w + ["w_%s_%s" % (i, j) for j in range(bnn_obj._w_layers[i].size)]
    if bnn_obj._freq_indicator:
        head.append("mean_ind")
    if add_prms:
        head = head + add_prms

    if bnn_obj._act_fun._trainable:
        head = head + ["alpha_%s" % (i) for i in range(bnn_obj._n_layers-1)]
    
    head.append("acc_prob")
    head.append("mcmc_id")
    
    if not continue_logfile:
        logfile_IO = open(logfile_name, "w")
        wlog = csv.writer(logfile_IO, delimiter='\t')
        wlog.writerow(head)

    if log_all_weights:
        w_file_IO = open(w_file_name, "w")
        wweights = csv.writer(w_file_IO, delimiter='\t')
        wweights.writerow(head_w)

    pkl_file = os.path.join(outpath, outname + ".pkl")
    return logfile_name, w_file_name, pkl_file


def randomize_data(tot_x, tot_labels, testsize=0.1, all_class_in_testset=1, inst_id=[], randomize=True, seed=1234, cv=-1):
    np.random.seed(seed)
    if randomize:
        if testsize:
            rnd_order = np.random.choice(range(len(tot_labels)), len(tot_labels), replace=False)
        else:
            rnd_order = np.arange(len(tot_labels))
    else:
        rnd_order = np.arange(len(tot_labels))
        all_class_in_testset=0
    tot_x = tot_x[rnd_order]
    tot_labels = tot_labels[rnd_order]
    test_set_ind = int(testsize * len(tot_labels))
    inst_id_x = []
    inst_id_test = []
    tot_inst_id = []
    if len(inst_id):
        tot_inst_id = inst_id[rnd_order]

    if cv > -1 and testsize:
        ind_start = test_set_ind * cv
        ind_end = np.min([ind_start + test_set_ind, len(tot_labels)])
        indx_test_set = range(ind_start, ind_end)
        x_test = tot_x[indx_test_set, :]
        labels_test = tot_labels[indx_test_set]
        x = np.delete(tot_x, indx_test_set, axis=0)
        labels = np.delete(tot_labels, indx_test_set, axis=0)
        if len(inst_id):
            inst_id_test = tot_inst_id[indx_test_set]
            inst_id_x = np.delete(tot_inst_id, indx_test_set)
        print("test set:", indx_test_set)

    elif all_class_in_testset and testsize:
        test_set_ind = []

        for i in np.unique(tot_labels):
            ind = np.where(tot_labels == i)[0]
            test_set_ind = test_set_ind + list(np.random.choice(ind, np.max([1, int(testsize*len(ind))])))

        test_set_ind = np.array(test_set_ind)
        x_test = tot_x[test_set_ind]
        labels_test = tot_labels[test_set_ind]
        train_ind = np.array([z for z in range(tot_labels.size) if not z in test_set_ind])

        x = tot_x[train_ind]
        labels = tot_labels[train_ind]
        if len(inst_id):
            inst_id_x = tot_inst_id[train_ind]
            inst_id_test = tot_inst_id[test_set_ind]
    elif test_set_ind == 0:
        x_test = []
        labels_test = []
        x = tot_x
        labels = tot_labels
        if len(inst_id):
            inst_id_x = tot_inst_id
            inst_id_test = []
    else:
        x_test = tot_x[-test_set_ind:, :]
        labels_test = tot_labels[-test_set_ind:]
        x = tot_x[:-test_set_ind, :]
        labels = tot_labels[:-test_set_ind]
        if len(inst_id):
            inst_id_test = tot_inst_id[-test_set_ind:]
            inst_id_x = tot_inst_id[:-test_set_ind]
         
    return x, labels, x_test, labels_test, inst_id_x, inst_id_test


def load_obj(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except:
        import pickle5
        with open(file_name, 'rb') as f:
            return pickle5.load(f)

def merge_dict(d1, d2):
    from collections import defaultdict
    d = defaultdict(list)
    for a, b in d1.items() + d2.items():
        d[a].append(b)
    return d


def combine_pkls(files=None, dir=None, tag=""):
    if dir is not None:
        files = glob.glob(os.path.join(dir, "*%s*.pkl" % tag))
        print("Combining %s files: \n" % len(files), files)
    i = 0
    w_list = []

    out_file = os.path.join(os.path.dirname(files[0]), "combine_pkl%s.pkl" % tag)
    for f in files:
        if f != out_file:
            a, b, w = load_obj(f)
            if i == 0:
                bnn_obj = a
                mcmc_obj = b
                logger_obj = w

            w_list.append(w._post_weight_samples)

    with open(out_file, 'wb') as output:  # Overwrites any existing file.
        pickle.dump([bnn_obj,mcmc_obj,logger_obj], output, pickle.HIGHEST_PROTOCOL)

    return out_file

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






