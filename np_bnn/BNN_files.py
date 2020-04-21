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
def get_data(i,testsize,f=False,l=False, seed=1234):
	np.random.seed(seed)
	if f and l: 
		tot_x = np.load(f)
		tot_labels = np.loadtxt(l,dtype=int)
		x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels,testsize=testsize)
	else: 
		if i == '0':  # ruNNer example files
			f = "/Users/dsilvestro/Software/ruNNer/example_files/example1_training_features.txt"
			l = "/Users/dsilvestro/Software/ruNNer/example_files/example1_training_labels.txt"
			tot_x = np.loadtxt(f)
			tot_labels = np.loadtxt(l)
			x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels)
		elif i == '1':  # IUC-NN data (3 classes)
			f = "/Users/dsilvestro/Documents/Projects/Zizka-IUC-NN/20200213/1_main_iucn_full_clean_three_classes/1_main_iucn_full_clean_detailed_features_fulldsRESCALE.txt"
			l = "/Users/dsilvestro/Documents/Projects/Zizka-IUC-NN/20200213/1_main_iucn_full_clean_three_classes/1_main_iucn_full_clean_detailed_labels_fulldsTHREE.txt"
			tot_x = np.loadtxt(f)
			tot_labels = np.loadtxt(l)
			x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels)
		elif i == '2':  # paleo veg data
			path = "/Users/dsilvestro/Software/BNNs/data/training_data/"
			# path = "/Users/danielesilvestro/Software/BNNs/training_data"
			f_train = os.path.join(path, "feature_arrays/training_features_0.90.npy")
			l_train = os.path.join(path, "feature_arrays/training_labels_0.90.npy")
			f_test = os.path.join(path, "feature_arrays/test_features_0.10.npy")
			l_test = os.path.join(path, "feature_arrays/test_labels_0.10.npy")
			x = np.load(f_train)
			labels = np.load(l_train)
			x_test = np.load(f_test)
			labels_test = np.load(l_test)
			# subset features
			select_features = np.loadtxt(os.path.join(path, "column_indices.txt")).astype(int)
			x = x[:, select_features]
			x_test = x_test[:, select_features]
			# subset traininginstances
			select_input = np.loadtxt(os.path.join(path, "row_indices.txt")).astype(int)
			x = x[select_input]
			labels = labels[select_input]
		elif i == '3':  # sklearn_wine data
			f = "./data/sklearn_wine/training_features.npy"
			l = "./data/sklearn_wine/training_labels.npy"
			tot_x = np.load(f)
			tot_labels = np.load(l)
			x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels,testsize=testsize)
	
		elif i == '4':  # sklearn_wine data
			f = "./data/sklearn_wine/training_features_categories_1_2.npy"
			l = "./data/sklearn_wine/training_labels_categories_1_2.npy"
			tot_x = np.load(f)
			tot_labels = np.load(l)
			x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels,testsize=testsize)
	
		elif i == '5':  # virus data
			f = "./data/virus_sequences/features_and_labels/one_hot_encoded_array_12_classes_NA_sequences_aligned_clipped_100_sequences_for_12_H1_subtypes_first_100_chars.npy"
			l = "./data/virus_sequences/features_and_labels/labels_12_classes_NA_sequences_aligned_clipped_100_sequences_for_12_H1_subtypes_first_100_chars.txt"
			tot_x = np.load(f)
			tot_labels = np.loadtxt(l,dtype=str)
			tot_labels_numeric = turn_labels_to_numeric(tot_labels,l)
			x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels_numeric,testsize=testsize)
	
		elif i == '6':  # virus data, only first half of categories
			f = "./data/virus_sequences/features_and_labels/one_hot_encoded_array_1_6_classes_NA_sequences_aligned_clipped_100_sequences_for_12_H1_subtypes_first_100_chars.npy"
			l = "./data/virus_sequences/features_and_labels/labels_1_6_classes_NA_sequences_aligned_clipped_100_sequences_for_12_H1_subtypes_first_100_chars.txt"
			tot_x = np.load(f)
			tot_labels = np.loadtxt(l,dtype=str)
			tot_labels_numeric = turn_labels_to_numeric(tot_labels,l)
			x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels_numeric,testsize=testsize)
	
		elif i == '7':  # virus data sequence stats
			f = "./data/virus_sequences/features_and_labels/features_sequence_stats_NA_sequences_unaligned_100_sequences_for_12_H1_subtypes.npy"
			l = "./data/virus_sequences/features_and_labels/labels_sequence_stats_NA_sequences_unaligned_100_sequences_for_12_H1_subtypes.txt"
			tot_x = np.load(f)
			tot_labels = np.loadtxt(l,dtype=str)
			try:
				tot_labels_numeric = tot_labels.astype(int)
			except:
				tot_labels_numeric = turn_labels_to_numeric(tot_labels,l)
			x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels_numeric,testsize=testsize)
	
		elif i == '8':  # virus data 6 classes sequence stats
			f = "./data/virus_sequences/features_and_labels/features_sequence_stats_classes_0_5_NA_sequences_unaligned_100_sequences_for_12_H1_subtypes.npy"
			l = "./data/virus_sequences/features_and_labels/labels_sequence_stats_classes_0_5_NA_sequences_unaligned_100_sequences_for_12_H1_subtypes.txt"
			tot_x = np.load(f)
			tot_labels = np.loadtxt(l,dtype=str)
			try:
				tot_labels_numeric = tot_labels.astype(int)
			except:
				tot_labels_numeric = turn_labels_to_numeric(tot_labels,l)
			x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels_numeric,testsize=testsize)
		elif i == '9':  # year data
			f = "./data/year_prediction_MSD/YearPrediction6070_features.npy"
			l = "./data/year_prediction_MSD/YearPrediction6070_labels.npy"
			tot_x = np.load(f)
			tot_labels = np.load(l)
			x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels)

		elif i == '10':  # beta data (only 10 first classes)
			f = "./data/beta_sim_data/betas_features.npy"
			l = "./data/beta_sim_data/betas_labels.npy"
			tot_x = np.load(f)
			tot_labels = np.load(l)
			tot_x = tot_x[tot_labels < 10,:]
			tot_labels = tot_labels[tot_labels < 10]
			x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels)

		elif i == '11':  # beta data (only 10 first classes)
			f = "./data/beta_sim_data/betas_features110.npy"
			l = "./data/beta_sim_data/betas_labels110.npy"
			tot_x = np.load(f)
			tot_labels = np.load(l)
			tot_x = tot_x[tot_labels < 10,:]
			tot_labels = tot_labels[tot_labels < 10]
			x, labels, x_test, labels_test = randomize_data(tot_x, tot_labels)

	labels = labels.astype(int)
	labels = labels - np.min(labels)  # make sure labels start from 0
	if len(labels_test) > 0:
		labels_test = labels_test.astype(int)
		labels_test = labels_test - np.min(labels_test)

	return x, labels, x_test, labels_test

def save_data(dat, lab, outname="data", test_dat=[], test_lab=[]):
	np.savetxt(outname+"_features.txt", dat, delimiter="\t")
	np.savetxt(outname+"_labeles.txt", lab.astype(int), delimiter="\t")
	if len(test_dat) > 0:
		np.savetxt(outname + "_test_features.txt", test_dat, delimiter="\t")
		np.savetxt(outname + "_test_labeles.txt", test_lab.astype(int), delimiter="\t")


def init_output_files(bnn_obj, dat, sample_from_prior=0, outpath=""):
	'prior_f = 0, p_scale = 1, hyper_p = 0, freq_indicator = 0'
	if bnn_obj._freq_indicator ==0:
		ind = ""
	else:
		ind = "_ind"
	if sample_from_prior:
		ind = ind + "_prior"
	outname = "BNN_d%s_p%s_h%s_l%s_s%s_b%s%s" % (dat, bnn_obj._prior,bnn_obj._hyper_p, "_".join(map(str, bnn_obj._n_nodes)), \
	                                         bnn_obj._p_scale, bnn_obj._w_bound, ind)
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


def randomize_data(tot_x, tot_labels, testsize=0.1):
	rnd_order = np.random.choice(range(len(tot_labels)), len(tot_labels), replace=False)
	tot_x = tot_x[rnd_order]
	tot_labels = tot_labels[rnd_order]
	test_set_ind = int(testsize * len(tot_labels))
	if test_set_ind == 0:
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


def turn_labels_to_numeric(labels,label_file):
	numerical_labels = labels.copy()
	for i, label in enumerate(np.unique(numerical_labels)):
		numerical_labels[numerical_labels==label] = i
	numerical_labels = numerical_labels.astype(int)
	outfile = label_file.replace('.txt','_numerical.txt')
	np.savetxt(outfile,numerical_labels,fmt='%i')
	return numerical_labels






