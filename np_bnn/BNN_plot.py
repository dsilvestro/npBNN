import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
import os

from . import BNN_files
from . import BNN_lib

def plotResults(bnn_predictions_file, bnn_lower_file, bnn_upper_file, nn_predictions_file, predictions_outdir, filename_str):

	filename_str = os.path.basename(filename_str)
	try:
		plot_nn = 1
		nn_predictions = np.loadtxt(nn_predictions_file)
	except:
		plot_nn = 0
		nn_predictions = 0
	bnn_predictions = np.loadtxt(bnn_predictions_file)
	bnn_lower = np.loadtxt(bnn_lower_file)
	bnn_upper = np.loadtxt(bnn_upper_file)

	delta_lower = np.abs(bnn_predictions[:,0]-bnn_lower)
	delta_upper = np.abs(bnn_upper-bnn_predictions[:,0])

	fig=plt.figure()
	if plot_nn:
		plt.plot(nn_predictions[:,0],'rx',label='NN predictions')
	plt.plot(bnn_predictions[:,0],'gx',label='BNN predictions')
	plt.axhline(0.5,color='black')
	plt.ylabel('Probability of cat 0')
	plt.xlabel('Prediction instances')
	plt.legend()
	fig.savefig(os.path.join(predictions_outdir, '%s_cat_0_probs_plot.pdf'%filename_str))

	fig=plt.figure()
	if plot_nn:
		plt.plot(nn_predictions[:,0],'rx',label='NN predictions')
	#plt.plot(bnn_predictions[:,0],'gx',label='BNN predictions')
	plt.errorbar(np.arange(bnn_predictions[:,0].shape[0]),bnn_predictions[:,0], yerr=np.array([delta_lower,delta_upper]),
	             fmt='gx',ecolor='black',elinewidth=1,capsize=2,label='BNN predictions')
	plt.axhline(0.5,color='black')
	plt.ylabel('Probability of cat 0')
	plt.xlabel('Prediction instances')
	plt.legend()
	fig.savefig(os.path.join(predictions_outdir, '%s_cat_0_probs_hpd_bars_plot.pdf'%filename_str))

def summarizeOutput(predictions_outdir, pred_features, w_file, nn_predictions_file, use_bias_node):
	if not os.path.exists(predictions_outdir):
		os.makedirs(predictions_outdir)
	loaded_weights = np.array(BNN_files.load_obj(w_file))
	predict_features = np.load(pred_features)
	if use_bias_node:
		predict_features = np.c_[np.ones(predict_features.shape[0]), predict_features]
	# run prediction with these weights
	post_predictions = []
	for weights in loaded_weights:
		pred =  BNN_lib.RunPredict(predict_features, weights)
		post_predictions.append(pred)
	post_predictions = np.array(post_predictions)
	out_name = os.path.splitext(w_file)[0]
	out_name = os.path.basename(out_name)
	
	out_file_post_pr = os.path.join(predictions_outdir, out_name + '_pred_pr.npy')
	out_file_mean_pr = os.path.join(predictions_outdir, out_name + '_pred_mean_pr.txt')
	out_file_upper_pr = os.path.join(predictions_outdir, out_name + '_pred_upper_pr.txt')
	out_file_lower_pr = os.path.join(predictions_outdir, out_name + '_pred_lower_pr.txt')

	# print the arrays to file
	np.save(out_file_post_pr, post_predictions)
	np.savetxt(out_file_mean_pr, np.mean(post_predictions, axis=0), fmt='%.3f')

	# just for plotting reasons, calculate the hpd interval of the first category predictions
	lower = [BNN_lib.calcHPD(point, 0.95)[0] for point in post_predictions[:, :, 0].T]
	upper = [BNN_lib.calcHPD(point, 0.95)[1] for point in post_predictions[:, :, 0].T]
	np.savetxt(out_file_upper_pr, upper, fmt='%.3f')
	np.savetxt(out_file_lower_pr, lower, fmt='%.3f')

	plotResults(out_file_mean_pr, out_file_lower_pr, out_file_upper_pr, nn_predictions_file, predictions_outdir, out_name)
