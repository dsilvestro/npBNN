import numpy as np
import scipy.stats
import scipy.special
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
import pickle
small_number = 1e-10
import random, sys
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from .BNN_files import *
from .BNN_lib import *
import os

# if alpha < 1 and non trainable: leaky ReLU (https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
# if trainable: parameteric ReLU (https://arxiv.org/pdf/1502.01852.pdf)

def create_mask(w_layers, indx_input_list, nodes_per_feature_list):
    m_layers = []
    for w in w_layers:
        # w = np.random.random((9,3))
        # indx_features = [0, 1, 2]
        # nodes_per_feature = [4, 3, 2]
        # indx_features = [0, 0, 2]
        # nodes_per_feature = [5, 4]
        indx_features = indx_input_list[len(m_layers)]
        nodes_per_feature = nodes_per_feature_list[len(m_layers)]
        if len(indx_features) == 0:
            # fully connect
            m = np.ones(w.shape)
        else:
            m = np.zeros(w.shape)
            max_indx_rows = 0
            j = 0
            for i in range(len(indx_features)):
                if i > 0:
                    if indx_features[i] != indx_features[i - 1]:
                        j += 1
                        indx_rows = np.arange(nodes_per_feature[j]) + max_indx_rows
                else:
                    indx_rows = np.arange(nodes_per_feature[j])
                indx_cols = np.repeat(i, nodes_per_feature[j])
                m[indx_rows, indx_cols] = 1
                # indx_cols2 = np.repeat(indx_features[i], nodes_per_feature[j])
                # m[indx_rows, indx_cols2] = 1
                max_indx_rows = np.max(indx_rows) + 1

        m_layers.append(m)
    return m_layers


def relu_f(z, _):
    z[z < 0] = 0
    return z

def leaky_relu_f(z, prm):
    z[z < 0] = prm * z[z < 0]
    return z

def swish_f(z, _):
    # https://arxiv.org/abs/1710.05941
    z = z * (1 + np.exp(-z))**(-1)
    return z

def tanh_f(z, _):
    return np.tanh(z)

class ActFun():
    def __init__(self, fun='ReLU', prm=np.zeros(1), trainable=False):
        self._prm = prm
        self._acc_prm = prm
        self._trainable = trainable
        self._function = fun
        if fun == "ReLU":
            self.activate = relu_f
        if fun == "genReLU" or trainable is True:
            self.activate = leaky_relu_f
        if fun == "swish":
            self.activate = swish_f
        if fun == "tanh":
            self.activate = tanh_f

    def eval(self, z, layer_n):
        if self._function == "genReLU":
            return self.activate(z, self._prm[layer_n])
        else:
            return self.activate(z, 0)


    def reset_prm(self, prm):
        self._prm = prm

    def reset_accepted_prm(self):
        self._acc_prm = self._prm + 0



# likelihood function (Categorical)
# TODO: refactor this as a class
def calc_likelihood(prediction, labels, sample_id, class_weight=[], lik_temp=1, _=0):
    if len(class_weight):
        return lik_temp * np.sum(np.log(prediction[sample_id, labels])*class_weight[labels])
    else:
        # if lik_temp != 1:
        #     tempered_prediction = lik_temp ** prediction
        #     normalized_tempered_prediction = np.einsum('xy,x->xy', tempered_prediction, 1 / np.sum(tempered_prediction,axis=1))
        #     return np.sum(np.log(normalized_tempered_prediction[sample_id, labels]))
        # else:
        return lik_temp * np.sum(np.log(prediction[sample_id, labels]))

def calc_likelihood_regression(prediction, # 2D array: inst x (mus + sigs)
                               true_values, # 2D array: val[inst x n_param
                               _, __,
                               lik_temp=1,
                               sig2=1):
    return lik_temp * np.sum(scipy.stats.norm.logpdf(true_values, prediction, sig2))


def calc_likelihood_regression_error(prediction, # 2D array: inst x (mus + sigs)
                               true_values, # 2D array: val[inst x n_param
                               _, __,
                               lik_temp=1,
                               ___=0):
    ind = true_values.shape[1] #int(prediction.shape[1] / 2)
    return lik_temp * np.sum(scipy.stats.norm.logpdf(true_values, prediction[:,:ind], prediction[:,ind:]))


def MatrixMultiplication(x1,x2):
    if x1.shape[1] == x2.shape[1]:
        z1 = np.einsum('nj,ij->ni', x1, x2)
    else:
        z1 = np.einsum('nj,ij->ni', x1, x2[:, 1:])
        z1 += x2[:, 0].T
    return z1

# SoftMax function
def SoftMax(z):
    # return ((np.exp(z).T)/np.sum(np.exp(z),axis=1)).T
    return scipy.special.softmax(z, axis=1)

def SoftPLus(z):
    return np.log(np.exp(z) + 1)

def RegressTransform(z):
    return z

def RegressTransformError(z, ind=None):
    if ind is None:
        ind = int(z.shape[1]/2)
    #z[:,ind:] = relu_f(z[:, ind:],0) + 0.01
    z[:, ind:] = SoftPLus(z[:, ind:])
    return z

def RunHiddenLayer(z0, w01, actFun, layer_n):
    z1 = MatrixMultiplication(z0, w01)
    if actFun:
        return actFun.eval(z1, layer_n)
    else:
        return z1

def CalcAccuracyRegression(y,lab):
    acc = np.mean( (y[:,0:lab.shape[1]]-lab)**2 )
    return acc

def CalcLabelAccuracyRegression(y, lab):
    acc = np.mean( (y[:,0:lab.shape[1]]-lab)**2, axis=0)
    return acc

def CalcAccuracy(y,lab):
    if len(y.shape) == 3: # if the posterior softmax array is used, return array of accuracies
        acc = np.array([np.sum(i==lab)/len(i) for i in np.argmax(y,axis=2)])
    else:
        prediction = np.argmax(y, axis=1)
        acc = np.sum(prediction==lab)/len(prediction)
    return acc

def CalcLabelAccuracy(y,lab):
    prediction = np.argmax(y, axis=1)
    label_accs = []
    for label in np.unique(lab):
        cat_lab = lab[lab==label]
        cat_prediction = prediction[lab==label]
        acc = np.sum(cat_prediction==cat_lab)/len(cat_prediction)
        label_accs.append(acc)
    return np.array(label_accs)

def CalcConfusionMatrix(y,lab):
    prediction = np.argmax(y, axis=1)
    y_actu = pd.Categorical(lab, categories=np.unique(lab))
    y_pred = pd.Categorical(prediction, categories=np.unique(lab))
    df_confusion = pd.crosstab(y_actu, y_pred, margins=True, rownames=['True'], colnames=['Predicted'],dropna=False)
    return df_confusion

def CalcLabelFreq(y):
    prediction = np.argmax(y, axis=1)
    f = np.zeros(y.shape[1])
    tmp = np.unique(prediction, return_counts = True)
    f[tmp[0]] = tmp[1]
    return f/len(prediction)


def SaveObject(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def RunPredict(data, weights, actFun, output_act_fun):
    # weights: list of 2D arrays
    tmp = data+0
    for i in range(len(weights)-1):
        tmp = RunHiddenLayer(tmp,weights[i],actFun, i)
    tmp = RunHiddenLayer(tmp, weights[i+1], False, i+1)
    # output
    y_predict = output_act_fun(tmp)
    return y_predict

def RunPredictInd(data, weights, ind, actFun, output_act_fun):
    # weights: list of 2D arrays
    tmp = data+0
    for i in range(len(weights)-1):
        if i ==0:
            tmp = RunHiddenLayer(tmp,weights[i]*ind,actFun, i)
        elif i < len(weights)-1:
            tmp = RunHiddenLayer(tmp,weights[i],actFun, i)
    tmp = RunHiddenLayer(tmp, weights[i+1], False, i+1)
    # output
    y_predict = output_act_fun(tmp)
    return y_predict


def RecurMeanVar(it, list_mu_var_old, list_curr_param, indx):
    min_var = 1
    [Ix, Iy] = indx
    [mu_it_old, var_it_old] = list_mu_var_old
    mu_it, var_it = mu_it_old+0, var_it_old+0
    curr_param = list_curr_param[Ix, Iy]
    it = it + 1
    mu_it[Ix, Iy] = (it - 1)/it * mu_it_old[Ix, Iy] + 1/it * curr_param
    var_it[Ix, Iy] = (it - 1)/it * var_it_old[Ix, Iy] + 1/(it - 1) * (curr_param - mu_it[Ix, Iy])**2
    return [mu_it, var_it]

def calcHPD(data, level):
    assert (0 < level < 1)
    d = list(data)
    d.sort()
    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        sys.exit('\n\nToo little data to calculate marginal parameters.')
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)):
            rk = d[k+nIn-1] - d[k]
            if rk < r :
                r = rk
                i = k
    assert 0 <= i <= i+nIn-1 < len(d)
    return (d[i], d[i+nIn-1])


def CalcTP(y,lab, threshold=0.95):
    prediction = np.argmax(y, axis=1)
    max_p = y[range(len(prediction)),prediction]
    z = np.zeros(len(prediction))
    z[max_p > threshold] = 1
    return np.sum(z[prediction == lab])/len(prediction)

def CalcFP(y,lab, threshold=0.95):
    prediction = np.argmax(y, axis=1)
    max_p = y[range(len(prediction)),prediction]
    z = np.zeros(len(prediction))
    z[max_p > threshold] = 1
    return np.sum(z[prediction != lab])/len(prediction)


def CalcTP_BF(y, y_p, lab, threshold=150):
    prediction = np.argmax(y, axis=1)
    max_p = y[range(len(prediction)), prediction]
    prior = y_p[range(len(prediction)), prediction]
    bf = (max_p / (small_number + 1 - max_p)) / (prior / (small_number + 1 - prior))
    z = np.zeros(len(prediction))
    z[bf > threshold] = 1
    return np.sum(z[prediction == lab]) / len(prediction)


def CalcFP_BF(y, y_p, lab, threshold=150):
    prediction = np.argmax(y, axis=1)
    max_p = y[range(len(prediction)), prediction]
    prior = y_p[range(len(prediction)), prediction]
    bf = (max_p / (small_number + 1 - max_p)) / (prior / (small_number + 1 - prior))
    z = np.zeros(len(prediction))
    z[bf > threshold] = 1
    return np.sum(z[prediction != lab]) / len(prediction)

def CalcAccAboveThreshold(y,lab, threshold=0.95):
    max_prob = np.max(y, axis=1)
    supported_estimate = np.where(max_prob > threshold)

    prediction = np.argmax(y, axis=1)[supported_estimate]
    max_p = y[range(len(prediction)),prediction]
    z = np.zeros(len(prediction))
    z[max_p > threshold] = 1
    res= np.sum(z[prediction == lab[supported_estimate]])/len(prediction)

    print(res)


def get_posterior_cat_prob(pred_features,
                           post_samples=None,
                           feature_index_to_shuffle=None,
                           post_summary_mode=0, # mode 0 is argmax, mode 1 is mean softmax
                           unlink_features_within_block=False,
                           actFun=None,
                           output_act_fun=None):
    if len(pred_features) ==0:
        print("Data not found.")
        return 0
    else:
        n_features = pred_features.shape[1]
    predict_features = pred_features.copy()
    # shuffle features if index is provided
    if feature_index_to_shuffle: # shuffle the feature values for the given feature between all instances
        if unlink_features_within_block and type(feature_index_to_shuffle)==list:
            for feature_index in feature_index_to_shuffle: # shuffle each column by it's own random indices
                predict_features[:,feature_index] = np.random.permutation(predict_features[:,feature_index])
        else:
            predict_features[:,feature_index_to_shuffle] = np.random.permutation(predict_features[:,feature_index_to_shuffle])
    # load posterior weights
    post_weights = [post_samples[i]['weights'] for i in range(len(post_samples))]
    post_alphas = [post_samples[i]['alphas'] for i in range(len(post_samples))]
    post_cat_probs = []
    for i in range(len(post_weights)):
        actFun_i = actFun
        actFun_i.reset_prm(post_alphas[i])
        pred = RunPredict(predict_features, post_weights[i], actFun=actFun_i, output_act_fun=output_act_fun)
        post_cat_probs.append(pred)
    post_softmax_probs = np.array(post_cat_probs)
    if post_summary_mode == 0: # use argmax for each posterior sample
        class_call_posterior = np.argmax(post_softmax_probs,axis=2).T
        n_posterior_samples,n_instances,n_classes = post_softmax_probs.shape
        posterior_prob_classes = np.zeros([n_instances,n_classes])
        classes_and_counts = [[np.unique(i,return_counts=True)[0],np.unique(i,return_counts=True)[1]] for i in class_call_posterior]
        for i,class_count in enumerate(classes_and_counts):
            for j,class_index in enumerate(class_count[0]):
                posterior_prob_classes[i,class_index] = class_count[1][j]
        posterior_prob_classes = posterior_prob_classes/n_posterior_samples
    elif post_summary_mode == 1: # use mean of softmax across posterior samples
        posterior_prob_classes = np.mean(post_softmax_probs, axis=0)
    elif post_summary_mode == 2: # resample classification based on softmax/categorical probabilities (posterior predictive)
        res = sample_from_categorical(posterior_weights=post_softmax_probs)
        posterior_prob_classes = res['predictions']
    
    return(post_softmax_probs,posterior_prob_classes)

        # if summary_mode == 0: # use argmax for each posterior sample
        #     pred = np.argmax(pred, axis=1)



def predictBNN(predict_features,
               pickle_file,
               test_labels=[],
               instance_id=[],
               pickle_file_prior=0,
               target_acc = None,
               post_cutoff = None, # this is for restricting predictions to only instances exceeding this threshold
               threshold=0.95, # this is to determine TP and FP
               bf=150,
               post_summary_mode=0,
               fname="",
               wd="",
               verbose=1):

    bnn_obj,mcmc_obj,logger_obj = load_obj(pickle_file)
    post_samples = logger_obj._post_weight_samples
    actFun = bnn_obj._act_fun
    output_act_fun = bnn_obj._output_act_fun
    out_name = os.path.splitext(pickle_file)[0]
    out_name = os.path.basename(out_name)
    if wd != "":
        predictions_outdir = wd
    else:
        predictions_outdir = os.path.dirname(pickle_file)

    post_softmax_probs,post_prob_predictions = get_posterior_cat_prob(predict_features,
                                                                      post_samples,
                                                                      post_summary_mode=post_summary_mode,
                                                                      actFun=actFun,
                                                                      output_act_fun=output_act_fun)

    if fname != "":
        fname = fname + "_"
    out_file_post_pr = os.path.join(predictions_outdir, fname + out_name + '_pred_pr.npy')
    out_file_mean_pr = os.path.join(predictions_outdir, fname + out_name + '_pred_mean_pr.txt')

    if len(test_labels) > 0:
        #CalcAccAboveThreshold(post_prob_predictions, test_labels, threshold=0.95)
        accuracy = CalcAccuracy(post_prob_predictions, test_labels)
        TPrate = CalcTP(post_prob_predictions, test_labels, threshold=threshold)
        FPrate = CalcFP(post_prob_predictions, test_labels, threshold=threshold)
        mean_accuracy = np.mean(accuracy)
        cm = CalcConfusionMatrix(post_prob_predictions, test_labels)
        cm_out = cm.values.astype(int)
        if verbose:
            print("Accuracy:", mean_accuracy)
            print("True positive rate:", np.mean(TPrate))
            print("False positive rate:", np.mean(FPrate))
            print("Confusion matrix:\n", cm)
        out_file_acc = os.path.join(predictions_outdir, fname + out_name + '_accuracy.txt')
        with open(out_file_acc,'w') as outf:
            outf.writelines("Mean accuracy: %s (TP: %s; FP: %s)" % (mean_accuracy, TPrate, FPrate))
    else:
        mean_accuracy = np.nan
        cm_out = np.nan

    if pickle_file_prior:
        prior_samples = load_obj(pickle_file_prior)
        prior_weights = [prior_samples[i]['weights'] for i in range(len(prior_samples))]
        prior_alphas = [prior_samples[i]['alphas'] for i in range(len(prior_samples))]
        prior_predictions = []
        for i in range(len(prior_weights)):
            actFun_i = actFun
            actFun_i.reset_prm(prior_alphas[i])
            pred = RunPredict(predict_features, prior_weights[i], actFun=actFun_i)
            prior_predictions.append(pred)
    
        prior_predictions = np.array(prior_predictions)
        prior_prob_predictions = np.mean(prior_predictions, axis=0)
    
        TPrate = CalcTP_BF(post_prob_predictions, prior_prob_predictions, test_labels, threshold=bf)
        FPrate = CalcFP_BF(post_prob_predictions, prior_prob_predictions, test_labels, threshold=bf)
        if verbose:
            print("True positive rate (BF):", np.mean(TPrate))
            print("False positive rate (BF):", np.mean(FPrate))

    if target_acc or post_cutoff: # only to be used in prediction mode, not to get test acc during training
        if target_acc:
            posterior_threshold = get_posterior_threshold(pickle_file,target_acc,post_summary_mode)
        elif post_cutoff:
            posterior_threshold = post_cutoff
        high_pp_indices = np.where(np.max(post_prob_predictions, axis=1) > posterior_threshold)[0]
        post_prob_predictions = turn_low_pp_instances_to_nan(post_prob_predictions,high_pp_indices)
        post_softmax_probs = np.array([turn_low_pp_instances_to_nan(i,high_pp_indices) for i in post_softmax_probs])    

    if len(instance_id):
        post_prob_predictions_id = np.hstack((instance_id.reshape(len(instance_id), 1),
                                              np.round(post_prob_predictions,4).astype(str)))
        np.savetxt(out_file_mean_pr, post_prob_predictions_id, fmt='%s',delimiter='\t')
    else:
        np.savetxt(out_file_mean_pr, post_prob_predictions, fmt='%.3f')
    # print the arrays to file
    np.save(out_file_post_pr, post_softmax_probs)
    if verbose:
        print("Predictions saved in files:")
        print('   ', out_file_post_pr)
        print('   ', out_file_mean_pr,"\n")
    return {'post_prob_predictions': post_prob_predictions, 'mean_accuracy': mean_accuracy, 'confusion_matrix': cm_out}


def feature_importance(input_features,
                       weights_pkl=None,
                       weights_posterior=None,
                       true_labels=[],
                       fname_stem='',
                       feature_names=[],
                       verbose=False,
                       post_summary_mode=0,
                       n_permutations=100,
                       feature_blocks=dict(),
                       write_to_file=True, 
                       predictions_outdir='',
                       unlink_features_within_block=True,
                       actFun=None,
                       output_act_fun=None):

    features = input_features.copy()
    feature_indices = np.arange(features.shape[1])
    # if no names are provided, name them by index
    if len(feature_names) == 0:
        feature_names = feature_indices.astype(str)
    if type(feature_blocks) is dict:
        if len(feature_blocks.keys()) > 0:
            selected_features = []
            feature_block_names = []
            for block_name, block_indices in feature_blocks.items():
                selected_features.append(block_indices)
                feature_block_names.append(block_name)
        else:
            selected_features = [[i] for i in feature_indices]
            feature_block_names = [i for i in feature_names]
    else:
        # if a list of lists is provided
        selected_features = feature_blocks
        feature_block_names = ['block_' + str(i) for i in range(len(feature_blocks))]

    # get accuracy with all features
    if weights_pkl:
        bnn_obj,mcmc_obj,logger_obj = load_obj(weights_pkl)
        weights_posterior = logger_obj._post_weight_samples
        actFun = bnn_obj._act_fun
        output_act_fun = bnn_obj._output_act_fun
    post_softmax_probs,post_prob_predictions = get_posterior_cat_prob(input_features, weights_posterior,
                                                                      post_summary_mode=post_summary_mode,
                                                                      actFun=actFun,
                                                                      output_act_fun=output_act_fun)
    ref_accuracy = CalcAccuracy(post_prob_predictions, true_labels)
    if verbose:
        print("Reference accuracy (mean):", np.mean(ref_accuracy))
    # go through features and shuffle one at a time
    accuracies_wo_feature = []
    for block_id,feature_block in enumerate(selected_features):
        if verbose:
            print('Processing feature block %i',block_id+1)
        n_accuracies = []
        for _ in np.arange(n_permutations):
            post_softmax_probs,post_prob_predictions = get_posterior_cat_prob(input_features, weights_posterior,
                                                                              feature_index_to_shuffle=feature_block,
                                                                              post_summary_mode=post_summary_mode,
                                                                              unlink_features_within_block=unlink_features_within_block,
                                                                              actFun=actFun,
                                                                              output_act_fun=output_act_fun)
            accuracy = CalcAccuracy(post_prob_predictions, true_labels)
            n_accuracies.append(accuracy)
        accuracies_wo_feature.append(n_accuracies)
    accuracies_wo_feature = np.array(accuracies_wo_feature)
    delta_accs = ref_accuracy-np.array(accuracies_wo_feature)    
    delta_accs_means = np.mean(delta_accs,axis=1)
    delta_accs_stds = np.std(delta_accs,axis=1)
    accuracies_wo_feature_means = np.mean(accuracies_wo_feature,axis=1)
    accuracies_wo_feature_stds = np.std(accuracies_wo_feature,axis=1)
    feature_importance_df = pd.DataFrame(np.array([np.arange(0,len(selected_features)),feature_block_names,
                                                   delta_accs_means,delta_accs_stds,
                                                   accuracies_wo_feature_means,accuracies_wo_feature_stds]).T,
                                         columns=['feature_block_index','feature_name','delta_acc_mean','delta_acc_std',
                                                  'acc_with_feature_randomized_mean','acc_with_feature_randomized_std'])
    feature_importance_df.iloc[:,2:] = feature_importance_df.iloc[:,2:].astype(float)
    feature_importance_df_sorted = feature_importance_df.sort_values('delta_acc_mean',ascending=False)
    # format the last two columns as numeric for applyign float printing formatting options
    feature_importance_df_sorted['delta_acc_mean'] = pd.to_numeric(feature_importance_df_sorted['delta_acc_mean'])
    feature_importance_df_sorted['acc_with_feature_randomized_mean'] = pd.to_numeric(feature_importance_df_sorted['acc_with_feature_randomized_mean'])

    if write_to_file:
        # define outfile name
        if predictions_outdir == "":
            predictions_outdir = os.path.dirname(weights_pkl)
        if not os.path.exists(predictions_outdir) and predictions_outdir != "":
            os.makedirs(predictions_outdir)
        if fname_stem != "":
            fname_stem = fname_stem + "_"
        feature_importance_df_filename = os.path.join(predictions_outdir, fname_stem + 'feature_importance.txt')
        feature_importance_df_sorted.to_csv(feature_importance_df_filename,sep='\t',index=False,header=True,float_format='%.6f')
        print("Output saved in: %s" % feature_importance_df_filename)
    return feature_importance_df_sorted

    

def get_weights_from_tensorflow_model(model_dir):
    try:
        import tensorflow as tf
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable tf compilation warning
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid error about multiple copies of the OpenMP runtime
        except:
            pass
    except:
        sys.exit("The required Tensorflow library not found.")
    model = tf.keras.models.load_model(model_dir)
    n_nodes_list = []
    init_weights = []
    bias_node_weights = []
    for layer in model.layers:
        #layer_name = layer.weights[0].name 
        layer_shape = np.array(layer.weights[0].shape)
        weights = layer.weights[0].numpy().T
        n_nodes_list.append(layer_shape[1])
        init_weights.append(weights)
        if len(layer.weights) == 2: #bias node layer
            bias_node = layer.weights[1].numpy()
            bias_node_weights.append(bias_node)
    return([n_nodes_list[:-1],init_weights,bias_node_weights])


def get_accuracy_threshold(probs, labels, threshold=0.75):
    indx = np.where(np.max(probs, axis=1)>threshold)[0]
    res_supported = probs[indx,:]
    labels_supported = labels[indx]
    pred = np.argmax(res_supported, axis=1)
    accuracy = len(pred[pred == labels_supported])/len(pred)
    # print(accuracy)
    # print(CalcAccuracy(res_supported,labels_supported))
    dropped_frequency = len(pred)/len(labels)
    cm = CalcConfusionMatrix(res_supported, labels_supported)
    return {'predictions': pred, 'accuracy': accuracy, 'retained_samples': dropped_frequency, 'confusion_matrix': cm}


def get_posterior_threshold(pkl_file,target_acc=0.9,post_summary_mode=1,output_file=None):
    # determine the posterior threshold based on given target accuracy
    bnn_obj, mcmc_obj, logger_obj = load_obj(pkl_file)
    post_pr_test = predictBNN(bnn_obj._test_data,
                              pickle_file=pkl_file,
                              test_labels=bnn_obj._test_labels,
                              post_summary_mode=post_summary_mode,
                              verbose=0)

    # CALC TRADEOFFS
    res = post_pr_test['post_prob_predictions']
    labels=bnn_obj._test_labels
    tbl_results = []
    for i in np.linspace(0.01, 0.99, 99):
        try:
            scores = get_accuracy_threshold(res, labels, threshold=i)
            tbl_results.append([i, scores['accuracy'], scores['retained_samples']])
        except:
            pass
    tbl_results = np.array(tbl_results)
    if output_file is not None:
        df = pd.DataFrame(tbl_results, columns=['Threshold', 'Accuracy', 'Retained_data'])
        df = np.round(df, 3)
        df.to_csv(path_or_buf=output_file, sep='\t', index=False, header=True)
    try:
        indx = np.min(np.where(np.round(tbl_results[:,1],2) >= target_acc))
    except ValueError:
        sys.exit('Target accuracy can not be reached. Please set threshold lower or try different post_summary_mode.')
    selected_row = tbl_results[indx,:]
    print("Selected threshold: PP =", np.round(selected_row[0], 3), "yielding test accuracy ~ %s" % (target_acc))
    print("Retained instances above threshold:", np.round(selected_row[2], 3))
    return selected_row


def turn_low_pp_instances_to_nan(pred,high_pp_indices):            
    pred_temp = np.zeros(pred.shape)
    pred_temp[:] = np.nan
    pred_temp[high_pp_indices] = pred[high_pp_indices]
    pred = pred_temp
    return pred


def sample_from_categorical(posterior_weights=None, post_prob_file=None, verbose=False):
    if posterior_weights is not None:
        pass
    elif post_prob_file:
        posterior_weights = np.load(post_prob_file)
    else:
        print("Input pickle file or posterior weights required.")
    n_post_samples = posterior_weights.shape[0]
    n_instances = posterior_weights.shape[1]
    n_classes = posterior_weights.shape[2]

    res = np.zeros((n_instances, n_post_samples))
    point_estimates = np.zeros((n_instances, n_classes))
    for instance_j in range(posterior_weights.shape[1]):
        if instance_j % 1000 == 0 and verbose is True:
            print(instance_j)
        post_sample = posterior_weights[:, instance_j, :]
        p = np.cumsum(post_sample, axis=1)
        r = np.random.random(len(p))
        q = p - r.reshape(len(r), 1)
        q[q < 0] = 1  # arbitrarily large number
        classification = np.argmin(q, axis=1)
        res[instance_j, :] = classification
        # mode (point estimate)
        counts = np.bincount(classification, minlength=n_classes)
        point_estimates[instance_j, :] = counts / np.sum(counts)
    
    class_counts = np.zeros((n_post_samples, n_classes))
    for i in range(res.shape[1]):
        class_counts[i] = np.bincount(res[:, i].astype(int), minlength=n_classes)
    
    return {'predictions': point_estimates, 'class_counts': class_counts, 'post_predictions': res}

def get_posterior_est(pkl_file):
    bnn_obj, mcmc_obj, logger_obj = load_obj(pkl_file)
    post_samples = logger_obj._post_weight_samples

    # load posterior weights
    post_weights = [post_samples[i]['weights'] for i in range(len(post_samples))]
    post_alphas = [post_samples[i]['alphas'] for i in range(len(post_samples))]
    if 'error_prm' in post_samples[0]:
        post_error = [post_samples[i]['error_prm'] for i in range(len(post_samples))]
    else:
        post_error = []
    actFun = bnn_obj._act_fun
    output_act_fun = bnn_obj._output_act_fun

    post_est = []
    post_est_test = []
    for i in range(len(post_weights)):
        actFun_i = actFun
        actFun_i.reset_prm(post_alphas[i])
        pred = RunPredict(bnn_obj._data, post_weights[i], actFun=actFun_i, output_act_fun=output_act_fun)
        post_est.append(pred)
        pred_test = RunPredict(bnn_obj._test_data, post_weights[i], actFun=actFun_i, output_act_fun=output_act_fun)
        post_est_test.append(pred_test)

    post_est = np.array(post_est)
    prm_mean = np.mean(post_est,axis=0)[:,:]
    post_est_test = np.array(post_est_test)
    prm_mean_test = np.mean(post_est_test,axis=0)[:,:]
    return {'prm_mean': prm_mean,
            'post_est': post_est,
            'prm_mean_test': prm_mean_test,
            'post_est_test': post_est_test,
            'error_prm': post_error
        }
