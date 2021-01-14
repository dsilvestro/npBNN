import numpy as np
import scipy.stats
import scipy.special
import pandas as pd
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
import pickle
small_number = 1e-10
import random, sys
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from . import BNN_files
import os
try:
    import tensorflow as tf
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable tf compilation warning
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # avoid error about multiple copies of the OpenMP runtime 
    except:
        pass
except:
    print(' ')

# Activation functions
class genReLU():
    def __init__(self, prm=np.zeros(1), trainable=False):
        self._prm = prm
        self._acc_prm = prm
        self._trainable = trainable
        # if alpha < 1 and non trainable: leaky ReLU (https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
        # if trainable: parameteric ReLU (https://arxiv.org/pdf/1502.01852.pdf)
        if prm[0] == 0 and not trainable:
            self._simpleReLU = True
        else:
            self._simpleReLU = False

    def eval(self, z, layer_n):
        if self._simpleReLU:
            z[z < 0] = 0
        else:
            z[z < 0] = self._prm[layer_n] * z[z < 0]
        return z
    def reset_prm(self, prm):
        self._prm = prm

    def reset_accepted_prm(self):
        self._acc_prm = self._prm + 0


# likelihood function (Categorical)
# TODO: refactor this as a class
def calc_likelihood(prediction, labels, sample_id, class_weight=[], lik_temp=1):
    if len(class_weight):
        return lik_temp * np.sum(np.log(prediction[sample_id, labels])*class_weight[labels])
    else:
        # if lik_temp != 1:
        #     tempered_prediction = lik_temp ** prediction
        #     normalized_tempered_prediction = np.einsum('xy,x->xy', tempered_prediction, 1 / np.sum(tempered_prediction,axis=1))
        #     return np.sum(np.log(normalized_tempered_prediction[sample_id, labels]))
        # else:
        return lik_temp * np.sum(np.log(prediction[sample_id, labels]))



def MatrixMultiplication(x1,x2):
    z1 = np.einsum('nj,ij->ni', x1, x2, optimize=False)
    # same as:
    # for i in range(n_samples):
    # 	print(np.einsum('j,ij->i', x[i], w_in_l1))
    return z1

# SoftMax function
def SoftMax(z):
    # return ((np.exp(z).T)/np.sum(np.exp(z),axis=1)).T
    return scipy.special.softmax(z, axis=1)


def RunHiddenLayer(z0, w01, actFun, layer_n):
    z1 = MatrixMultiplication(z0, w01)
    if actFun:
        return actFun.eval(z1, layer_n)
    else:
        return z1

def UpdateFixedNormal(i, d=1, n=1, Mb=100, mb= -100, rs=0):
    if not rs:
        rseed = random.randint(1000, 9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    Ix = rs.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = rs.randint(0, i.shape[1],n)
    current_prm = i[Ix,Iy]
    new_prm = rs.normal(0, d[Ix,Iy], n)
    hastings = np.sum(scipy.stats.norm.logpdf(current_prm, 0, d[Ix,Iy]) - \
               scipy.stats.norm.logpdf(new_prm, 0, d[Ix,Iy]))
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = new_prm
    z[z > Mb] = Mb- (z[z>Mb]-Mb)
    z[z < mb] = mb- (z[z<mb]-mb)
    return z, (Ix, Iy), hastings

def UpdateNormal1D(i, d=0.01, n=1, Mb=100, mb= -100, rs=0):
    if not rs:
        rseed = random.randint(1000, 9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    Ix = rs.randint(0, len(i),n) # faster than np.random.choice
    z = np.zeros(i.shape) + i
    z[Ix] = z[Ix] + rs.normal(0, d, n)
    z[z > Mb] = Mb- (z[z>Mb]-Mb)
    z[z < mb] = mb- (z[z<mb]-mb)
    hastings = 0
    return z, Ix, hastings

def UpdateNormal(i, d=0.01, n=1, Mb=100, mb= -100, rs=0):
    if not rs:
        rseed = random.randint(1000, 9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    Ix = rs.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = rs.randint(0, i.shape[1],n)
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = z[Ix,Iy] + rs.normal(0, d[Ix,Iy], n)
    z[z > Mb] = Mb- (z[z>Mb]-Mb)
    z[z < mb] = mb- (z[z<mb]-mb)
    hastings = 0
    return z, (Ix, Iy), hastings

def UpdateNormalNormalized(i, d=0.01, n=1, Mb=100, mb= -100, rs=0):
    if not rs:
        rseed = random.randint(1000, 9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    Ix = rs.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = rs.randint(0, i.shape[1],n)
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = z[Ix,Iy] + rs.normal(0, d[Ix,Iy], n)
    z = z/np.sum(z)
    hastings = 0
    return z, (Ix, Iy), hastings



def UpdateUniform(i, d=0.1, n=1, Mb=100, mb= -100):
    Ix = np.random.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = np.random.randint(0, i.shape[1],n)
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = z[Ix,Iy] + np.random.uniform(-d[Ix,Iy], d[Ix,Iy], n)
    z[z > Mb] = Mb- (z[z>Mb]-Mb)
    z[z < mb] = mb- (z[z<mb]-mb)
    hastings = 0
    return z, (Ix, Iy), hastings



def UpdateBinomial(ind,update_f,shape_out):
    return np.abs(ind - np.random.binomial(1, np.random.random() * update_f, shape_out))

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
    y_actu = pd.Series(lab, name='True')
    y_pred = pd.Series(prediction, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, margins=True, rownames=['True'], colnames=['Predicted'])
    return df_confusion

def CalcLabelFreq(y):
    prediction = np.argmax(y, axis=1)
    f = np.zeros(y.shape[1])
    tmp = np.unique(prediction, return_counts = True)
    f[tmp[0]] = tmp[1]
    return f/len(prediction)

def GibbsSampleNormStdGammaVector(x,a=2,b=0.1,mu=0):
    Gamma_a = a + len(x)/2.
    Gamma_b = b + np.sum((x-mu)**2)/2.
    tau = np.random.gamma(Gamma_a, scale=1./Gamma_b)
    return 1/np.sqrt(tau)


def GibbsSampleNormStdGamma2D(x,a=1,b=0.1,mu=0):
    Gamma_a = a + (x.shape[0])/2. #
    Gamma_b = b + np.sum((x-mu)**2,axis=0)/2.
    tau = np.random.gamma(Gamma_a, scale=1./Gamma_b)
    return 1/np.sqrt(tau)

def GibbsSampleNormStdGammaONE(x,a=1.5,b=0.1,mu=0):
    Gamma_a = a + 1/2. # one observation for each value (1 Y for 1 s2)
    Gamma_b = b + ((x-mu)**2)/2.
    tau = np.random.gamma(Gamma_a, scale=1./Gamma_b)
    return 1/np.sqrt(tau)

def GibbsSampleGammaRateExp(sd,a,alpha_0=1.,beta_0=1.):
    # prior is on precision tau
    tau = 1./(sd**2) #np.array(tau_list)
    conjugate_a = alpha_0 + len(tau)*a
    conjugate_b = beta_0 + np.sum(tau)
    return np.random.gamma(conjugate_a,scale=1./conjugate_b)

def SaveObject(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def RunPredict(data, weights, actFun):
    # weights: list of 2D arrays
    tmp = data+0
    for i in range(len(weights)-1):
        tmp = RunHiddenLayer(tmp,weights[i],actFun, i)
    tmp = RunHiddenLayer(tmp, weights[i+1], False, i+1)
    # output
    y_predict = SoftMax(tmp)
    return y_predict

def RunPredictInd(data, weights, ind, actFun):
    # weights: list of 2D arrays
    tmp = data+0
    for i in range(len(weights)-1):
        if i ==0:
            tmp = RunHiddenLayer(tmp,weights[i]*ind,actFun, i)
        elif i < len(weights)-1:
            tmp = RunHiddenLayer(tmp,weights[i],actFun, i)
    tmp = RunHiddenLayer(tmp, weights[i+1], False, i+1)
    # output
    y_predict = SoftMax(tmp)
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
        sys.exit('\n\nToo little data to calculate marginal rates.')
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
                           pickle_file=None,
                           post_samples=None,
                           feature_index_to_shuffle=None,
                           post_summary_mode=0, # mode 0 is argmax, mode 1 is mean softmax
                           unlink_features_within_block=False):
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
    if pickle_file is not None:
        post_samples = BNN_files.load_obj(pickle_file)
    post_weights = [post_samples[i]['weights'] for i in range(len(post_samples))]
    post_alphas = [post_samples[i]['alphas'] for i in range(len(post_samples))]
    if n_features < post_weights[0][0].shape[1]:
        "add bias node"
        predict_features = np.c_[np.ones(predict_features.shape[0]), predict_features]
    post_cat_probs = []
    for i in range(len(post_weights)):
        actFun = genReLU(prm=post_alphas[i])
        pred = RunPredict(predict_features, post_weights[i], actFun=actFun)
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
    return(post_softmax_probs,posterior_prob_classes)

        # if summary_mode == 0: # use argmax for each posterior sample
        #     pred = np.argmax(pred, axis=1)



def predictBNN(predict_features, pickle_file=None, post_samples=None, test_labels=[], instance_id=[],
               pickle_file_prior=0, threshold=0.95, bf=150, fname="",post_summary_mode=0,
               wd=""):

    post_softmax_probs,post_prob_predictions = get_posterior_cat_prob(predict_features,
                                                                      pickle_file,
                                                                      post_samples,
                                                                      post_summary_mode=post_summary_mode)
    
    if pickle_file:
        predictions_outdir = os.path.dirname(pickle_file)
        out_name = os.path.splitext(pickle_file)[0]
        out_name = os.path.basename(out_name)
    else:
        predictions_outdir = wd
        out_name = ""
    if fname != "":
        fname = fname + "_"
    out_file_post_pr = os.path.join(predictions_outdir, fname + out_name + '_pred_pr.npy')
    out_file_mean_pr = os.path.join(predictions_outdir, fname + out_name + '_pred_mean_pr.txt')

    if len(test_labels) > 0:
        CalcAccAboveThreshold(post_prob_predictions, test_labels, threshold=0.95)
        accuracy = CalcAccuracy(post_prob_predictions, test_labels)
        TPrate = CalcTP(post_prob_predictions, test_labels, threshold=threshold)
        FPrate = CalcFP(post_prob_predictions, test_labels, threshold=threshold)
        mean_accuracy = np.mean(accuracy)
        print("Accuracy:", mean_accuracy)
        print("True positive rate:", np.mean(TPrate))
        print("False positive rate:", np.mean(FPrate))
        print("Confusion matrix:\n", CalcConfusionMatrix(post_prob_predictions, test_labels))
        out_file_acc = os.path.join(predictions_outdir, fname + out_name + '_accuracy.txt')
        with open(out_file_acc,'w') as outf:
            outf.writelines("Mean accuracy: %s (TP: %s; FP: %s)" % (mean_accuracy, TPrate, FPrate))
    else:
        mean_accuracy = np.nan
    if pickle_file_prior:
        prior_samples = BNN_files.load_obj(pickle_file_prior)
        prior_weights = [prior_samples[i]['weights'] for i in range(len(prior_samples))]
        prior_alphas = [prior_samples[i]['alphas'] for i in range(len(prior_samples))]
        prior_predictions = []
        for i in range(len(prior_weights)):
            actFun = BNN_lib.genReLU(prm=prior_alphas[i])
            pred = RunPredict(predict_features, prior_weights[i], actFun=actFun)
            prior_predictions.append(pred)
    
        prior_predictions = np.array(prior_predictions)
        prior_prob_predictions = np.mean(prior_predictions, axis=0)
    
        TPrate = CalcTP_BF(post_prob_predictions, prior_prob_predictions, test_labels, threshold=bf)
        FPrate = CalcFP_BF(post_prob_predictions, prior_prob_predictions, test_labels, threshold=bf)
    
        print("True positive rate (BF):", np.mean(TPrate))
        print("False positive rate (BF):", np.mean(FPrate))

    if len(instance_id):
        post_prob_predictions_id = np.hstack((instance_id.reshape(len(instance_id), 1),
                                              np.round(post_prob_predictions,4).astype(str)))
        np.savetxt(out_file_mean_pr, post_prob_predictions_id, fmt='%s',delimiter='\t')
    else:
        np.savetxt(out_file_mean_pr, post_prob_predictions, fmt='%.3f')
    # print the arrays to file
    np.save(out_file_post_pr, post_softmax_probs)
    print("Predictions saved in files:")
    print('   ', out_file_post_pr)
    print('   ', out_file_mean_pr,"\n")
    return {'post_prob_predictions': post_prob_predictions, 'mean_accuracy': mean_accuracy}


def feature_importance(input_features,
                       weights_pkl=None,
                       weights_posterior=None,
                       true_labels=[],
                       fname_stem='',
                       feature_names=[],
                       verbose=False,
                       post_summary_mode=0,
                       n_permutations=100,
                       feature_blocks=[],
                       predictions_outdir='',
                       unlink_features_within_block=False):
    features = input_features.copy()
    feature_indices = np.arange(features.shape[1])
    # if no names are provided, name them by index
    if len(feature_names) == 0:
        feature_names = feature_indices.astype(str)
    if len(feature_blocks) > 0:
        selected_features = []
        selected_feature_names = []
        for block_indices in feature_blocks:
            selected_features.append(list(np.array(feature_indices)[block_indices]))
            selected_feature_names.append(list(np.array(feature_names)[block_indices]))
    else:
        selected_features = [[i] for i in feature_indices]
        selected_feature_names = [[i] for i in feature_names]
    feature_block_names = [','.join(np.array(i).astype(str)) for i in selected_feature_names] #join the feature names into one string for each block for output df
    # get accuracy with all features
    post_softmax_probs,post_prob_predictions = get_posterior_cat_prob(input_features, weights_pkl, weights_posterior,
                                                                      post_summary_mode=post_summary_mode)
    ref_accuracy = CalcAccuracy(post_prob_predictions, true_labels)
    if verbose:
        print("Reference accuracy (mean):", np.mean(ref_accuracy))
    # go through features and shuffle one at a time
    accuracies_wo_feature = []
    for block_id,feature_block in enumerate(selected_features):
        if verbose:
            print('Processing feature block %i',block_id+1)
        n_accuracies = []
        for rep in np.arange(n_permutations):
            post_softmax_probs,post_prob_predictions = get_posterior_cat_prob(input_features, weights_pkl, weights_posterior,
                                                                              feature_index_to_shuffle=feature_block,
                                                                              post_summary_mode=post_summary_mode,
                                                                              unlink_features_within_block=unlink_features_within_block)
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
    # define outfile name
    #predictions_outdir = os.path.dirname(weights_pkl)
    if not os.path.exists(predictions_outdir):
        os.makedirs(predictions_outdir)
    if fname_stem != "":
        fname_stem = fname_stem + "_"
    feature_importance_df_filename = os.path.join(predictions_outdir, fname_stem + '_feature_importance.txt')
    # format the last two columns as numeric for applyign float printing formatting options
    feature_importance_df_sorted['delta_acc_mean'] = pd.to_numeric(feature_importance_df_sorted['delta_acc_mean'])
    feature_importance_df_sorted['acc_with_feature_randomized_mean'] = pd.to_numeric(feature_importance_df_sorted['acc_with_feature_randomized_mean'])
    feature_importance_df_sorted.to_csv(feature_importance_df_filename,sep='\t',index=False,header=True,float_format='%.6f')
    return feature_importance_df_sorted


def get_weights_from_tensorflow_model(model_dir):
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
    dropped_frequency = len(pred)/len(labels)
    cm = CalcConfusionMatrix(res_supported, labels_supported)
    return {'predictions': pred, 'accuracy': accuracy, 'retained_samples': dropped_frequency, 'confusion_matrix': cm}

def run_mcmc(bnn, mcmc, logger):
    while True:
        mcmc.mh_step(bnn)
        # print some stats (iteration number, likelihood, training accuracy, test accuracy
        if mcmc._current_iteration % mcmc._print_f == 0 or mcmc._current_iteration == 1:
            print(mcmc._current_iteration, np.round([mcmc._logLik, mcmc._accuracy, mcmc._test_accuracy],3))
        # save to file
        if mcmc._current_iteration % mcmc._sampling_f == 0:
            logger.log_sample(bnn,mcmc)
            logger.log_weights(bnn,mcmc)
        # stop MCMC after running desired number of iterations
        if mcmc._current_iteration == mcmc._n_iterations:
            break
