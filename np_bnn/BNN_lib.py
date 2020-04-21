import numpy as np
import scipy.stats
import scipy.special
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
import pickle

# likelihood function
def calc_likelihood(prediction, labels, sample_id):
	return np.sum(np.log(prediction[sample_id, labels]))


# ReLU function
def ReLU(zi):
	z = np.copy(zi)
	z[z<0] = 0
	return z

def MatrixMultiplication(x1,x2):
	z1 = np.einsum('nj,ij->ni', x1, x2)
	# same as:
	# for i in range(n_samples):
	# 	print(np.einsum('j,ij->i', x[i], w_in_l1))
	return z1

# SoftMax function
def SoftMax(z):
	# return ((np.exp(z).T)/np.sum(np.exp(z),axis=1)).T
	return scipy.special.softmax(z, axis=1)


def RunHiddenLayer(z0,w01):
	z1 = MatrixMultiplication(z0, w01)
	z2 = ReLU(z1)
	return z2


def UpdateNormal(i, d=0.01, n=1, Mb=100, mb= -100):
	Ix = np.random.randint(0, i.shape[0],n) # faster than np.random.choice
	Iy = np.random.randint(0, i.shape[1],n)
	z = np.zeros(i.shape) + i
	z[Ix,Iy] = z[Ix,Iy] + np.random.normal(0, d[Ix,Iy], n)
	z[z > Mb] = Mb- (z[z>Mb]-Mb)
	z[z < mb] = mb- (z[z<mb]-mb)
	return z, (Ix, Iy)

def UpdateUniform(i, d=0.1, n=1, Mb=100, mb= -100):
	Ix = np.random.randint(0, i.shape[0],n) # faster than np.random.choice
	Iy = np.random.randint(0, i.shape[1],n)
	z = np.zeros(i.shape) + i
	z[Ix,Iy] = z[Ix,Iy] + np.random.uniform(-d[Ix,Iy], d[Ix,Iy], n)
	z[z > Mb] = Mb- (z[z>Mb]-Mb)
	z[z < mb] = mb- (z[z<mb]-mb)
	return z, (Ix, Iy)



def UpdateBinomial(ind,update_f,shape_out):
	return np.abs(ind - np.random.binomial(1, np.random.random() * update_f, shape_out))

def CalcAccuracy(y,lab):
	prediction = np.argmax(y, axis=1)
	return len(prediction[prediction==lab])/len(prediction)

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


def RunPredict(data, weights):
	# weights: list of 2D arrays
	tmp = data+0
	for i in range(len(weights)):
		tmp = RunHiddenLayer(tmp,weights[i])
	# output
	y_predict = SoftMax(tmp)
	return y_predict

def RunPredictInd(data, weights, ind):
	# weights: list of 2D arrays
	tmp = data+0
	for i in range(len(weights)):
		if i ==0:
			tmp = RunHiddenLayer(tmp,weights[i]*ind)
		else:
			tmp = RunHiddenLayer(tmp,weights[i])
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

