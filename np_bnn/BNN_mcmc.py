import numpy as np
import scipy.special
import scipy.stats

np.set_printoptions(suppress=True, precision=3)
small_number = 1e-10
import random
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

def init_weight_prm(n_nodes, n_features, size_output, init_std=0.1, bias_node=0):
    bn, bn2, bn3 = 0, 0, 0
    if bias_node >= 1:
        bn = 1
    if bias_node >= 2:
        bn2 = 1
    n_layers = len(n_nodes) + 1
    if bias_node == 3 or bias_node == -1:
        bn3 = 1
    # 1st layer
    w_layers = [np.random.normal(0, init_std, (n_nodes[0], n_features + bn))]
    # add hidden layers
    for i in range(1, n_layers - 1):
        w_layers.append(np.random.normal(0, init_std, (n_nodes[i], n_nodes[i - 1] + bn2)))
    # last layer
    w_layers.append(np.random.normal(0, init_std, (size_output, n_nodes[-1] + bn3)))
    return w_layers

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
    z[z > Mb] = Mb - (z[z>Mb] - Mb)
    z[z < mb] = mb + (mb - z[z<mb])
    return z, (Ix, Iy), hastings

def UpdateNormal1D(i, d=0.01, n=1, Mb=100, mb= -100, rs=0):
    i = np.array(i)
    if not rs:
        rseed = random.randint(1000, 9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    Ix = rs.randint(0, len(i),n) # faster than np.random.choice
    z = np.zeros(i.shape) + i
    z[Ix] = z[Ix] + rs.normal(0, d, n)
    z[z > Mb] = Mb - (z[z>Mb] - Mb)
    z[z < mb] = mb + (mb - z[z<mb])
    hastings = 0
    return z, Ix, hastings

def UpdateNormal(i, d=0.01, n=1, Mb=100, mb= -100, rs=0):
    i = np.array(i)
    if not rs:
        rseed = random.randint(1000, 9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    Ix = rs.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = rs.randint(0, i.shape[1],n)
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = z[Ix,Iy] + rs.normal(0, d[Ix,Iy], n)
    z[z > Mb] = Mb - (z[z>Mb] - Mb)
    z[z < mb] = mb + (mb - z[z<mb])
    hastings = 0
    return z, (Ix, Iy), hastings

def UpdateNormalNormalized(i, d=0.01, n=1, Mb=100, mb= -100, rs=0):
    i = np.array(i)
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
    i = np.array(i)
    Ix = np.random.randint(0, i.shape[0],n) # faster than np.random.choice
    Iy = np.random.randint(0, i.shape[1],n)
    z = np.zeros(i.shape) + i
    z[Ix,Iy] = z[Ix,Iy] + np.random.uniform(-d[Ix,Iy], d[Ix,Iy], n)
    z[z > Mb] = Mb - (z[z>Mb] - Mb)
    z[z < mb] = mb + (mb - z[z<mb])
    hastings = 0
    return z, (Ix, Iy), hastings


def UpdateBinomial(ind,update_f,shape_out):
    return np.abs(ind - np.random.binomial(1, np.random.random() * update_f, shape_out))

def multiplier_proposal_vector(q, d=1.05, f=1, rs=0):
    if not rs:
        rseed = random.randint(1000, 9999)
        rs = RandomState(MT19937(SeedSequence(rseed)))
    S = q.shape
    ff = rs.binomial(1,f,S)
    u = rs.random(S)
    l = 2 * np.log(d)
    m = np.exp(l * (u - .5))
    m[ff==0] = 1.
    new_q = q * m
    U=np.sum(np.log(m))
    return new_q, 0, U


def multiplier_proposal(i, d=1.05):
    z = i + 0
    u = np.random.random()
    l = 2 * np.log(d)
    m = np.exp(l * (u - .5))
    z = z * m
    U = np.log(m)
    return z, 0, U


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


def run_mcmc(bnn, mcmc, logger):
    while True:
        mcmc.mh_step(bnn)
        # print some stats (iteration number, likelihood, training accuracy, test accuracy
        if mcmc._current_iteration % mcmc._print_f == 0 or mcmc._current_iteration == 1:
            acceptance_rate = mcmc._acceptance_rate
            print(mcmc._current_iteration, np.round([mcmc._logLik, mcmc._accuracy, mcmc._test_accuracy, acceptance_rate],3),flush=True)
            if bnn._estimation_mode == "regression":
                print(bnn._error_prm)
        # save to file
        if mcmc._current_iteration % mcmc._sampling_f == 0:
            logger.log_sample(bnn,mcmc)
            logger.log_weights(bnn,mcmc)
        # stop MCMC after running desired number of iterations
        if mcmc._current_iteration == mcmc._n_iterations:
            break
