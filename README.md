# npBNN - Bayesian Neural Networks
The npBNN package is a Python implementation of Bayesian neural networks for classification, using the Numpy and Scipy libraries. The program is used in our [arXiv paper](https://arxiv.org/abs/2005.04987).

The example file [`npBNN.py`](https://github.com/dsilvestro/npBNN/blob/master/npBNN.py) shows how to set up a BNN model, train it, and use it to make predictions.
The npBNN package implements Markov Chain Monte Carlo (MCMC) to estimated the model parmaeters. A parallelized version using Metropolis Coupled MCMC (or MC3) is also available: [`npBNNMC3.py`](https://github.com/dsilvestro/npBNN/blob/master/npBNNMC3.py).

