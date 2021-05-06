# npBNN - Bayesian Neural Networks
The np_bnn library is a Python implementation of Bayesian neural networks for classification, using the Numpy and Scipy libraries. The program is used in our [arXiv paper](https://arxiv.org/abs/2005.04987).

To install the np_bnn library you can use:

```
python -m pip install https://github.com/dsilvestro/npBNN/archive/v0.1.9.tar.gz
```
Note that you may have to use `python3` depending on which version of Python is set as default in your operating system. 


The example file [`bnn_runner.py`](https://github.com/dsilvestro/npBNN/blob/master/bnn_runner.py) shows how to set up a BNN model, train it, and use it to make predictions.
The npBNN package implements Markov Chain Monte Carlo (MCMC) to estimate the model parameters. A parallelized version using Metropolis Coupled MCMC (or MC3) is also available: [`bnn_runner_MC3.py`](https://github.com/dsilvestro/npBNN/blob/master/bnn_runner_MC3.py).
