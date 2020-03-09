import gpflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import BNQD
import os
import warnings
import importlib
import scipy.stats as stats
import bisect
import abc

warnings.simplefilter('ignore')
importlib.reload(BNQD)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# plt.style.use('seaborn-dark-palette') # plt.style.use('ggplot')
np.random.seed(1)


class GPRegressionModel():
    """
    Abstract class for continuous and discontinuous models
    """

    def __init__(self, X, Y, kernel, likelihood=gpflow.likelihoods.Gaussian(), mean_function=None):
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.kernel = kernel
        self.likelihood = likelihood
        self.mean_function = mean_function
        self.BICscore = None
        self.isOptimized = False

    def train(self, optim, max_iter, verbose=False):
        raise NotImplementedError

    def predict(self, x_test, n_samples):
        raise NotImplementedError

    def get_log_marginal_likelihood(self):
        if not self.isOptimized:
            print('Parameters have not been optimized; training now')
            self.train()

        if self.BICscore is None:
            k = len(self.m.parameters) # parameters are represented as tuple, for documentation see gpflow.Module
            L = self.m.log_likelihood()
            BIC = L - k / 2 * np.log(self.n)
            # BIC = -2 * L + np.log(self.n) * k # toDo: is this formula from wiki equivalent? https://en.wikipedia.org/wiki/Bayesian_information_criterion, https://github.com/SheffieldML/GPy/issues/298
            self.BICscore = BIC
        return self.BICscore
        #raise NotImplementedError

    def plot(self, x_test, mean, var, samples, b = None):
        raise NotImplementedError


class ContinuousModel(GPRegressionModel):

    def __init__(self, X, Y, kernel, likelihood=gpflow.likelihoods.Gaussian(), mean_function=None, noise_variance=1.0):
        super().__init__(X, Y, kernel, likelihood, mean_function)
        self.m = gpflow.models.GPR(data=(X, Y), kernel=kernel, mean_function=self.mean_function, noise_variance = noise_variance)


    def train(self, optim = gpflow.optimizers.Scipy(), max_iter=1000, verbose=True):
        # Minimization
        def objective_closure():
            return - self.m.log_marginal_likelihood()

        opt_logs = optim.minimize(objective_closure,
                                  self.m.trainable_variables,
                                  options=dict(maxiter=max_iter))
        if verbose:
            gpflow.utilities.print_summary(self.m)
        self.isOptimized = True


    def predict(self, x_test, n_samples = 5):
        # predict mean and variance of latent GP at test points
        mean, var = self.m.predict_f(x_test)

        # generate n samples from posterior
        samples = self.m.predict_f_samples(x_test, n_samples)
        return mean, var, samples


    def get_log_marginal_likelihood(self):
        super().get_log_marginal_likelihood()

    def plot(self, x_test, mean, var, samples, true_func = None, b = None):
        plt.figure(figsize=(12,8))
        plt.plot(self.X, self.Y, 'kx', mew=2)
        plt.plot(x_test, mean, 'C0', lw=2)
        plt.fill_between(x_test[:, 0],
                        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                        color='C0', alpha=0.2)

        plt.plot(x_test, samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
        if mu is not None:
            plt.plot(x_test, true_func, label='True function', linewidth=2.0, color='black')
        if b is not None:
            plt.axvline(x=b, color='black', linestyle=':')
        plt.set_xlabel = "x"
        plt.set_ylabel = 'y'
        name = self.kernel.__class__.__name__
        if name == 'Sum':
            name = self.kernel.kernels[0].name + ' and ' + self.kernel.kernels[1].name
        plt.title("Posterior prediction using the " + name + " kernel")
        plt.show()


class DiscontinuousModel(GPRegressionModel):

    def __init__(self, X, Y, kernel, label_func, likelihood=gpflow.likelihoods.Gaussian(), mean_function=None,
                 noise_variance=1.0):
        super().__init__(X, Y, kernel, likelihood, mean_function)
        # Label data using provided label function
        self.label_func = label_func
        labels1 = label_func(X)
        labels2 = np.logical_not(labels1)
        self.x1, self.x2 = X[labels1,], X[labels2,]
        self.y1, self.y2 = Y[labels1,], Y[labels2,]

        # Create two continuous models
        m1 = ContinuousModel(self.x1, self.y2, kernel, likelihood, mean_function, noise_variance)
        m2 = ContinuousModel(self.x2, self.y2, kernel, likelihood, mean_function, noise_variance)
        self.submodels = [m1, m2]

    def train(self, optim = gpflow.optimizers.Scipy(), max_iter=1000, verbose=True, share_hyp = False):
        param_list = list()
        for submodel in self.submodels:
            submodel.train(optim, max_iter, verbose)
            param_list.append(submodel.m.trainable_parameters)

        if share_hyp:
            raise NotImplementedError
        self.isOptimized = True

    def predict(self, x_test, n_samples = 5):
        return (self.submodels[0].predict(x_test, n_samples),
                self.submodels[1].predict(x_test, n_samples))


    def plot(self):
        raise NotImplementedError




# Testing samples
#------------------------------
# Create data
N = 10
b = 0.5
n_samples_prior = 4
X = np.random.rand(N,1)
def mean_function(x):
    return np.sin(12*x) + 0.66*np.cos(25*x) + 3
Y = mean_function(X) + np.random.randn(N,1)*0.1
# test points for prediction
xlim = [2, 2]
xx = np.linspace(xlim[0], xlim[1], 100).reshape(100, 1)  # (N, D)
mu = mean_function(xx)

# Kernels
linear = gpflow.kernels.Linear()
rbf = gpflow.kernels.RBF()
matern12 = gpflow.kernels.Matern12()
matern32 = gpflow.kernels.Matern32()
polynomial = gpflow.kernels.Polynomial()
exp = gpflow.kernels.Exponential()
arccos = gpflow.kernels.ArcCosine()
periodic = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
cosine = gpflow.kernels.Cosine(lengthscale = 0.12) + gpflow.kernels.Constant()
periodic_matern52 = gpflow.kernels.Periodic(gpflow.kernels.Matern52())

kernels = [matern12, rbf, periodic, cosine]

# Optimizer for minimization
optim = gpflow.optimizers.Scipy()
max_iter = 1000
continuous = ContinuousModel(X, Y, linear)
model = continuous.m
continuous.train()
mean, var, samples =  continuous.predict(xx, n_samples_prior)
continuous.plot(xx, mean, var, samples, mu, b = 0.5)

# label_func = lambda x: x < b
# discontinuous = DiscontinuousModel(X,Y, linear, label_func)
# discontinuous.train()
# (mean1, var1, samples1), (mean2, var2, sampels2) = discontinuous.predict(xx, n_samples_prior)

