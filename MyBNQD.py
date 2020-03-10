import abc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gf
import warnings

from typing import Optional, Tuple

from gpflow import optimizers
from gpflow.kernels import Kernel, Constant, Linear, Exponential, SquaredExponential
from gpflow.models import GPModel, GPR
from gpflow.likelihoods import Likelihood, Gaussian
from gpflow.models.model import DataPoint, MeanAndVariance
from gpflow.mean_functions import MeanFunction


class BNQDModel(GPModel):

    def __init__(self, X, Y, kernel: Kernel, likelihood: Likelihood, mean_function: Optional[MeanFunction] = None,
                 num_latent: int = 1):
        # What is a latent variable? (num_latent)
        super().__init__(kernel, likelihood, mean_function, num_latent)
        self.X = X
        self.Y = Y
        self.isTrained = False
        self.BICScore = None

    @abc.abstractmethod
    def train(self, optimizer=optimizers.Scipy(), verbose=True):
        # TODO: Make an optimizer wrapper that allows for easy specification of parameters for different optimizers
        raise NotImplementedError

    def log_likelihood(self, *args, **kwargs) -> tf.Tensor:
        raise NotImplementedError

    def predict_f(self, predict_at: DataPoint, full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        raise NotImplementedError

    @abc.abstractmethod
    def plot(self, n_samples: int = 100):
        raise NotImplementedError


class ContinuousModel(BNQDModel):

    def __init__(self, X, Y, kernel: Kernel, likelihood: Likelihood, mean_function: Optional[MeanFunction] = None,
                 num_latent: int = 1):
        self.MAX_OPT_ITER = 100

        super().__init__(X, Y, kernel, likelihood, mean_function, num_latent)
        data = (self.X[:, None], self.Y[:, None])
        self.m = GPR(data, self.kernel)

    def train(self, optimizer=optimizers.Scipy(), verbose=True):
        closure = lambda: -self.m.log_marginal_likelihood()
        optimizer.minimize(closure, self.m.trainable_parameters, options=(dict(maxiter=self.MAX_OPT_ITER)))
        if verbose:
            gf.utilities.print_summary(self.m)
        self.isTrained = True

    def log_likelihood(self, *args, **kwargs) -> tf.Tensor:
        if not self.isTrained:
            msg = "`The model hasn't been trained yet." \
                  " It will be trained using the default optimizer." \
                  " If you wish to use a different optimizer, please use the ContinuousModel.train(optimizer) function`"
            warnings.warn(msg, category=UserWarning)
            self.train()

        if not self.BICScore:
            k = len(self.m.parameters)  # parameters are represented as tuple, for documentation see gpf.Module
            L = self.m.log_likelihood()
            BIC = L - k / 2 * np.log(self.N)
            self.BICscore = BIC
        return self.BICscore

    def predict_f(self, predict_at: DataPoint, full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        return self.m.predict_f(predict_at, full_cov, full_output_cov)

    def plot(self, n_samples: int = 100):
        x_samples = np.linspace(min(self.X), max(self.X), n_samples)
        mean, var = self.predict_f(x_samples[:, None])

        plt.plot(x_samples, mean, c='green', label='$M_c$')
        plt.fill_between(x_samples, mean[:, 0] - 1.96 * np.sqrt(var[:, 0]), mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                         color='green', alpha=0.2)


################
## TEST STUFF ##
################

b = 0.0
n = 100
x = np.linspace(-3, 3, n)
f = 0.8*np.sin(x) + 0.2*x**2 + 0.2*np.cos(x/4) + 1.0*(x>b)
sigma = np.sqrt(1)
y = np.random.normal(f, sigma, size=n)

plt.figure()

if True:  # disables / enables plotting of the data points and generative function
    plt.plot(x[x<=b], f[x<=b], label='True f', c='k')
    plt.plot(x[x>=b], f[x>=b], c='k')
    plt.axvline(x=b, linestyle='--', c='k')
    plt.plot(x, y, linestyle='none', marker='x', color='k', label='obs')

cm = ContinuousModel(x, y, Exponential(), Gaussian())
cm.train()
cm.plot()
plt.show()
