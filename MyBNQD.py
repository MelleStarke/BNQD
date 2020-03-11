import abc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gf
import warnings

from typing import Optional, Tuple, Union, List, Callable, Any

from numpy import ndarray

from gpflow import optimizers
from gpflow.kernels import Kernel, Constant, Linear, Exponential, SquaredExponential, Periodic, Cosine
from gpflow.models import GPModel, GPR
from gpflow.likelihoods import Likelihood, Gaussian
from gpflow.models.model import DataPoint, MeanAndVariance
from gpflow.mean_functions import MeanFunction


##################################
###### Data Type Definitions #####
##################################

# Data for the continuous model: tuple of tensors / ndarrays
ContinuousData = Tuple[Union[ndarray, tf.Tensor], Union[ndarray, tf.Tensor]]

# Data for the discontinuous model: tuple of continuous data
DiscontinuousData = Tuple[ContinuousData, ContinuousData]

# Data for the abstract BNQD model: either continuous data or discontinuous data
Data = Union[ContinuousData, DiscontinuousData]


##############################
###### Class Definitions #####
##############################

class BNQDModel(GPModel):

    def __init__(self, data: Data, kernel: Kernel, likelihood: Likelihood, mean_function: Optional[MeanFunction] = None,
                 num_latent: int = 1):
        # TODO: figure out what a latent variable is in this context (num_latent)
        super().__init__(kernel, likelihood, mean_function, num_latent)
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

    def __init__(self, data: ContinuousData, kernel: Kernel, likelihood: Likelihood,
                 mean_function: Optional[MeanFunction] = None,
                 num_latent: int = 1):
        # TODO: allow the user to set this in an elegant way
        # Maximum optimizer iterations
        self.MAX_OPT_ITER = 1000

        super().__init__(data, kernel, likelihood, mean_function, num_latent)

        # TODO: check if this data set shape stuff actually works
        if len(data[0].shape) == 1:  # Checks if the input data is of rank 1 (i.e. vector)
            # Converts the input data of shape (N) to shape (N,1), to comply with tensorflow.matmul requirements
            x, y = data
            self.data = (x[:, None], y[:, None])

        else:  # Assigns without changing the dimensions of the input data
            self.data = data

        self.N = data[0].shape[0]  # nr. of data points
        self.m = GPR(self.data, self.kernel)

    def train(self, optimizer=optimizers.Scipy(), verbose=True):
        # function that the optimizer aims to minimize
        closure = lambda: -self.m.log_marginal_likelihood()

        optimizer.minimize(closure, self.m.trainable_parameters, options=(dict(maxiter=self.MAX_OPT_ITER)))
        if verbose:
            gf.utilities.print_summary(self.m)

        self.isTrained = True
        self.BICScore = None  # Ensures the BIC score is updated when trying to access it after training the model

    def log_likelihood(self, *args, **kwargs) -> tf.Tensor:
        if not self.isTrained:
            # Prints a warning if the model hasn't been trained.
            # Then trains it with the default optimizer
            msg = "The model hasn't been trained yet." \
                  "It will be trained using the default optimizer.\n" \
                  "If you wish to use a different optimizer, please use the ContinuousModel.train(optimizer) function"
            warnings.warn(msg, category=UserWarning)
            self.train()

        if not self.BICScore:  # Means: if self.BICScore == None
            k = len(self.m.parameters)  # parameters are represented as tuple, for documentation see gpf.Module
            L = self.m.log_likelihood()
            BIC = L - k / 2 * np.log(self.N)  # BIC score is the likelihood with penalization for the nr. of parameters
            self.BICScore = BIC
        return self.BICScore

    def predict_f(self, predict_at: DataPoint, full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        return self.m.predict_f(predict_at, full_cov, full_output_cov)

    def plot(self, n_samples: int = 100, verbose: bool = True):
        # finds minimum and maximum x values
        x_vals = self.data[0]
        min_x, max_x = (min(x_vals[:, 0]), max(x_vals[:, 0]))

        # creates n_samples data points between min_x and max_x
        x_samples = np.linspace(min_x, max_x, n_samples)

        # finds the mean and variance for each element in x_samples
        mean, var = self.predict_f(x_samples[:, None])
        if verbose:
            print("min_x: {}\tshape: {}\nmax_x: {}\tshape: {}\nx_samples shape: {}\nmean shape: {}\nvar shape: {}"
                  .format(min_x, min_x.shape, max_x, max_x.shape, x_samples.shape, mean.shape, var.shape))

        # Plots the mean function predicted by the GP
        plt.plot(x_samples, mean, c='green', label='$M_c$')
        # Plots the 95% confidence interval
        # TODO: figure out why the variance is so small
        plt.fill_between(x_samples, mean[:, 0] - 1.96 * np.sqrt(var[:, 0]), mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                         color='green', alpha=0.2)


class DiscontinuousModel(BNQDModel):

    def __init__(self, data: DiscontinuousData, kernel: Kernel, likelihood: Likelihood, intv_point: float,
                 labels: Union[List, Callable], shared_params: bool = False,
                 mean_function: Optional[MeanFunction] = None, num_latent: int = 1):
        super().__init__(data, kernel, likelihood, mean_function, num_latent)
        raise NotImplementedError

    def train(self, optimizer=optimizers.Scipy(), verbose=True):
        raise NotImplementedError

    def log_likelihood(self, *args, **kwargs) -> tf.Tensor:
        raise NotImplementedError

    def predict_f(self, predict_at: DataPoint, full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        raise NotImplementedError

    def plot(self, n_samples: int = 100):
        raise NotImplementedError


######################
##### TEST STUFF #####
######################

np.random.seed(1984)

b = 0.0
n = 100
x = np.linspace(-3, 3, n)
f = 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 4) + 1.0 * (x > b)
sigma = 0.65
y = np.random.normal(f, sigma, size=n)
print(sigma)
plt.figure()

if True:  # disables / enables plotting of the data points and generative function
    plt.plot(x[x <= b], f[x <= b], label='True f', c='k')
    plt.plot(x[x >= b], f[x >= b], c='k')
    plt.axvline(x=b, linestyle='--', c='k')
    plt.plot(x, y, linestyle='none', marker='x', color='k', label='obs')


###### Kernel Options #####

#k = Linear() + Constant() # "Linear" kernel
#k = Exponential()
k = SquaredExponential()
#k = Periodic(SquaredExponential())
#k = Cosine() + Constant()

cm = ContinuousModel((x, y), k, Gaussian())
print("trainable parameters: {}".format(cm.m.trainable_parameters))
cm.train()
cm.plot()
plt.show()
print("\ncontinuous model:\n\tBIC score: {}\n\tlog likelihood: {}".format(cm.log_likelihood(), cm.m.log_likelihood()))
