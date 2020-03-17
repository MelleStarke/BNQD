import abc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gf
import warnings
import copy

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
            self.data = (x.reshape(-1, 1), y.reshape(-1, 1))

        else:  # Assigns without changing the dimensions of the input data
            self.data = data

        self.N = data[0].shape[0]  # nr. of data points
        self.m = GPR(self.data, self.kernel)

    def train(self, optimizer=optimizers.Scipy(), verbose=True):
        # function that the optimizer aims to minimize
        closure = lambda: -self.m.log_marginal_likelihood()

        optimizer.minimize(closure, self.m.trainable_variables, options=(dict(maxiter=self.MAX_OPT_ITER)))
        if verbose:
            gf.utilities.print_summary(self.m)

        self.isTrained = True
        self.BICScore = None  # Ensures the BIC score is updated when trying to access it after training the model

    def log_likelihood(self, *args, **kwargs) -> tf.Tensor:
        if not self.isTrained:
            # Prints a warning if the model hasn't been trained.
            msg = "The model hasn't been trained yet. Please use the ContinuousModel.train() function"
            warnings.warn(msg, category=UserWarning)

        return self.BICScore

    def log_marginal_likelihood(self, method="bic"):
        method = method.lower()

        if method in ["bic", "bic score", "bicscore"]:
            if not self.BICScore:  # Means: if self.BICScore == None
                k = len(self.m.parameters)  # parameters are represented as tuple, for documentation see gpf.Module
                L = self.m.log_likelihood()
                BIC = L - k / 2 * np.log(self.N)
                self.BICScore = BIC

            return self.BICScore

        elif method in ["native", "nat", "gpflow"]:
            return self.m.log_marginal_likelihood()

        else:
            raise ValueError("Incorrect method for log marginal likelihood calculation: {}"
                             "Please use either 'bic' or 'native' (i.e. gpflow method)".format(method))


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
        mean, var = self.predict_f(x_samples.reshape(-1, 1))
        if verbose:
            print("min_x: {}\tshape: {}\nmax_x: {}\tshape: {}\nx_samples shape: {}\nmean shape: {}\nvar shape: {}"
                  .format(min_x, min_x.shape, max_x, max_x.shape, x_samples.shape, mean.shape, var.shape))

        # Plots the mean function predicted by the GP
        plt.plot(x_samples, mean, c='green', label='$M_c$')
        # Plots the 95% confidence interval
        # TODO: figure out why the variance is so small
        plt.fill_between(x_samples, mean[:, 0] - 1.96 * np.sqrt(var[:, 0]), mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                         color='green', alpha=0.2)  # 0.5


class DiscontinuousModel(BNQDModel):

    def __init__(self, data: DiscontinuousData, kernel: Kernel, likelihood: Likelihood, intv_point,
                 shared_params: bool = False, mean_function: Optional[MeanFunction] = None, num_latent: int = 1):
        # TODO: allow the user to set this in an elegant way
        # Maximum optimizer iterations
        self.MAX_OPT_ITER = 1000

        super().__init__(data, kernel, likelihood, mean_function, num_latent)
        print("len data[0][0] shape: {}".format(len(data[0][0].shape)))
        # TODO: check if this data set shape stuff actually works
        if len(data[0][0].shape) == 1:  # Checks if the input data is of rank 1 (i.e. vector)
            # Converts the input data of shape (N) to shape (N,1), to comply with tensorflow.matmul requirements
            # The data is split between control and intervention data
            (x_c, y_c), (x_i, y_i) = data
            self.data = ((x_c.reshape(-1, 1), y_c.reshape(-1, 1)), (x_i.reshape(-1, 1), y_i.reshape(-1, 1)))

        else:  # Assigns without changing the dimensions of the input data
            self.data = data

        self.shared_params = shared_params
        self.intv_point = intv_point
        self.N = data[0][0].shape[0] + data[1][0].shape[0] # nr. of data points
        data_c, data_i = self.data
        self.m_c = GPR(data_c, self.kernel)
        self.m_i = GPR(data_i, self.kernel)

    def train(self, optimizer=optimizers.Scipy(), verbose=True):
        # function that the optimizer aims to minimize
        closure = lambda: -(self.m_c.log_marginal_likelihood() + self.m_i.log_marginal_likelihood())

        optimizer.minimize(closure, self.m_c.trainable_parameters + self.m_i.trainable_parameters,
                           options=(dict(maxiter=self.MAX_OPT_ITER)))
        if verbose:
            gf.utilities.print_summary(self.m_c)
            gf.utilities.print_summary(self.m_i)

        self.isTrained = True
        self.BICScore = None  # Ensures the BIC score is updated when trying to access it after training the model

    def log_likelihood(self, *args, **kwargs) -> tf.Tensor:
        if not self.isTrained:
            # Prints a warning if the model hasn't been trained.
            # Then trains it with the default optimizer
            msg = "The model hasn't been trained yet. It will be trained using the default optimizer.\n" \
                  "If you wish to use a different optimizer, please use the DiscontinuousModel.train(optimizer) function"
            warnings.warn(msg, category=UserWarning)
            self.train()

        if not self.BICScore:  # Means: if self.BICScore == None
            # Parameters are represented as tuple, for documentation see gpf.Module
            # Will only give the number of parameters of the control model if the parameters are shared
            k = len(self.m_c.parameters) + len(self.m_i.parameters) * (not self.shared_params)
            L = self.m_c.log_likelihood() + self.m_i.log_likelihood()
            BIC = L - k / 2 * np.log(self.N)  # BIC score is the likelihood with penalization for the nr. of parameters
            self.BICScore = BIC
        return self.BICScore

    def predict_f(self, predict_at: DataPoint, use_control_model_at_intervention_point: bool = False,
                  full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        # TODO: make this work for non-sequential predict_at tensors
        print("predict_at shape: {}".format(predict_at.shape))

        predict_c = predict_at[predict_at[:, 0] < self.intv_point, :]
        predict_i = predict_at[predict_at[:, 0] > self.intv_point, :]

        print("predict_c shape: {}\tpredict_i shape: {}".format(predict_c.shape, predict_i.shape))

        # predict the data point at the intervention point depending on which model is specified
        if self.intv_point in predict_at:
            print("intv point in samples")
            if use_control_model_at_intervention_point:
                predict_c += self.intv_point
            else:
                predict_i = self.intv_point + predict_i

        mean_c, var_c = self.m_c.predict_f(predict_c, full_cov, full_output_cov)
        mean_i, var_i = self.m_i.predict_f(predict_i, full_cov, full_output_cov)

        return tf.concat([mean_c, mean_i], 0), tf.concat([var_c, var_i], 0)
        #return self.m_i.predict_f(predict_at, full_cov, full_output_cov)

    def plot(self, n_samples: int = 100, verbose: bool = True):
        # finds minimum and maximum x values
        x_vals = self.data[0][0] + self.data[1][0]
        min_x, max_x = (min(x_vals[:, 0]), max(x_vals[:, 0]))

        # creates n_samples data points between min_x and max_x
        x_samples = np.linspace(min_x, max_x, n_samples)
        x_samples_c = x_samples[x_samples < self.intv_point] + self.intv_point
        x_samples_i = self.intv_point + x_samples[x_samples > self.intv_point]

        print("x_samples_c shape: {}, x_samples_i shape: {}".format(x_samples_c.shape, x_samples_i.shape))

        # finds the mean and variance for each element in x_samples
        mean_c, var_c = self.predict_f(x_samples_c.reshape(-1, 1), use_control_model_at_intervention_point = True)
        mean_i, var_i = self.predict_f(x_samples_i.reshape(-1, 1), use_control_model_at_intervention_point = False)
        if verbose:
            print("min_x: {}\tshape: {}\nmax_x: {}\tshape: {}\nx_samples_c shape: {}\nmean_c shape: {}\nvar_c shape: {}\n"
                  "x_samples_c shape: {}\nmean_c shape: {}\nvar shape: {}"
                  .format(min_x, min_x.shape, max_x, max_x.shape, x_samples_c.shape, mean_c.shape, var_c.shape,
                          x_samples_i.shape, mean_i.shape, var_i.shape))

        # Plots the mean function predicted by the GP
        plt.plot(x_samples_c, mean_c[:, 0], c='blue', label='$M_c$')
        plt.plot(x_samples_i, mean_i[:, 0], c='blue', label='$M_c$')
        # Plots the 95% confidence interval
        # TODO: figure out why the variance is SO BIG AFTER THE INTERVENTION POINT
        plt.fill_between(x_samples_c, mean_c[:, 0] - 1.96 * np.sqrt(var_c[:, 0]),
                         mean_c[:, 0] + 1.96 * np.sqrt(var_c[:, 0]), color='blue', alpha=0.2)
        plt.fill_between(x_samples_i, mean_i[:, 0] - 1.96 * np.sqrt(var_i[:, 0]),
                         mean_i[:, 0] + 1.96 * np.sqrt(var_i[:, 0]),
                         color='blue', alpha=0.2)


######################
##### TEST STUFF #####
######################

# TODO: figure out why different values for sigma either don't change the variance or cause a tensorflow error
np.random.seed(1984)

#gf.reset_default_graph_and_session()

b = 0.0
n = 100
x = np.linspace(-3, 3, n)
f = 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 4) + 1.0 * (x > b)
sigma = 3
y = np.random.normal(f, sigma, size=n)
print(sigma)

ip = 0
x_c = x[x < ip]
y_c = y[x < ip]
x_i = x[x >= ip]
y_i = y[x >= ip]

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

# Use the continuous model or discontinuous one
useContinuous = 1

m = None
if useContinuous:
    m = ContinuousModel((x, y), k, Gaussian())
else:
    m = DiscontinuousModel(((x_c, y_c), (x_i, y_i)), k, Gaussian(), ip)
m.train()
m.plot(100)
plt.show()

if useContinuous:
    print("\ncontinuous model:\n\tBIC score: {}\n\tlog marginal likelihood: {}"
          .format(m.log_marginal_likelihood("bic"), m.log_marginal_likelihood("native")))
    print(m.m.trainable_variables)
    print(m.m.trainable_parameters)

# TODO: look into tensorflow_probability
# TODO: figure out the arguments for the log_likelihood() and log_marginal_likelihood() functions
# TODO: figure out what the role of the priors is in the BayesianModel.log_marginal_likelihood() function
# TODO: figure out where I can add simulated annealing

