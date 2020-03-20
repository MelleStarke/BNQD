import abc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gf
import warnings
import copy
import sys

from typing import Optional, Tuple, Union, List, Callable, Any

from numpy import ndarray

from tensorflow import Tensor

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
ContinuousData = Tuple[Union[ndarray, Tensor], Union[ndarray, Tensor]]

# Data for the discontinuous model: tuple of continuous data
DiscontinuousData = List[ContinuousData]

# Data for the abstract BNQD model: either continuous data or discontinuous data
Data = Union[ContinuousData, DiscontinuousData]


##############################
###### Global Constants ######
##############################

MAX_OPTIMIZER_ITERATIONS = 100


##############################
###### Class Definitions #####
##############################

class BNQDAnalysis:
    """
    BNQD analysis object.
    This contains both the continuous and the discontinuous model,
    as well as some methods for manipulating and comparing these models.
    """

    def __init__(self, data: Data, kernel: Kernel, likelihood: Likelihood, intervention_point: Tensor,
                 share_params: bool = False, marginal_likelihood_method: str = "bic", optimizer: Any = None, **kwargs):
        """

        :param data:
                    Data object used to train the models. Either a tuple of tensors, or a list of tuples of tensors.
        :param kernel:
                    Kernel used by the models.
        :param likelihood:
                    Method for computing the likelihood over the data points. Currently not used, since GPR. objects
                    create their own likelihood parameter (which is always Gaussian).
        :param intervention_point:
                    Point at which the effect size is measured, and where the discontinuous model changes sub-model.
        :param share_params:
                    Whether or not the discontinuous sub-models (control and intervention) share hyper parameters.
        :param marginal_likelihood_method:
                    Method used to compute the marginal likelihood. Can be either BIC-score or the native GPflow method.
        :param optimizer:
                    Optimizer used for estimating the hyper parameters.
        :param kwargs:
                    Additional optional parameters for the BayesianModel and GPModel classes. These include the
                    mean_function: MeanFunction = None, and num_latent: int = 1
        """

        # TODO: make the class able to handle ContinuousData, and then split it using the intervention_point
        self.continuous_model = ContinuousModel(flatten_data(data), kernel, **kwargs)
        self.discontinuous_model = DiscontinuousModel(data, kernel, intervention_point, share_params, **kwargs)
        self.optimizer = optimizer
        self.marginal_likelihood_method = marginal_likelihood_method

    def train(self, optimizer=None):
        """
        Trains both the continuous and the discontinuous model
        """

        # Uses the optimizer passed to the function, if not,
        # then the optimizer assigned to the BNQDAnalysis object.
        # Will only be None if both are None
        optimizer = optimizer if optimizer else self.optimizer

        # If neither the optimizer passed to the function, nor self.optimizer are None,
        # use this optimizer to train the models
        if optimizer:
            self.continuous_model.train(optimizer)
            self.discontinuous_model.train(optimizer)

        # If both are none, train the models with the default optimizer
        else:
            self.continuous_model.train()
            self.discontinuous_model.train()

    def plot(self):
        """
        Plots both the continuous and the discontinuous model
        """

        # TODO: return a pyplot object instead to allow separate plotting
        self.continuous_model.plot()
        self.discontinuous_model.plot()
        plt.show()  # <- ugly

    def bayes_factor(self, method: str = None) -> tf.Tensor:
        """
        Computes the Bayes factor of the two models

        :param method: Method used for calculating the marginal likelihood (BIC or the native GPflow method)
        :return: Bayes factor of the discontinuous model to the continuous model: $BF_{M_D M_C}$
        """

        # Results in True if and only if both models are trained
        if not all(map((lambda m: m.is_trained), [self.continuous_model, self.discontinuous_model])):
            msg = "Not all models have been trained, so the Bayes factor will not be representative.\n"\
                  "Assuming your BNQDAnalysis object is called 'ba', you can check this with:\n"\
                  "\t'ba.continuous_model.is_trained' and 'ba.discontinuous_model.is_trained'\n"\
                  "Train both models at the same time with 'ba.train()'"
            warnings.warn(msg, category=UserWarning)

        # Determines which marginal likelihood computation method to use.
        # Uses the method passed to the function, if it exists. Otherwise, uses the method assigned to the object.
        method = method if method else self.marginal_likelihood_method

        # Computes the Bayes factor by subtracting the two tensors element-wise, on the first axis.
        # Typically, these tensors will only contain one element.
        bayes_factor = tf.reduce_sum([self.discontinuous_model.log_marginal_likelihood(method),
                                      -self.continuous_model.log_marginal_likelihood(method)], 0)
        print("Bayes factor M_D - M_C: {}".format(bayes_factor))
        return bayes_factor


class BNQDModel(GPModel):
    """
    Abstract BNQD model class.
    Inherits GPModel, which is an abstract class used for Bayesian prediction.
    """

    def __init__(self, data: Data, kernel: Kernel, likelihood: Likelihood,
                 mean_function: Optional[MeanFunction] = None, num_latent: int = 1):

        # TODO: figure out what a latent variable is in this context (num_latent)
        super().__init__(kernel, likelihood, mean_function, num_latent)
        self.data = data
        self.is_trained = False
        self.BIC_score = None

    @abc.abstractmethod
    def train(self, optimizer=optimizers.Scipy(), verbose=True) -> None:
        # TODO: Make an optimizer wrapper that allows for easy specification of parameters for different optimizers
        raise NotImplementedError

    def log_likelihood(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def predict_f(self, predict_at: DataPoint, full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        raise NotImplementedError

    @abc.abstractmethod
    def plot(self, n_samples: int = 100):
        raise NotImplementedError


class ContinuousModel(BNQDModel):
    """
    Continuous BNQD model
    """

    def __init__(self, data: ContinuousData, kernel: Kernel, **kwargs):

        # TODO: check if this data set shape stuff actually works
        # Checks if the input data is of rank 1 (i.e. vector)
        if len(data[0].shape) == 1:
            # Converts the input data of shape (N) to shape (N,1), to comply with tensorflow.matmul requirements
            x, y = data
            data = (x[:, None], y[:, None])

        self.N = data[0].shape[0]  # nr. of data points
        self.model = GPR(data, kernel)  # Single GP regression model
        super().__init__(data, self.model.kernel, self.model.likelihood, **kwargs)

    def train(self, optimizer=optimizers.Scipy(), verbose=True) -> None:
        """
        Trains the GPR
        :param optimizer: Optimizer used to estimate the optimal hyper parameters
        :param verbose: Whether or not it should print a summary of the model after training
        """

        # Uses the optimizer to minimize the objective_closure function, by adjusting the trainable variables.
        # The trainable variables are obtained by recursively finding fields of type Variable,
        # if and only if they're defined as being trainable.
        optimizer.minimize(self.objective_closure, self.trainable_variables,
                           options=(dict(maxiter=MAX_OPTIMIZER_ITERATIONS)))

        if verbose:
            print("Continuous model:")
            # Prints a summary of the model
            gf.utilities.print_summary(self.model)

        self.is_trained = True

        # Ensures the BIC score is updated when trying to access it after training the model
        self.BIC_score = None

    def log_likelihood(self, *args, **kwargs) -> Tensor:
        """
        Returns the log likelihood of the GPR model
        :param args:
        :param kwargs:
        :return: Log likelihood of the BNQD model
        """

        if not self.is_trained:
            # Prints a warning if the model hasn't been trained.
            msg = "The model hasn't been trained yet. Please use the ContinuousModel.train() function"
            warnings.warn(msg, category=UserWarning)
        return self.model.log_likelihood()

    def log_marginal_likelihood(self, method="bic"):
        """
        Computes (if non-existent) and returns the log marginal likelihood of the GPR model.
        This is done via one of two methods: using the BIC score, or with GPflow's native implementation.
        :param method:
        :return:
        """

        method = method.lower()

        if method in ["bic", "bic score", "BIC_score"]:
            if not self.BIC_score:  # Means: if self.BIC_score == None
                # Parameters are represented as tuple, for documentation see gpf.Module
                k = len(self.trainable_parameters)
                L = self.log_likelihood()
                BIC = L - k / 2 * np.log(self.N)
                self.BIC_score = BIC
            return self.BIC_score

        elif method in ["native", "nat", "gpflow"]:
            return self.model.log_marginal_likelihood()

        else:
            raise ValueError("Incorrect method for log marginal likelihood calculation: {}"
                             "Please use either 'bic' or 'native' (i.e. gpflow method)".format(method))

    def predict_f(self, predict_at: DataPoint, full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        """
        Computes the mean and variance of the latent function.
        :param predict_at:
        :param full_cov:
        :param full_output_cov:
        :return: Mean and variance of the latent function.
        """

        return self.model.predict_f(predict_at, full_cov, full_output_cov)

    def plot(self, n_samples: int = 100, verbose: bool = False) -> None:
        # finds minimum and maximum x values
        x_vals = self.data[0]
        min_x, max_x = (min(x_vals[:, 0]), max(x_vals[:, 0]))

        # creates n_samples data points between min_x and max_x
        x_samples = np.linspace(min_x, max_x, n_samples)

        # finds the mean and variance for each element in x_samples
        mean, var = self.predict_y(x_samples[:, None])
        if verbose:
            print("min_x: {}\tshape: {}\nmax_x: {}\tshape: {}\nx_samples shape: {}\nmean shape: {}\nvar shape: {}"
                  .format(min_x, min_x.shape, max_x, max_x.shape, x_samples.shape, mean.shape, var.shape))

        # Plots the mean function predicted by the GP
        plt.plot(x_samples, mean, c='green', label='$continuous_model$')

        # Plots the 95% confidence interval
        plt.fill_between(x_samples, mean[:, 0] - 1.96 * np.sqrt(var[:, 0]), mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                         color='green', alpha=0.2)  # 0.5

    def objective_closure(self) -> Tensor:
        """
        Function that the optimizer aims to minimize.
        :return: The negative log likelihood of the GPR model
        """

        return -self.model.log_likelihood()


class DiscontinuousModel(BNQDModel):
    """
    Discontinuous BNQD model.
    """

    def __init__(self, data: DiscontinuousData, kernel: Kernel, intervention_point: Tensor,
                 share_params: bool = True, **kwargs):

        # TODO: check if this data set shape stuff actually works
        # Checks if the input data is of rank 1 (i.e. vector)
        # Converts the input data of shape (N) to shape (N,1), to comply with tensorflow.matmul requirements
        # The data is split between control and intervention data, and is stored as a list of tuples of tensors
        if len(data[0][0].shape) == 1:
            res = list()
            for x_vals, y_vals in data:
                res.append((x_vals[:, None], y_vals[:, None]))
            data = res

        # Cannot pass the likelihood and kernel to the superclass.
        # This is because discontinuous model contains two sub-models, with their own kernel and likelihood objects.
        # And so far I haven't seen a way to create a new kernel and likelihood object that properly "splits"
        # TODO: figure this out maybe?
        super().__init__(data, None, None, **kwargs)

        self.share_params = share_params
        self.intervention_point = intervention_point
        self.N = np.shape(data[0][0])[0] + np.shape(data[1][0])[0]  # nr. of data points

        # Model used before the intervention point
        self.control_model = GPR(self.data[0], copy.deepcopy(kernel))

        # Model used after the intervention point
        self.intervention_model = GPR(self.data[1], copy.deepcopy(kernel))

        # Sets all parameters in the intervention model as non-trainable, if the models share parameters
        if share_params:
            for p in self.intervention_model.trainable_parameters:
                gf.utilities.set_trainable(p, False)

    def train(self, optimizer=optimizers.Scipy(), verbose=True):
        """
        Trains both sub-models
        :param optimizer:
        :param verbose:
        :return:
        """

        # Uses the optimizer to minimize the objective_closure function, by adjusting the trainable variables.
        # The trainable variables are obtained by recursively finding fields of type Variable,
        # if and only if they're defined as being trainable.
        optimizer.minimize(self.objective_closure, self.trainable_variables,
                           options=(dict(maxiter=MAX_OPTIMIZER_ITERATIONS)))

        if verbose:

            # Prints summaries of both models
            for name, model in [("Control", self.control_model), ("Intervention", self.intervention_model)]:
                print("{} model:".format(name))
                gf.utilities.print_summary(model)

        if self.share_params:
            # Sets all trainable variables of the intervention model to be equal to the ones of the control model
            self.equalize_parameters()

        self.is_trained = True

        # Ensures the BIC score is updated when trying to access it after training the model
        self.BIC_score = None

    def log_likelihood(self, *args, **kwargs) -> Tensor:
        """
        Returns the log likelihood as the sum of the log likelihoods of the sub-models
        :param args:
        :param kwargs:
        :return: Log likelihood of the BNQD model
        """

        if not self.is_trained:
            # Prints a warning if the model hasn't been trained.
            msg = "The model hasn't been trained yet. Please use the DiscontinuousModel.train() function"
            warnings.warn(msg, category=UserWarning)
        return self.control_model.log_likelihood() + self.intervention_model.log_likelihood()

    def log_marginal_likelihood(self, method="bic"):
        """
        Computes (if non-existent) and returns the log marginal likelihood of the GPR model.
        This is done via one of two methods: using the BIC score, or with GPflow's native implementation.
        :param method:
        :return:
        """

        method = method.lower()

        if method in ["bic", "bic score", "BIC_score"]:
            if not self.BIC_score:  # Means: if self.BIC_score == None
                k = len(self.trainable_variables)
                L = self.log_likelihood()
                BIC = L - k / 2 * np.log(self.N)
                self.BIC_score = BIC
            return self.BIC_score

        elif method in ["native", "nat", "gpflow"]:
            return self.control_model.log_marginal_likelihood() + self.intervention_model.log_marginal_likelihood()

        else:
            raise ValueError("Incorrect method for log marginal likelihood calculation: {}"
                             "Please use either 'bic' or 'native' (i.e. gpflow method)".format(method))

    def predict_f(self, predict_at: DataPoint, use_control_model_at_intervention_point: bool = False,
                  full_cov: bool = False, full_output_cov: bool = False) -> List[MeanAndVariance]:
        # TODO: make this work for non-sequential predict_at tensors
        # TODO: Change the code to output a list of MeanAndVariance, and override inherited functions
        #       (e.g. predict_y) to comply with this change

        predict_c = predict_at[predict_at[:, 0] < self.intervention_point, :]
        predict_i = predict_at[predict_at[:, 0] > self.intervention_point, :]

        # predict the data point at the intervention point depending on which model is specified
        # TODO: remove this part cause it's ugly
        if self.intervention_point in predict_at:
            if use_control_model_at_intervention_point:
                predict_c += self.intervention_point
            else:
                predict_i = self.intervention_point + predict_i

        mean_c, var_c = self.control_model.predict_f(predict_c, full_cov, full_output_cov)
        mean_i, var_i = self.intervention_model.predict_f(predict_i, full_cov, full_output_cov)

        return tf.concat([mean_c, mean_i], 0), tf.concat([var_c, var_i], 0)
        #return self.intervention_model.predict_f(predict_at, full_cov, full_output_cov)

    def plot(self, n_samples: int = 100, verbose: bool = False):
        # TODO: make this plotting function not so shitty
        # finds minimum and maximum x values
        x_vals = self.data[0][0] + self.data[1][0]
        min_x, max_x = (min(x_vals[:, 0]), max(x_vals[:, 0]))

        # creates n_samples data points between min_x and max_x
        x_samples = np.linspace(min_x, max_x, n_samples)
        x_samples_c = x_samples[x_samples < self.intervention_point] + self.intervention_point
        x_samples_i = self.intervention_point + x_samples[x_samples > self.intervention_point]

        if verbose:
            print("x_samples_c shape: {}, x_samples_i shape: {}".format(x_samples_c.shape, x_samples_i.shape))

        # finds the mean and variance for each element in x_samples
        mean_c, var_c = self.control_model.predict_y(x_samples_c[:, None])
        mean_i, var_i = self.intervention_model.predict_y(x_samples_i[:, None])
        if verbose:
            print("min_x: {}\tshape: {}\nmax_x: {}\tshape: {}\nx_samples_c shape: {}\nmean_c shape: {}\nvar_c shape: {}\n"
                  "x_samples_c shape: {}\nmean_c shape: {}\nvar shape: {}"
                  .format(min_x, min_x.shape, max_x, max_x.shape, x_samples_c.shape, mean_c.shape, var_c.shape,
                          x_samples_i.shape, mean_i.shape, var_i.shape))

        # Plots the mean function predicted by the GP
        plt.plot(x_samples_c, mean_c[:, 0], c='blue', label='$control_model$')
        plt.plot(x_samples_i, mean_i[:, 0], c='blue', label='$control_model$')
        # Plots the 95% confidence interval
        # TODO: figure out why the variance is SO BIG AFTER THE INTERVENTION POINT
        plt.fill_between(x_samples_c, mean_c[:, 0] - 1.96 * np.sqrt(var_c[:, 0]),
                         mean_c[:, 0] + 1.96 * np.sqrt(var_c[:, 0]), color='blue', alpha=0.2)
        plt.fill_between(x_samples_i, mean_i[:, 0] - 1.96 * np.sqrt(var_i[:, 0]),
                         mean_i[:, 0] + 1.96 * np.sqrt(var_i[:, 0]),
                         color='blue', alpha=0.2)

    def objective_closure(self):
        if self.share_params:
            self.equalize_parameters()

        return -(self.control_model.log_likelihood() + self.intervention_model.log_likelihood())

    def equalize_parameters(self) -> None:
        """
        Sets the trainable parameters of the intervention model to be equal to the ones of the control model.
        However, it assumes that all parameters occur in the same order in both models.
        If both models are of the same class, and initialized with the same arguments, this shouldn't be an issue.
        """

        for pc, pi in zip(self.control_model.parameters, self.intervention_model.parameters):
            if pc.trainable:
                # Assigns the value of the trainable parameter of the control model to the one of the intervention model
                pi.assign(pc)


##########################
##### STATIC METHODS #####
##########################


def flatten_data(data):
    """
    Turns data fit for the discontinuous model into data fit for the continuous model.
    :param data:
    :return:
    """
    x_res, y_res = np.array([]), np.array([])
    for x, y in data:
        x_res = np.append(x_res, x)
        y_res = np.append(y_res, y)
    return (x_res, y_res)


######################
##### TEST STUFF #####
######################


# TODO: look into tensorflow_probability
# TODO: figure out if (and where) the kernel parameter of GPModel is used (I guess nowhere)
# TODO: figure out if I can do a print() or warn() when calling a parameter
# TODO: figure out where I can add simulated annealing
# TODO: figure out what num_latent means
# TODO: add an effect size measure

