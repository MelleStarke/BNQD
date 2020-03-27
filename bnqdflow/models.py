import numpy as np
import matplotlib.pyplot as plt
import gpflow as gf
import warnings
import bnqdflow.util as util

from abc import abstractmethod

from bnqdflow.data_types import ContinuousData, DiscontinuousData

from typing import Optional, Tuple, Union, List

from numpy import ndarray

from tensorflow import Tensor

from gpflow import optimizers
from gpflow.kernels import Kernel
from gpflow.models import GPModel, GPR
from gpflow.likelihoods import Likelihood
from gpflow.models.model import DataPoint, MeanAndVariance, Data
from gpflow.mean_functions import MeanFunction


##############################
###### Global Constants ######
##############################

counter = 0

MAX_OPTIMIZER_ITERATIONS = 100


class BaseGPRModel(GPModel):
    """
    Abstract GP regression model class.
    Inherits GPModel, which is an abstract class used for Bayesian prediction.
    """

    def __init__(self, data: Union[ContinuousData, DiscontinuousData], kernel: Kernel, likelihood: Likelihood,
                 mean_function: Optional[MeanFunction] = None, num_latent: int = 1):
        # TODO: figure out what a latent variable is in this context (num_latent)
        super().__init__(kernel, likelihood, mean_function, num_latent)
        self.data = data
        self.is_trained = False
        self.BIC_score = None

    @abstractmethod
    def train(self, optimizer=optimizers.Scipy(), verbose=True) -> None:
        # TODO: Make an optimizer wrapper that allows for easy specification of parameters for different optimizers
        raise NotImplementedError

    def log_likelihood(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def predict_f(self, predict_at: DataPoint, full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        raise NotImplementedError

    @abstractmethod
    def plot(self, n_samples: int = 100):
        raise NotImplementedError


class ContinuousModel(BaseGPRModel):
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


class DiscontinuousModel(BaseGPRModel):
    """
    Discontinuous BNQD model.
    """

    def __init__(self, data: DiscontinuousData, kernel: Kernel, intervention_point: Tensor,
                 share_params: bool = True, **kwargs):

        # TODO: check if this data set shape stuff actually works
        # Checks if the input data is of rank 1 (i.e. vector)
        # Converts the input data of shape (N) to shape (N,1), to comply with tensorflow.matmul requirements
        # The data is split between control and intervention data, and is stored as a list of tuples of tensors
        data = list(map(lambda section: (util.ensure_tf_vector_format(section[0]),
                                         util.ensure_tf_vector_format(section[1])),
                        data))

        # Cannot pass the likelihood and kernel to the superclass.
        # This is because discontinuous model contains two sub-models, with their own kernel and likelihood objects.
        # And so far I haven't seen a way to create a new kernel and likelihood object that properly "splits"
        # TODO: figure this out maybe?
        super().__init__(data, None, None, **kwargs)

        self.share_params = share_params
        self.intervention_point = intervention_point
        self.N = np.shape(data[0][0])[0] + np.shape(data[1][0])[0]  # nr. of data points

        # Model used before the intervention point
        self.control_model = GPR(self.data[0], gf.utilities.deepcopy_components(kernel))

        # Model used after the intervention point
        self.intervention_model = GPR(self.data[1], gf.utilities.deepcopy_components(kernel))

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
        '''
        if self.share_params:
            # Sets all trainable variables of the intervention model to be equal to the ones of the control model
            self.equalize_parameters()
        '''
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

    def predict_f(self, predict_at: List[Union[DataPoint, ndarray]], use_control_model_at_intervention_point: bool = False,
                  full_cov: bool = False, full_output_cov: bool = False) -> List[MeanAndVariance]:
        # TODO: make this work for non-sequential predict_at tensors
        # TODO: Change the code to output a list of MeanAndVariance, and override inherited functions
        #       (e.g. predict_y) to comply with this change
        predict_at = list(map(util.ensure_tf_vector_format, predict_at))
        res = list()
        for model, section in zip([self.control_model, self.intervention_model], predict_at):
            res.append(model.predict_f(section, full_cov, full_output_cov))

        return res

    def predict_f_samples(self, predict_at: List[Union[DataPoint, ndarray]], num_samples: int = 1, full_cov: bool = True,
                          full_output_cov: bool = True):
        predict_at = list(map(util.ensure_tf_vector_format, predict_at))
        res = list()
        for model, section in zip([self.control_model, self.intervention_model], predict_at):
            res.append(model.predict_f_samples(section, num_samples, full_cov, full_output_cov))
        return res

    def predict_y(self, predict_at: List[Union[DataPoint, ndarray]], full_cov: bool = False,
                  full_output_cov: bool = False) -> List[MeanAndVariance]:
        predict_at = list(map(util.ensure_tf_vector_format, predict_at))
        res = list()
        for model, section in zip([self.control_model, self.intervention_model], predict_at):
            res.append(model.predict_y(section, full_cov, full_output_cov))
        return res

    def predict_log_density(self, data: DiscontinuousData, full_cov: bool = False, full_output_cov: bool = False):
        # TODO: check if I have to ensure the data is in the correct tensorflow format
        res = list()
        for model, section in zip([self.control_model, self.intervention_model], data):
            res.append(model.predict_log_density(section, full_cov, full_output_cov))
        return res

    def plot(self, n_samples: int = 100, verbose: bool = True):
        # TODO: make this plotting function not so shitty
        # finds minimum and maximum x values

        x_vals = self.data[0][0] + self.data[1][0]
        min_x, max_x = (min(x_vals[:, 0]), max(x_vals[:, 0]))
        ip = self.intervention_point

        # The following lines split the number of x samples into a number of control samples and intervention samples
        # in such a way that they have the same "density", where n_c + n_i = n_samples + 1. The sum is 1 higher than
        # n_samples because the intervention point should be present in both, basically meaning that it overlaps.
        control_ratio, intervention_ratio = (ip - min_x) / (max_x - min_x), (max_x - ip) / (max_x - min_x)
        # n_c is the number of x samples used to plot the control graph, idem for n_i and the intervention graph
        n_c, n_i = n_samples * control_ratio + 1., n_samples * intervention_ratio + 1.
        if n_c % 1 == 0:
            n_c += -1.

        x_samples_list = [np.linspace(min_x, ip, int(n_c))[:, None],
                          np.linspace(ip, max_x, int(n_i))[:, None]]

        # Predicts the means and variances for both x_samples
        means_and_vars = self.predict_y(x_samples_list)

        if verbose:
            print(
                "min_x: {}\tshape: {}\nmax_x: {}\tshape: {}\nx_samples_list shape: {}"
                .format(min_x, min_x.shape, max_x, max_x.shape, np.shape(x_samples_list)))

        for x_samples, (mean, var) in zip(x_samples_list, means_and_vars):
            # Plots the mean function predicted by the GP
            plt.plot(x_samples[:, 0], mean[:, 0], c='blue', label='$control_model$')
            # Plots the 95% confidence interval
            # TODO: figure out why the variance is SO BIG AFTER THE INTERVENTION POINT
            plt.fill_between(x_samples[:, 0], mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                             mean[:, 0] + 1.96 * np.sqrt(var[:, 0]), color='blue', alpha=0.2)

    def objective_closure(self):
        if self.share_params:
            self.equalize_parameters()

        return -(self.control_model.log_likelihood() + self.intervention_model.log_likelihood())

    def equalize_parameters(self) -> None:
        """
        Sets the trainable parameters of the intervention model to be equal to the ones of the control model.
        """

        params = gf.utilities.parameter_dict(self.control_model)
        gf.utilities.multiple_assign(self.intervention_model, params)
