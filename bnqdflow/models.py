import numpy as np
import matplotlib.pyplot as plt
import gpflow as gf
import warnings
import bnqdflow.util as util
import tensorflow as tf

from abc import abstractmethod

from bnqdflow.data_types import ContinuousData, DiscontinuousData

from typing import Optional, Tuple, Union, List, Any

from numpy import ndarray

from tensorflow import Tensor

from itertools import cycle

from gpflow import optimizers
from gpflow.kernels import Kernel
from gpflow.models import GPModel, GPR
from gpflow.likelihoods import Likelihood
from gpflow.models.model import InputData, MeanAndVariance
from gpflow.mean_functions import MeanFunction

##############################
###### Global Constants ######
##############################

counter = 0

MAX_OPTIMIZER_ITERATIONS = 100
N_MODELS = 2  # Nr. of sub-models in the discontinuous model


class BNQDRegressionModel(GPModel):
    """
    Abstract BNQD GP regression model.
    Inherits GPModel, which is an abstract class used for Bayesian prediction.
    """
    def __init__(
            self,
            kernel: Kernel,
            likelihood: Likelihood,
            mean_function: Optional[MeanFunction] = None,
            num_latent_gps: Optional[int] = 1
    ):
        # TODO: figure out what a latent variable is in this context (num_latent)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.is_trained = False

    def train(self, optimizer: Any = optimizers.Scipy(), verbose=True) -> None:
        """
        Trains the model.

        :param optimizer: Optimizer used for estimation of the optimal hyper parameters.
        :param verbose: Prints the model's summary if true.
        """
        # Uses the optimizer to minimize the objective_closure function, by adjusting the trainable variables.
        # The trainable variables are obtained by recursively finding fields of type Variable,
        # if and only if they're defined as being trainable.
        optimizer.minimize(self._training_loss, self.trainable_variables,
                           options=(dict(maxiter=MAX_OPTIMIZER_ITERATIONS)))

        if verbose:
            gf.utilities.print_summary(self)

        self.is_trained = True

    '''
    def __objective_closure(self) -> Tensor:
        """
        NOTE: Not used since GPflow.models.BayesianModel now has a _training_loss() method specified

        Function that the optimizer aims to minimize.

        :return: Negative log likelihood of the BNQD regression model.
        """
        return -self.maximum_log_likelihood_objective()
    '''

    @abstractmethod
    def plot_regression(self, n_samples=100, num_f_samples=5, plot_data=True, predict_y=False):
        """
        Plots the regression.

        :param plot_data: Plots the training data if true.
        :param n_samples: Number og x-samples used for the plot.
        :param num_f_samples: Number of samples of the latent function that are plotted.
        :param predict_y: Plots the prediction of new data points if true.
                          Plots the prediction of the latent function if otherwise.
        """
        raise NotImplementedError


class ContinuousModel(BNQDRegressionModel):
    """
    Continuous GP regression model.
    """
    def __init__(
            self,
            data: ContinuousData,
            model_or_kernel: Union[GPModel, Kernel],
            mean_function: Optional[MeanFunction] = None,
            num_latent_gps: Optional[int] = 1
    ):
        """
        :param data: Training data of type bnqdflow.data_types.ContinuousData
        :param model_or_kernel: Model or kernel object used for regression.
                                If a kernel is passed, a GPR object will be used.
        :param mean_function: Mean function used for the regression.
        :param num_latent_gps: Number of latent Gaussian processes.
        """
        # TODO: check if this data set shape stuff actually works
        self.data = tuple(map(util.ensure_tf_vector_format, data))

        assert len(data) == 2, \
            "The data should be a tuple in the shape of (x, y). i.e. gpflow.models.model.RegressionData"

        self.N = data[0].shape[0]  # nr. of data points
        if isinstance(model_or_kernel, Kernel):
            self.model = GPR(self.data, model_or_kernel)
        else:
            self.model = model_or_kernel

        super().__init__(self.model.kernel, self.model.likelihood, mean_function, num_latent_gps)

    def maximum_log_likelihood_objective(self, *args, **kwargs) -> Tensor:
        """
        Log likelihood of the continuous model.

        :param args:
        :param kwargs:
        :return: Log likelihood of the continuous model.
        """
        return self.model.maximum_log_likelihood_objective(*args, **kwargs)

    def log_posterior_density(self, method="bic", *args, **kwargs) -> Tensor:
        """
        Log marginal likelihood of the continuous model.
        This is done via one of two methods: using the BIC score, or with GPflow's native implementation.

        :param method: Method used for estimation of the log marginal likelihood. Either "bic" or "native"
        :return: Log marginal likelihood of the discontinuous model.
        """
        method = method.lower()

        if method in ["bic", "bic score", "BIC_score"]:
            # Parameters are represented as tuple, for documentation see gpf.Module
            k = len(self.trainable_parameters)
            L = self.maximum_log_likelihood_objective()
            BIC = L - k / 2 * np.log(self.N)
            return BIC

        elif method in ["native", "nat", "gpflow"]:
            return self.model.log_posterior_density(*args, **kwargs)

        else:
            raise ValueError(f"Incorrect method for log marginal likelihood calculation: {method}. "
                             "Please use either 'bic' or 'native' (i.e. gpflow method)")

    def predict_f(self, Xnew: Union[InputData, ndarray], full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Computes the mean and variance of the posterior latent function at the input points.

        :param Xnew: Input locations at which to compute the mean and variance
        :param full_cov: whether or not to return the full covariance matrix of the latent function
        :param full_output_cov:
        :return: Mean and variance of the posterior latent function.
        """
        return self.model.predict_f(Xnew, full_cov, full_output_cov)

    def plot_regression(self, n_samples=100, num_f_samples=5, plot_data=True, predict_y=False) -> None:
        """
        Plots the regression.

        :param plot_data: Plots the training data if true.
        :param n_samples: Number of x-samples used for the plot.
        :param num_f_samples: Number of samples of the latent function that are plotted.
        :param predict_y: Plots the prediction of new data points if true.
                          Plots the prediction of the latent function otherwise.
        """
        # Temporary value fo adding margins to the sides of the regression. Shows the flared-out probability.
        MARGIN = 1.3
        col = 'green'

        if plot_data:
            # Plots the training data
            x, y = self.data
            plt.plot(x[:, None], y[:, None], linestyle='none', marker='x', color='k', label='obs')

        # finds minimum and maximum x values
        x_vals = self.data[0]
        min_x, max_x = (min(x_vals[:, 0]) * MARGIN, max(x_vals[:, 0]) * MARGIN)

        # creates n_samples data points between min_x and max_x
        x_samples = np.linspace(min_x, max_x, n_samples)

        # Uses either self.predict_y or self.predict_f depending on whether or not predict_y is True
        predict = self.predict_y if predict_y else self.predict_f

        # finds the mean and variance for each element in x_samples
        mean, var = predict(x_samples[:, None])

        # Plots the 95% confidence interval
        plt.fill_between(x_samples, mean[:, 0] - 1.96 * np.sqrt(var[:, 0]), mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                         color=col, alpha=0.2)  # 0.5

        # Plots the mean function predicted by the GP
        plt.plot(x_samples, mean, c=col, label='$M_C$')

        if num_f_samples > 0 and not predict_y:
            # Plots samples of the latent function.
            # Only if num_f_samples > 0 and the latent function is plotted instead of
            # the prediction of held-out data points
            f_samples = self.predict_f_samples(x_samples[:, None], num_f_samples)
            for f_sample in f_samples:
                plt.plot(x_samples, f_sample[:, 0], linewidth=0.2, c=col)


class DiscontinuousModel(BNQDRegressionModel):
    """
    Discontinuous GP regression model.
    """

    def __init__(
            self,
            data: DiscontinuousData,
            model_or_kernel: Union[GPModel, Kernel],
            intervention_point: Tensor,
            share_params: bool = True,
            mean_function: MeanFunction = None,
            num_latent_gps: int = 1
    ):
        """
        :param data: Training data of type bnqdflow.data_types.ContinuousData
        :param model_or_kernel: Model or kernel object used for regression.
                                If a kernel is passed, a GPR object will be used.
        :param intervention_point: Input point at which to switch sub-models
        :param share_params: Whether or not the sub models have the same hyper parameters.
        :param mean_function: Mean function used for the regression.
        :param num_latent_gps: Number of latent Gaussian processes.
        """
        # TODO: check if this data set shape stuff actually works
        # Converts the input data to comply with tensorflow.matmul requirements. The data is split between control and
        # intervention data, and is stored as a list of tuples of tensors
        self.data = list(map(lambda section: tuple(map(util.ensure_tf_vector_format, section)), data))

        assert all(map(lambda section: len(section) == 2, self.data)), \
            "The data should be list of tuples in the shape of (x, y). i.e. List[gpflow.models.model.RegressionData]"

        self.share_params = share_params
        self.intervention_point = intervention_point
        self.N = np.shape(data[0][0])[0] + np.shape(data[1][0])[0]  # nr. of data points

        self.models = list()
        # Checks of the argument is a Kernel or GPModel
        if isinstance(model_or_kernel, Kernel):
            # Initializes the models as standard GPR objects, with the same kernel and mean_function objects
            for i in range(N_MODELS):
                self.models.append(GPR(self.data[i], model_or_kernel, mean_function, num_latent_gps))

        else:  # If it's a GPModel
            # Initializes the models as deep copies of the provided model
            for i in range(N_MODELS):
                self.models.append(gf.utilities.deepcopy(model_or_kernel))

        if self.share_params:
            # Makes all models use the same Kernel and Likelihood objects
            for i in range(1, N_MODELS):
                self.models[i].kernel = self.models[0].kernel
                self.models[i].likelihood = self.models[0].likelihood
                self.models[i].mean_function = self.models[0].mean_function
        else:
            # Ensures the Kernels of the models are different objects if a Kernel was provided instead of a GPModel
            if isinstance(model_or_kernel, Kernel):
                for i in range(N_MODELS):
                    self.models[i].kernel = gf.utilities.deepcopy(model_or_kernel)
                    if not (mean_function is None):
                        self.models[i].mean_function = gf.utilities.deepcopy(mean_function)

        if self.share_params:
            super().__init__(self.models[0].kernel, self.models[0].likelihood, mean_function, num_latent_gps)
        else:
            # Cannot pass the likelihood and kernel to the superclass.
            # This is because discontinuous model contains two sub-models, with their own kernel and likelihood objects.
            # And so far I haven't seen a way to create a new kernel and likelihood object that properly "splits"
            # TODO: figure this out maybe?
            super().__init__(None, None, mean_function=None, num_latent_gps=num_latent_gps)

    @property
    def control_model(self):
        return self.models[0]

    @property
    def intervention_model(self):
        return self.models[1]

    def old_equal_params(self):
        """
        Old method of sharing the hyper parameters between the control and intervention model.
        """
        control_params = gf.utilities.parameter_dict(self.control_model)
        intervention_params = gf.utilities.parameter_dict(self.intervention_model)
        new_params = {k1: (v1+v2)/2 for (k1, v1), (k2, v2) in zip(control_params.items(), intervention_params.items())}
        gf.utilities.multiple_assign(self.control_model, new_params)
        gf.utilities.multiple_assign(self.intervention_model, new_params)

    def maximum_log_likelihood_objective(self, *args, **kwargs) -> Tensor:
        """
        Log likelihood of the discontinuous model. Computed as the sum of all sub-models' log likelihoods.

        :param args:
        :param kwargs:
        :return: Log likelihood of the discontinuous model.
        """
        return tf.reduce_sum(list(map(lambda m: m.maximum_log_likelihood_objective(), self.models)))

    def log_posterior_density(self, method="bic", *args, **kwargs) -> Tensor:
        """
        Log marginal likelihood of the discontinuous model.
        This is done via one of two methods: using the BIC score, or with GPflow's native implementation.
        If using the native implementation, sums the marginal likelihoods of all sub-models.

        :param method: Method used for estimation of the log marginal likelihood. Either "bic" or "native"
        :return: Log marginal likelihood of the discontinuous model.
        """

        method = method.lower()

        if method in ["bic", "bic score", "BIC_score"]:
            k = len(self.trainable_variables)
            L = self.maximum_log_likelihood_objective()
            BIC = L - k / 2 * np.log(self.N)
            return BIC

        elif method in ["native", "nat", "gpflow"]:
            # Sums all log marginal likelihood of the sub-models.
            return tf.reduce_sum(list(map(lambda m: m.log_posterior_density(*args, **kwargs), self.models)))

        else:
            raise ValueError("Incorrect method for log marginal likelihood calculation: {}"
                             "Please use either 'bic' or 'native' (i.e. gpflow method)".format(method))

    def predict_f(self, Xnew: List[Union[InputData, ndarray]], full_cov=False, full_output_cov=False) -> List[MeanAndVariance]:
        """
        Computes the means and variances of the posterior latent functions of the sub models at the input points.

        :param Xnew: List of input locations at which to compute the means and variances.
        :param full_cov: whether or not to return the full covariance matrices of the latent functions.
        :param full_output_cov:
        :return: List of means and variances of the posterior latent functions.
        """
        assert len(Xnew) is len(self.models), \
            "The number of elements in Xnew should be the same as the number of sub-models. " \
            "Each element in the list of input data is predicted by one model each."

        Xnew = list(map(util.ensure_tf_vector_format, Xnew))

        res = list()
        for model, section in zip(self.models, Xnew):
            res.append(model.predict_f(section, full_cov, full_output_cov))

        return res

    def predict_f_samples(self, Xnew: List[Union[InputData, ndarray]], num_samples: Optional[int] = None,
                          full_cov: bool = True, full_output_cov: bool = False) -> List[Tensor]:
        """
        Produce samples from the posterior latent function(s) at the input points.

        :param Xnew: List of input locations at which to draw samples.
        :param num_samples: Number of samples to draw.
        :param full_cov: If True, draw correlated samples over the inputs. If False, draw samples that are uncorrelated
                         over the inputs.
        :param full_output_cov: If True, draw correlated samples over the outputs. If False, draw samples that are
                                uncorrelated over the outputs.
        :return: List of samples.
        """
        assert len(Xnew) is len(self.models), \
            "The number of elements in Xnew should be the same as the number of sub-models. " \
            "Each element in the list of input data is predicted by one model each."

        Xnew = list(map(util.ensure_tf_vector_format, Xnew))

        res = list()
        for model, section in zip(self.models, Xnew):
            res.append(model.predict_f_samples(section, num_samples, full_cov, full_output_cov))

        return res

    def predict_y(self, Xnew: List[Union[InputData, ndarray]], full_cov: bool = False,
                  full_output_cov: bool = False) -> List[MeanAndVariance]:
        """
        Compute the mean and variance of the held-out data at the input points.

        :param Xnew: List of input locations at which to compute the means and variances.
        :param full_cov: whether or not to return the full covariance matrices of the latent functions.
        :param full_output_cov:
        :return: List of means and variances of the held-out data points.
        """
        assert len(Xnew) is len(self.models), \
            "The number of elements in Xnew should be the same as the number of sub-models. " \
            "Each element in the list of input data is predicted by one model each."

        Xnew = list(map(util.ensure_tf_vector_format, Xnew))

        res = list()
        for model, section in zip([self.control_model, self.intervention_model], Xnew):
            res.append(model.predict_y(section, full_cov, full_output_cov))

        return res

    def predict_log_density(self, data: DiscontinuousData, full_cov: bool = False, full_output_cov: bool = False):
        """
        Compute the log densities of the data at the new data points.

        :param data: List of RegressionData (i.e. tuples of shape (x, y)) for which to compute the log densities.
        :param full_cov:
        :param full_output_cov:
        :return: List of predicted log densities.
        """
        assert len(data) is len(self.models), \
            "The number of elements in Xnew should be the same as the number of sub-models. " \
            "Each element in the list of input data is predicted by one model each."

        res = list()
        for model, section in zip([self.control_model, self.intervention_model], data):
            res.append(model.predict_log_density(section, full_cov, full_output_cov))

        return res

    def plot_regression(self, n_samples=100, num_f_samples=5, plot_data=True, predict_y=False):
        """
        Plots the regressions of the sub-models.

        :param plot_data: Plots the training data if true.
        :param n_samples: Number of x-samples used for the plot.
        :param num_f_samples: Number of samples of the latent function that are plotted.
        :param predict_y: Plots the prediction of new data points if true.
                          Plots the prediction of the latent function otherwise.
        """
        # TODO: make this plotting function not so shitty

        # Temporary value fo adding margins to the sides of the regression. Shows the flared-out probability.
        MARGIN = 1.3
        col = 'blue'
        markers = ['x', '+', '.', '*', 'd', 'v', 's', 'p', 'X', 'P', 'h']

        # Plots the vertical intervention point line
        plt.axvline(x=self.intervention_point, linestyle='--', c='k')

        if plot_data:
            # Plots the training data
            for i, ((x, y), m) in enumerate(zip(self.data, cycle(markers))):
                plt.plot(x[:, 0], y[:, 0], linestyle='none', marker=m, color='k', label=f'$obs_{i}$')

        # Finds minimum and maximum x values
        x_vals = tf.concat(list(map(lambda section: section[0], self.data)), 0)
        min_x, max_x = (min(x_vals[:, 0]) * MARGIN, max(x_vals[:, 0]) * MARGIN)
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

        # Uses either self.predict_y or self.predict_f depending on whether or not predict_y is True
        predict = self.predict_y if predict_y else self.predict_f

        # Predicts the means and variances for both x_samples
        means_and_vars = predict(x_samples_list)

        # Quick fix to ensure only a single label occurs in the pyplot legend
        # TODO: make this more elegant
        labeled = False

        for x_samples, (mean, var) in zip(x_samples_list, means_and_vars):
            # Plots the 95% confidence interval
            plt.fill_between(x_samples[:, 0], mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                             mean[:, 0] + 1.96 * np.sqrt(var[:, 0]), color=col, alpha=0.2)

            # Plots the mean function predicted by the GP
            plt.plot(x_samples[:, 0], mean[:, 0], c=col, label=('$M_D$' if not labeled else ""))
            labeled = True

        if num_f_samples > 0 and not predict_y:
            # Plots samples of the latent function.
            # Only if num_f_samples > 0 and the latent function is plotted instead of
            # the prediction of held-out data points
            f_samples_list = self.predict_f_samples(x_samples_list, num_f_samples)
            for f_samples, x_samples in zip(f_samples_list, x_samples_list):
                for f_sample in f_samples:
                    plt.plot(x_samples, f_sample[:, 0], linewidth=0.2, c=col)

