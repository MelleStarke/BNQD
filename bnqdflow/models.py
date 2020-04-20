import numpy as np
import matplotlib.pyplot as plt
import gpflow as gf
import warnings
import bnqdflow.util as util
import tensorflow as tf

from typing import Optional, Tuple, Union, List, Any

from numpy import ndarray

from tensorflow import Tensor

from itertools import cycle

from gpflow import optimizers, Module
from gpflow.kernels import Kernel
from gpflow.models import GPModel, GPR
from gpflow.models.model import InputData, MeanAndVariance, RegressionData

##############################
###### Global Constants ######
##############################

counter = 0

MAX_OPTIMIZER_ITERATIONS = 100
N_MODELS = 2  # Nr. of sub-models in the discontinuous model


class GPMContainer(Module):
    """
    Generic container for GPModel objects.

    Computes all statistics related to the data accordingly.
    (e.g. log likelihood is computed as the sum of the likelihood of each contained GPModel over its respective data)

    Every functioality related to prediction uses lists, where the index of each element corresponds to the index of
    the GPModel that is used.

    Due to the generic nature of this class, it can be adapted to support more than two GPModels and multiple
    intervention points.

    More user-friendly classes are defined below this class (i.e. ContinuousModel and DiscontinuousModel)
    """

    def __init__(
            self,
            regression_source: Union[Kernel, GPModel, List[GPModel]],
            data_list: Optional[List[RegressionData]] = None,
            intervention_points: Optional[List[Tensor]] = None,
            share_params: Optional[bool] = True
    ):
        super().__init__()

        assert isinstance(regression_source, (Kernel, GPModel)) \
               or (isinstance(regression_source, list) and isinstance(regression_source[0], GPModel)), \
            "The regression_source objects needs to be an instance of either a Kernel or a GPModel, " \
            "or a list of GPModel objects."

        # Initializes the list of models
        if isinstance(regression_source, list):
            # Sets the list of models as the regression_source if it's a list of GPModel objects
            self.models = regression_source
        else:
            # Generates a list of GPModel objects if otherwise
            assert len(data_list) > 0, \
                "The list of RegressionData cannot be empty."

            self.models = self.generate_gp_models(regression_source, data_list)

        # Sets the list of intervention points
        self.intervention_points = intervention_points if intervention_points else []
        assert self.n_models == (len(self.intervention_points) + 1), \
            f"The number of GPModel objects contained in GPMContainer ({self.n_models}) should be one higher " \
            f"than the number of intervention points ({len(self.intervention_points)})."

        # Only applicable to GPMContainers with more than one model (i.e. discontinuous)
        if self.n_models > 1:
            # The names of the parameters that should be shared or kept separated.
            # These can be different depending on what GPModel implementation is used.
            # The following parameters are always used in GPModel implementations.
            applicable_params = ['kernel', 'likelihood', 'mean_function']

            # Tries to access the inducing_variable field.
            # If it exists, add 'inducing_variable' to the list of applicable parameters.
            try:
                temp = self.models[0].inducing_variable
                applicable_params.append('inducing_variable')
            except AttributeError:
                pass

            if share_params:
                # Ensures all models use the same hyper parameter objects
                self._ensure_same_params(applicable_params)
            else:
                # Ensures all models use different hyper parameter objects
                self._ensure_different_params(applicable_params)

        else:
            # If the GPMContainer only contains one model, intialize the list of intervention points as empty.
            # It's better to do this here than by using a default value, because that would make the parameter mutable.
            self.intervention_points = []

        self.is_trained = False

    @staticmethod
    def generate_gp_models(
            model_or_kernel: Union[GPModel, Kernel],
            data_list: List[RegressionData]
    ):
        """
        Generates a list of GPModel objects with the same length as data_list.

        If a GPModel object was passed, the list will consist of deep copies of the GPModel, with the data reassigned.
        If a Kernel was passed, the list will consist of GPR (all containing the Kernel) instead.

        :param model_or_kernel: GPModel or Kernel object used to generate the list of models
        :param data_list: List of RegressionData. Each model will get one element.
        :return:
        """
        assert isinstance(model_or_kernel, (Kernel, GPModel)), \
            "The regression_source object needs to be an instance of either a Kernel or a GPModel, "
        assert all(map(lambda data: type(data) is tuple and len(data) is 2, data_list)), \
            "data_list should be a list of tuples of length 2 (i.e. a list of RegressionData)"

        is_kernel = isinstance(model_or_kernel, Kernel)

        models = list()
        for data in data_list:
            # Ensures both the InputData and OutputData are in a format usable by tensorflow
            data = tuple(map(util.ensure_tf_matrix, data))

            if is_kernel:
                # Appends a GPR object to the list of models if a Kernel was passed instead of a GPModel
                models.append(GPR(data, model_or_kernel))

            else:
                # Appends a deepcopy of the passed GPModel to the list of models
                model = gf.utilities.deepcopy(model_or_kernel)
                model.data = data
                models.append(model)

        return models

    @property
    def n_models(self) -> int:
        """
        The number of models contained in the class.
        """
        return len(self.models)

    @property
    def is_continuous(self) -> bool:
        """
        Returns whether or not the GPMContainer only contains one GPModel. Which would make it continuous
        """
        return len(self.models) == 1

    @property
    def share_params(self):
        """
        Whether or not the models share hyper parameters.

        Done by checking if all parameters of the contained models refer use the same pointers as the first model.
        :return:
        """
        if self.n_models < 2:
            warnings.warn("The GPMContainer contains less then two models. Therefore, parameters cannot be shared "
                          "between models by definition. share_params will return True by default in this case.")
            return True

        m0_params = self.models[0].parameters
        return all(map(lambda m: all(map(lambda x: x[0] is x[1],
                                         zip(m.parameters, m0_params))),
                       self.models[1:self.n_models]))

    @property
    def data_list(self) -> List[RegressionData]:
        """
        Collects all data objects of the contained models.

        :return: List of data of the models.
        """
        return list(map(lambda m: m.data, self.models))

    @property
    def kernel(self):
        if self.n_models < 2 or self.share_params:
            return self.models[0].kernel

        warnings.warn("The models contained in the GPMContainer don't share parameters. Therefore, a single kernel "
                      "object cannot be returned. You can call the kernel of s specific model via e.g. "
                      "container.models[0].kernel")
        return None

    @property
    def likelihood(self):
        if self.n_models < 2 or self.share_params:
            return self.models[0].likelihood

        warnings.warn("The models contained in the GPMContainer don't share parameters. Therefore, a single likelihood "
                      "object cannot be returned. You can call the likelihood of s specific model via e.g. "
                      "container.models[0].likelihood")
        return None

    @property
    def mean_function(self):
        if self.n_models < 2 or self.share_params:
            return self.models[0].mean_function

        warnings.warn("The models contained in the GPMContainer don't share parameters. Therefore, a single "
                      "mean_function object cannot be returned. You can call the kernel of s specific model via e.g. "
                      "container.models[0].mean_function")
        return None

    def maximum_log_likelihood_objective(self, *args, **kwargs) -> Tensor:
        """
        Combined log likelihood of the contained models over their respective data.

        This can be written as a sum since log(a) + log(b) = log(a * b).

        :param args:
        :param kwargs:
        :return: Total log likelihood of the GPMContainer.
        """
        return tf.reduce_sum(list(map(lambda m: m.maximum_log_likelihood_objective(*args, **kwargs), self.models)),
                             0)

    def log_posterior_density(self, method="bic", *args, **kwargs) -> Tensor:
        """
        Combined log marginal likelihood of the contained models over their respective data.
        This is done via one of two methods: using the BIC score, or with GPflow's native implementation.

        :param method: Method used for estimation of the log marginal likelihood. Either "bic" or "native"
        :return: Total log marginal likelihood of GPMContainer.
        """
        method = method.lower()

        if method in ["bic", "bic score", "bic_score"]:
            k = len(self.trainable_parameters)
            L = self.maximum_log_likelihood_objective()
            N = sum(map(lambda data: len(data[0]), self.data_list))
            BIC = L - k / 2 * np.log(N)
            return BIC

        elif method in ["native", "nat", "gpflow"]:
            return tf.reduce_sum(list(map(lambda m: m.log_posterior_density(*args, **kwargs), self.models)), 0)

        else:
            raise ValueError(f"Incorrect method for log marginal likelihood calculation: {method}. "
                             "Please use either 'bic' or 'native' (i.e. gpflow method)")

    def _training_loss(self, *args, **kwargs) -> Tensor:
        """
        Function that the optimizer aims to minimize.

        :return: Negative log likelihood of the BNQD regression model.
        """
        return tf.reduce_sum(list(map(lambda m: m._training_loss(*args, **kwargs), self.models)), 0)

    def train(self, optimizer: Any = optimizers.Scipy(), verbose=True) -> None:
        """
        Trains all contained models.

        This is done by optimizing all trainable variables found in the GPMContainer, according to the combined
        training loss of all contained models.

        :param optimizer: Optimizer used for estimation of the optimal hyper parameters.
        :param verbose: Prints the model's summary if true.
        """
        # Uses the optimizer to minimize the objective_closure function, by adjusting the trainable variables.
        # The trainable variables are obtained by recursively finding fields of type Variable,
        # if and only if they're defined as being trainable.
        optimizer.minimize(self._training_loss, self.trainable_variables)

        if verbose:
            gf.utilities.print_summary(self)

        self.is_trained = True

    def predict_f(self, Xnew_list: List[Union[InputData, ndarray]], full_cov=False, full_output_cov=False) \
            -> List[MeanAndVariance]:
        """
        Computes the means and variances of the posterior latent functions of the contained models at the input points.

        Each element in the list of input points will be given to the corresponding GPModel. Therefore, the length of
        Xnew_list should be the same as the number of contained models. If you wish to, for example, only use the 2nd
        model, and the GPMContainer contains two models, Xnew_list should look something like: [[], [0.2, 0.3, 0.4]].
        This will produce the predicted mean and variance of the 2nd model at input points 0.2, 0.3, and 0.4.

        :param Xnew_list: List of input locations at which to compute the means and variances.
        :param full_cov: whether or not to return the full covariance matrices of the latent functions.
        :param full_output_cov:
        :return: List of means and variances of the posterior latent functions.
        """
        assert len(Xnew_list) is len(self.models), \
            f"The number of elements in Xnew_list ({len(Xnew_list)}) should be the same as the number of sub-models " \
            f"({self.n_models}). Each element in the list of input data is predicted by one model each."

        res = list()
        for model, section in zip(self.models, Xnew_list):
            if ((-1,) + np.shape(section))[-1] == 0:
                # Adds an empty element to the result list if the section is empty.
                # Necessary since gpflow doesn't allow prediction of empty InputData.
                res.append([])
            else:
                res.append(model.predict_f(util.ensure_tf_matrix(section), full_cov, full_output_cov))

        return res

    def predict_f_samples(self, Xnew_list: List[Union[InputData, ndarray]], num_samples: Optional[int] = None,
                          full_cov: bool = True, full_output_cov: bool = False) -> List[Tensor]:
        """
        Produce a list of samples from the posterior latent function(s) at the input points.

        Each element in the list of input points will be given to the corresponding GPModel. Therefore, the length of
        Xnew_list should be the same as the number of contained models. If you wish to, for example, only use the 2nd
        model, and the GPMContainer contains two models, Xnew_list should look something like: [[], [0.2, 0.3, 0.4]].
        This will produce samples of the 2nd model at input points 0.2, 0.3, and 0.4.

        :param Xnew_list: List of input locations at which to draw samples.
        :param num_samples: Number of samples to draw.
        :param full_cov: If True, draw correlated samples over the inputs. If False, draw samples that are uncorrelated
                         over the inputs.
        :param full_output_cov: If True, draw correlated samples over the outputs. If False, draw samples that are
                                uncorrelated over the outputs.
        :return: List of samples.
        """
        assert len(Xnew_list) is len(self.models), \
            "The number of elements in Xnew_list should be the same as the number of sub-models. " \
            "Each element in the list of input data is predicted by one model each."

        res = list()
        for model, section in zip(self.models, Xnew_list):
            if ((-1,) + np.shape(section))[-1] == 0:
                # Adds an empty element to the result list if the section is empty.
                # Necessary since gpflow doesn't allow prediction of empty InputData.
                res.append([])
            else:
                res.append(model.predict_f_samples(util.ensure_tf_matrix(section), num_samples, full_cov,
                                                   full_output_cov))

        return res

    def predict_y(self, Xnew_list: List[Union[InputData, ndarray]], full_cov: bool = False,
                  full_output_cov: bool = False) -> List[MeanAndVariance]:
        """
        Compute the mean and variance of the held-out data at the input points.

        Each element in the list of input points will be given to the corresponding GPModel. Therefore, the length of
        Xnew_list should be the same as the number of contained models. If you wish to, for example, only use the 2nd
        model, and the GPMContainer contains two models, Xnew_list should look something like: [[], [0.2, 0.3, 0.4]].
        This will produce the predicted mean and variance of the 2nd model at input points 0.2, 0.3, and 0.4.

        :param Xnew_list: List of input locations at which to compute the means and variances.
        :param full_cov: whether or not to return the full covariance matrices of the latent functions.
        :param full_output_cov:
        :return: List of means and variances of the held-out data points.
        """
        assert len(Xnew_list) is len(self.models), \
            "The number of elements in Xnew_list should be the same as the number of sub-models. " \
            "Each element in the list of input data is predicted by one model each."

        res = list()
        for model, section in zip(self.models, Xnew_list):
            if ((-1,) + np.shape(section))[-1] == 0:
                # Adds an empty element to the result list if the section is empty.
                # Necessary since gpflow doesn't allow prediction of empty InputData.
                res.append([])
            else:
                res.append(model.predict_y(util.ensure_tf_matrix(section), full_cov, full_output_cov))

        return res

    def predict_log_density(self, data_list: List[RegressionData], full_cov: bool = False,
                            full_output_cov: bool = False):
        """
        Compute the log densities of the data at the new data points.

        Each element in the list of input points will be given to the corresponding GPModel. Therefore, the length of
        data_list should be the same as the number of contained models. If you wish to, for example, only use the 2nd
        model, and the GPMContainer contains two models, data_list should look something like:
        [[], ([0.2, 0.3, 0.4], [2.0, 2.0, 2.0])]. This will produce the predicted log density of the 2nd model at
        coordinates (0.2, 2.0), (0.3, 2.0), and (0.4, 2.0).

        :param data_list: List of RegressionData (i.e. tuples of shape (x, y)) for which to compute the log densities.
        :param full_cov:
        :param full_output_cov:
        :return: List of predicted log densities.
        """
        assert len(data_list) is len(self.models), \
            "The number of elements in data_list should be the same as the number of sub-models. " \
            "Each element in the list of input data is predicted by one model each."
        assert all(map(lambda data: len(data) is 0 or (len(data) is 2 and len(data[0]) is len(data[1])), data_list)), \
            "The list of data should consist of either empty lists (where you don't want predictions to be made), " \
            "or tuples of size 2, where both elements have the same length."

        res = list()
        for model, section in zip(self.models, data_list):
            if ((-1,) + np.shape(section))[-1] == 0:
                # Adds an empty element to the result list if the section is empty.
                # Necessary since gpflow doesn't allow prediction of empty InputData.
                res.append([])
            else:
                section = tuple(map(util.ensure_tf_matrix, section))
                res.append(model.predict_log_density(section, full_cov, full_output_cov))

        return res

    def plot_regression(self, n_samples=100, padding: Union[float, Tuple[float]] = 0.2, num_f_samples=5, plot_data=True,
                        predict_y=False):
        """
        Plots the regressions of the models.

        :param padding: Proportion of the x-range that is added to the sides of the plot.
                        Can also be a tuple to allow for different paddings on the left and right.
        :param plot_data: Plots the training data if true.
        :param n_samples: Number of x-samples used for the plot.
        :param num_f_samples: Number of samples of the latent function that are plotted.
        :param predict_y: Plots the prediction of new data points if true.
                          Plots the prediction of the latent function otherwise.
        """
        colours = ['green', 'blue', 'red', 'cyan', 'magenta', 'yellow', 'orange', 'purple']
        col = colours[(self.n_models - 1) % len(colours)]
        markers = ['x', '+', '.', '*', 'd', 'v', 's', 'p', 'X', 'P', 'h']

        # Formats the margin as a tuple with duplicate elements.
        # Ensures the code works for a single float as well as a tuple of two floats.
        if type(padding) is not tuple:
            padding = (padding, padding)

        # Plots the vertical intervention point line(s).
        for ip in self.intervention_points:
            plt.axvline(ip, linestyle='--', c='k')

        if plot_data:
            # Plots the training data with different markers per section.
            for i, ((x, y), m) in enumerate(zip(self.data_list, cycle(markers))):
                plt.plot(x[:, 0], y[:, 0], linestyle='none', marker=m, color='k', label=f'$obs_{i}$')

        # Finds minimum and maximum x values.
        x_vals = tf.concat(list(map(lambda section: section[0], self.data_list)), 0)
        min_x, max_x = (min(x_vals[:, 0]), max(x_vals[:, 0]))
        x_range = max_x - min_x
        min_x, max_x = (min_x - (x_range * padding[0]), max_x + (x_range * padding[1]))

        # List of intervention points plus the min and max x-value. Used to calculate the x-samples list.
        separations = [min_x] + self.intervention_points + [max_x]
        # List of x-samples used for plotting. Each x-sample is plotted using its respective model.
        x_samples_list = list()

        for i in range(len(separations) - 1):
            # Left-most x-value of the section
            left_bound = separations[i]
            # Right-most x-value of the section
            right_bound = separations[i + 1]

            section_ratio = (right_bound - left_bound) / x_range
            # Number of x-samples in the section.
            # Incremented by one to account for integer conversion and the overlapping of bounds between sections.
            section_samples = int(n_samples * section_ratio + 1)
            x_samples_list.append(np.linspace(left_bound, right_bound, section_samples))

        # Which prediction function to use. Depends on the value of predict_y.
        predict = self.predict_y if predict_y else self.predict_f

        # Predicts the means and variances for both x_samples
        means_and_vars = predict(x_samples_list)

        # Ensures only a single label occurs in the pyplot legend
        labeled = False

        for x_samples, (mean, var) in zip(x_samples_list, means_and_vars):
            # Plots the 95% confidence interval
            plt.fill_between(x_samples, mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                             mean[:, 0] + 1.96 * np.sqrt(var[:, 0]), color=col, alpha=0.2)

            # Plots the mean function predicted by the GP
            plt.plot(x_samples, mean[:, 0], c=col, label=('$M_D$' if not labeled else ""))
            labeled = True

        if num_f_samples > 0 and not predict_y:
            # Plots samples of the latent function.
            # Only if num_f_samples > 0 and the latent function is plotted instead of
            # the prediction of held-out data points
            f_samples_list = self.predict_f_samples(x_samples_list, num_f_samples)
            for f_samples, x_samples in zip(f_samples_list, x_samples_list):
                for f_sample in f_samples:
                    plt.plot(x_samples, f_sample[:, 0], linewidth=0.2, c=col)

    def _ensure_same_params(self, params: List[str]) -> None:
        """
        Sets all parameters of the models corresponding to the 'params' list to be the same object.

        Currently, the only options are 'kernel', 'likelihood', 'mean_function', and 'inducing_variable'

        :param params: List of strings. Each string corresponds to a parameter.
        """
        for p in params:
            p = p.lower()

            if p in ['k', 'kern', 'kernel']:
                for i in range(1, self.n_models):
                    self.models[i].kernel = self.models[0].kernel

            elif p in ['l', 'lik', 'likelihood']:
                for i in range(1, self.n_models):
                    self.models[i].likelihood = self.models[0].likelihood

            elif p in ['mf', 'mean', 'mean function', 'mean_function']:
                for i in range(1, self.n_models):
                    self.models[i].mean_function = self.models[0].mean_function

            elif p in ['iv', 'ind var', 'inducing variable', 'inducing_variable']:
                for i in range(1, self.n_models):
                    self.models[i].inducing_variable = self.models[0].inducing_variable

            else:
                warnings.warn(f"'{p}' is not a valid name of a parameter that can be shared.")

    def _ensure_different_params(self, params: List[str]) -> None:
        """
        Sets all parameters of the models corresponding to the 'params' list to be the different objects.

        Currently, the only options are 'kernel', 'likelihood', 'mean_function', and 'inducing_variable'

        :param params: List of strings. Each string corresponds to a parameter.
        """
        for p in params:
            p = p.lower()

            if p in ['k', 'kern', 'kernel']:
                for i in range(1, self.n_models):
                    if self.models[i].kernel is self.models[0].kernel:
                        self.models[i].kernel = gf.utilities.deepcopy(self.models[0].kernel)

            elif p in ['l', 'lik', 'likelihood']:
                for i in range(1, self.n_models):
                    if self.models[i].likelihood is self.models[0].likelihood:
                        self.models[i].likelihood = gf.utilities.deepcopy(self.models[0].likelihood)

            elif p in ['mf', 'mean', 'mean function', 'mean_function']:
                for i in range(1, self.n_models):
                    if self.models[i].mean_function is self.models[0].mean_function:
                        self.models[i].mean_function = gf.utilities.deepcopy(self.models[0].mean_function)

            elif p in ['iv', 'ind var', 'inducing variable', 'inducing_variable']:
                for i in range(1, self.n_models):
                    if self.models[i].inducing_variable is self.models[0].inducing_variable:
                        self.models[i].inducing_variable = gf.utilities.deepcopy(self.models[0].inducing_variable)

            else:
                warnings.warn(f"'{p}' is not a valid name of a parameter that can be shared.")


class ContinuousModel(GPMContainer):
    """
    Simplification of a GPMContainer with only one model.

    All inputs and outputs are adjusted accordingly.
    """

    def __init__(
            self,
            model_or_kernel: Union[GPModel, Kernel],
            data: Optional[RegressionData] = None
    ):
        """
        :param data: Data used for GP regression.
        :param model_or_kernel: Model or kernel object used for regression.
                                If a kernel is passed, a GPR object will be generated.
        :param mean_function: Mean function used for the regression.
        :param num_latent_gps: Number of latent Gaussian processes.
        """
        super().__init__(model_or_kernel, [data], intervention_points=[])

    @property
    def model(self):
        return self.models[0]

    @property
    def data(self):
        return self.data_list[0]

    def predict_f(self, Xnew: Union[InputData, ndarray], full_cov=False, full_output_cov=False) -> MeanAndVariance:
        return super().predict_f([Xnew], full_cov, full_output_cov)[0]

    def predict_f_samples(self, Xnew: Union[InputData, ndarray], num_samples: Optional[int] = None,
                          full_cov: bool = True, full_output_cov: bool = False) -> Tensor:
        return super().predict_f_samples([Xnew], num_samples, full_cov, full_output_cov)[0]

    def predict_y(self, Xnew: Union[InputData, ndarray], full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        return super().predict_y([Xnew], full_cov, full_output_cov)[0]

    def predict_log_density(self, data: RegressionData, full_cov: bool = False, full_output_cov: bool = False):
        return super().predict_log_density([data], full_cov, full_output_cov)[0]

    def plot_regression(self, n_samples=100, padding: Union[float, Tuple[float]] = 0.2, num_f_samples=5, plot_data=True,
                        predict_y=False):
        """
        Plots the regressions of the model.

        :param padding: Proportion of the x-range that is added to the sides of the plot.
                        Can also be a tuple to allow for different paddings on the left and right.
        :param plot_data: Plots the training data if true.
        :param n_samples: Number of x-samples used for the plot.
        :param num_f_samples: Number of samples of the latent function that are plotted.
        :param predict_y: Plots the prediction of new data points if true.
                          Plots the prediction of the latent function otherwise.
        """
        colours = ['green', 'blue', 'red', 'cyan', 'magenta', 'yellow', 'orange', 'purple']
        col = colours[(self.n_models - 1) % len(colours)]
        markers = ['x', '+', '.', '*', 'd', 'v', 's', 'p', 'X', 'P', 'h']

        # Formats the margin as a tuple with duplicate elements.
        # Ensures the code works for a single float as well as a tuple of two floats.
        if type(padding) is not tuple:
            padding = (padding, padding)

        if plot_data:
            x, y = self.data
            plt.plot(x[:, 0], y[:, 0], linestyle='none', marker=markers[0], color='k', label=f'$obs$')

        # Finds minimum and maximum x values.
        x_vals = self.data[0]
        min_x, max_x = (min(x_vals[:, 0]), max(x_vals[:, 0]))
        x_range = max_x - min_x
        min_x, max_x = (min_x - (x_range * padding[0]), max_x + (x_range * padding[1]))
        x_samples = np.linspace(min_x, max_x, n_samples)

        # Which prediction function to use. Depends on the value of predict_y.
        predict = self.predict_y if predict_y else self.predict_f

        # Predicts the means and variances for both x_samples
        mean, var = predict(x_samples)

        # Plots the 95% confidence interval
        plt.fill_between(x_samples, mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                         mean[:, 0] + 1.96 * np.sqrt(var[:, 0]), color=col, alpha=0.2)

        # Plots the mean function predicted by the GP
        plt.plot(x_samples, mean[:, 0], c=col, label='$M_C$')

        if num_f_samples > 0 and not predict_y:
            # Plots samples of the latent function.
            # Only if num_f_samples > 0 and the latent function is plotted instead of
            # the prediction of held-out data points
            f_samples = self.predict_f_samples(x_samples, num_f_samples)
            for f_sample in f_samples:
                plt.plot(x_samples, f_sample[:, 0], linewidth=0.2, c=col)


class DiscontinuousModel(GPMContainer):
    """
    Simplification of a GPMContainer with two models.
    """

    def __init__(
            self,
            model_or_kernel: Union[GPModel, Kernel],
            data: Optional[List[RegressionData]],
            intervention_point: Tensor,
            share_params: Optional[bool] = True
    ):
        """
        :param data: List of data used for GP regression.
        :param model_or_kernel: Model or kernel object used for regression.
                                If a kernel is passed, a GPR object will be used.
        :param intervention_point: Input point at which to switch sub-models
        :param share_params: Whether or not the sub models have the same hyper parameters.
        """
        super().__init__(model_or_kernel, data, intervention_points=[intervention_point], share_params=share_params)

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
        new_params = {k1: (v1 + v2) / 2 for (k1, v1), (k2, v2) in
                      zip(control_params.items(), intervention_params.items())}
        gf.utilities.multiple_assign(self.control_model, new_params)
        gf.utilities.multiple_assign(self.intervention_model, new_params)