import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import numpy as np
import gpflow as gf
import bnqdflow as bf
import math

from typing import Union, List, Callable, Any, Tuple, Optional, Dict

from tensorflow import Tensor

from gpflow.kernels import Kernel
from gpflow.models.model import GPModel, InputData
from gpflow.likelihoods import Likelihood, Gaussian

from bnqdflow.models import ContinuousModel, DiscontinuousModel, GPMContainer
from bnqdflow import util, ContinuousData, DiscontinuousData, USE_CUSTOM_KERNEL_COPY_FUNCTION

from copy import deepcopy

class Analysis(tf.Module):
    """
    Abstract analysis class
    """

    def __init__(
            self,
            data: Union[ContinuousData, DiscontinuousData],
            intervention_point: Tensor,
            share_params: bool = True,
            optimizer: Any = None,
            effect_size_measure=None):

        super().__init__()

        self.__cont_data = None
        self.__disc_data = None
        self.cont_data = None
        self.disc_data = None

        if util.is_continuous_data(data):
            self.cont_data = data
        else:
            self.disc_data = data

        self.intervention_point = intervention_point
        self.shared_params = share_params
        self.optimizer = optimizer

        self.effect_size_measure = effect_size_measure

    @property
    def shared_params(self):
        return self.__shared_params

    @shared_params.setter
    def shared_params(self, val):
        self.__shared_params = val

    def set_effect_size_measure(self, measure):
        """
        Sets the effect size measure.
        :param measure: Measure of type EffectSizeMeasure
        """
        self.effect_size_measure = measure


class SimpleAnalysis(Analysis):
    """
    Simple BNQD analysis object.
    This contains both the continuous and the discontinuous model,
    as well as some methods for manipulating and comparing these models.
    """

    def __init__(
            self,
            regression_source: Union[Kernel, GPModel, Tuple[GPMContainer, GPMContainer]],
            data: Union[ContinuousData, DiscontinuousData],
            intervention_point: Tensor,
            likelihood: Optional[Likelihood] = Gaussian(),
            share_params: Optional[bool] = True,
            gpm_type: Optional[str] = 'gpr',
            inducing_var_ratio: Optional[Union[Tuple[float, ...], float]] = 0.01,
            optimizer: Optional[object] = gf.optimizers.Scipy(),
            effect_size_measure: Optional[object] = None):
        """
        :param data:
                    Data object used to train the models. Either a tuple of tensors, or a list of tuples of tensors.
        :param regression_source:
                    Object used for regression. Can be a pre-made list of BNQDRegressionModels. If it's a Kernel or
                    GPModel, the continuous and discontinuous model will be generated from it.
        :param intervention_point:
                    Point at which the effect size is measured, and where the discontinuous model changes sub-model.
        :param share_params:
                    Whether or not the discontinuous sub-models share hyper parameters.
        :param marginal_likelihood_method:
                    Method used to compute the marginal likelihood. Can be either BIC-score or the native GPflow method.
        :param optimizer:
                    Optimizer used for estimating the hyper parameters.
        """
        # TODO: make the class able to handle ContinuousData, and then split it using the intervention_point
        self.cont_m = None
        self.disc_m = None

        super().__init__(data, intervention_point, share_params, optimizer, effect_size_measure)

        if isinstance(regression_source, tuple):
            assert len(regression_source) == 2, "Exactly two GPMContainers should be provided."
            cm, dm = regression_source
            assert cm.n_kernels == 1, "The first GPMContainer should contain 1 model"
            assert dm.n_kernels == 2, "The second GPMContainer should contain 2 models"

            self.cont_m = cm
            self.disc_m = dm

        else:
            self.__regression_source = regression_source

        self.summary_object = None
        self.marginal_likelihood_method = None

        self.__container_kwargs = {
            'likelihood': deepcopy(likelihood),
            'gpm_type': gpm_type,
            'inducing_var_ratio': inducing_var_ratio,
            'share_params': self.shared_params,
        }

    def init_models(
            self,
            labeler: Union[List[InputData], Callable[[InputData], int]] = None) -> None:
        """
        Initializes the continuous and discontinuous model.
        Uses the cont_data and disc_data fields if they exist, and attempts to generate it if otherwise.

        For generation of the discontinuous data, a labeler should be specified. This can either be a list of data
        points, or a function that takes a data point and returns an int (0 or 1) representing whether or not that data
        point belongs to the control model or intervention model.
        If no labeler is specified, splitting is done on the intervention point.

        :param labeler: either a list of data points, or a function of type InputData -> int
        """
        # Ensures the function is only executed if either of the two models is None
        if self.cont_m is not None and self.disc_m is not None:
            warnings.warn("Both models have already been initialized", category=UserWarning)
            return

        # Generates the continuous data. But only if necessary and possible.
        if self.cont_data is None:
            if self.disc_data is None:
                raise RuntimeError("No data has been passed to the analysis object.")
            else:
                # Converts discontinuous data into continuous data
                self.cont_data = util.flatten_data(self.disc_data)

        # Generates the discontinuous data. But only if necessary and possible.
        elif self.disc_data is None:
            # Checks of some labeler was passed with which the data can be split
            if labeler is None:
                warnings.warn("No labeler was specified. Splitting data at the intervention point",
                              category=UserWarning)
                # TODO: test this automatic splitting function
                self.disc_data = util.split_data(self.cont_data,
                                                 lambda x: int(x > self.intervention_point))
            else:
                # Splits the continuous data using the labeler
                self.disc_data = util.split_data(self.cont_data, labeler)

        # Initializes the continuous model if it's None
        if self.cont_m is None:
            # Uses BNQDflow's custom kernel copying function if both bnqdflow.USE_CUSTOM_KERNEL_COPY_FUNCTION is true
            # and the regression object is a kernel
            if isinstance(self.__regression_source, Kernel) and False:
                print("what the fuck are you doing, Python?")
                regression_object = util.copy_kernel(self.__regression_source)
            else:
                regression_object = gf.utilities.deepcopy(self.__regression_source)

            self.cont_m = GPMContainer(regression_object,
                                       data_list=[self.cont_data],
                                       **self.__container_kwargs)
        else:
            # If it already exists, checks if the data the model uses is the same as the data in the analysis object
            if not (self.cont_m.data == self.cont_data):
                warnings.warn("The continuous model isn't using the same data as the continuous data contained in the"
                              "analysis object.")

        # Initializes the discontinuous model if it's None
        if self.disc_m is None:
            # Uses BNQDflow's custom kernel copying function if both bnqdflow.USE_CUSTOM_KERNEL_COPY_FUNCTION is true
            # and the regression object is a kernel
            if (isinstance(self.__regression_source, Kernel) and False):
                regression_object = util.copy_kernel(self.__regression_source)
            else:
                regression_object = gf.utilities.deepcopy(self.__regression_source)

            self.disc_m = GPMContainer(regression_object,
                                       data_list=self.disc_data,
                                       intervention_points=[self.intervention_point],
                                       **self.__container_kwargs)
        else:
            # If it already exists, checks if the data the model uses is the same as the data in the analysis object
            if not (self.disc_m.data == self.disc_data):
                warnings.warn("The discontinuous model isn't using the same data as the discontinuous data contained in"
                              "the analysis object.")

        self.__regression_source = None
        self.__container_kwargs = None

    @property
    def shared_params(self):
        if self.cont_m is not None and self.disc_m is not None:
            return self.cont_m.shared_params and self.disc_m.shared_params
        else:
            return self.__shared_params

    @shared_params.setter
    def shared_params(self, val):
        self.__shared_params = val

    @property
    def cont_data(self):
        if self.cont_m is None:
            return self.__cont_data
        else:
            self.__cont_data = None
            return self.cont_m.data_list[0]

    @cont_data.setter
    def cont_data(self, data):
        if self.cont_m is None:
            self.__cont_data = data

    @property
    def disc_data(self):
        if self.disc_m is None:
            return self.__disc_data
        else:
            self.__disc_data = None
            return self.disc_m.data_list

    @disc_data.setter
    def disc_data(self, data):
        if self.disc_m is None:
            self.__disc_data = data

    @property
    def N(self):
        Nc = self.cont_m.N
        Nd = self.disc_m.N
        return Nc if Nc == Nd else None

    @property
    def containers(self):
        return (self.cont_m, self.disc_m)

    def train(
            self,
            optimizer: object = None,
            scipy_method: Optional[str] = None,
            max_iter: Optional[int] = 1000,
            loss_variance_goal: Optional[float] = None):
        """
        Trains both the continuous and the discontinuous model
        """
        # Initializes the models if either is None.
        # Separating this step from the SimpleAnalysis.__init__() function allows for manual specification of the
        # continuous and discontinuous data via set_continuous_data() and set_discontinuous_data() before assigning it
        # to the models
        if not self.cont_m or not self.disc_m:
            self.init_models()

        # Uses the optimizer passed to the function, if not,
        # then the optimizer assigned to the BNQDAnalysis object.
        # Will only be None if both are None
        optimizer = optimizer if optimizer is not None else self.optimizer

        res = list()

        for container in self.containers:
            res.append(container.train(optimizer=optimizer,
                                       scipy_method=scipy_method,
                                       max_iter=max_iter,
                                       loss_variance_goal=loss_variance_goal))
        return res

    def sample_posterior_params(self, *args, **kwargs):
        res = list()

        for container in self.containers:
            res.append(container.sample_posterior_params(*args, **kwargs))

        return res

    def log_bayes_factor(self, method: str = None, verbose: bool = False, *args, **kwargs) -> tf.Tensor:
        """
        Computes the Bayes factor of the two models

        :param verbose: Whether or not to plint the Bayes factor.
        :param method: Method used for calculating the marginal likelihood (BIC or the native GPflow method)
        :return: Bayes factor of the discontinuous model to the continuous model: $BF_{M_D M_C}$
        """
        # Results in True if and only if both models are trained
        if not all(map(lambda m: m.is_trained, [self.cont_m, self.disc_m])):
            msg = "Not all models have been trained, so the Bayes factor will not be representative.\n" \
                  "Assuming your Analysis object is called 'a', you can check this with:\n" \
                  "\t'a.cont_m.is_trained' and 'a.disc_m.is_trained'\n" \
                  "Train both models at the same time with 'a.train()'"
            warnings.warn(msg, category=UserWarning)

        # Determines which marginal likelihood computation method to use.
        # Uses the method passed to the function, if it exists. Otherwise, uses the method assigned to the object.
        method = method if method else self.marginal_likelihood_method

        # Computes the Bayes factor by subtracting the two tensors element-wise, on the first axis.
        # Typically, these tensors will only contain one element.
        log_bf = tf.reduce_sum([self.disc_m.log_posterior_density(method, *args, **kwargs),
                                -self.cont_m.log_posterior_density(method, *args, **kwargs)], 0)
        if verbose:
            print("Bayes factor Md-Mc: {}".format(log_bf))
        return log_bf

    def posterior_model_probabilities(self, method: str = None, *args, **kwargs):
        """
        Gives the posterior model probabilities as a tuple.

        :param method: Method used to estimate the marginal likelihood.
        :return: (continuous posterior probability, discontinuous posterior probability)
        """
        bf = tf.exp(self.log_bayes_factor(method, *args, **kwargs))
        if tf.math.is_inf(bf):
            return 0.0, 1.0
        else:
            discont_prob = bf / (1 + bf)
            cont_prob = 1 - discont_prob
            return cont_prob, discont_prob

    def get_effect_size(self, measure=None, force_recalc: bool = False) -> dict:
        """
        Calculates the effect size and related statistics. Formatted as a dictionary.

        :param measure: EffectSizeMeasure object to calculate the effect size with.
        :param force_recalc: Whether or not to force recalculation of the effect size.
        """
        # Checks if the passed parameter exists
        if measure:
            print("esm passed")
            # Changes the effect size measure to the passed parameter
            self.effect_size_measure = measure

        elif self.effect_size_measure is None:
            raise ValueError("No effect size measure was specified. See bnqdflow.effect_size_measures")

        # Only recalculates the effect size if either an EffectSizeMeasure was specified on force_recalc is True
        if measure is not None or force_recalc or self.effect_size_measure.effect_size is None:
            self.effect_size_measure.calculate_effect_size(self)

        return self.effect_size_measure.effect_size

    def get_reduced_effect_size(self):
        return self.effect_size_measure.calculate_reduced_effect_size(self)

    def plot_regressions(self, n_samples=100, padding: Union[float, Tuple[float]] = 0.2, num_f_samples=5,
                         plot_data=True, predict_y=False, separate=True):
        """
        Plots both the continuous and the discontinuous model
        """
        # TODO: return a pyplot object instead to allow separate plotting
        # TODO: allow for splitting into one plot per model, for better visibility
        f = plt.figure()
        self.cont_m.plot_regression(n_samples, padding, num_f_samples, (plot_data if separate else False), predict_y)
        if separate:
            f.savefig("gp_reg.svg")
            plt.show()
            f = plt.figure()
        self.disc_m.plot_regression(n_samples, padding, num_f_samples, plot_data, predict_y)
        if separate:
            plt.show()
        f.savefig("gp_reg2.svg")


class MultiKernelAnalysis(Analysis):
    def __init__(
            self,
            kernels: Union[List[Kernel], Dict[str, Kernel]],
            data: Union[ContinuousData, DiscontinuousData],
            intervention_point: Tensor,
            likelihood: Optional[Likelihood] = Gaussian(),
            share_params: bool = True,
            gpm_type: Optional[str] = 'gpr',
            inducing_var_ratio: Optional[Union[Tuple[float, ...], float]] = 0.01,
            optimizer: Any = None,
            effect_size_measure=None):

        def kernel_name(kernel) -> str:
            if isinstance(kernel, gf.kernels.Combination):
                reducer = ''
                name = ""

                for sub_kernel in kernel.kernels:
                    sub_name = kernel_name(sub_kernel)
                    sub_name = f"({sub_name})" if isinstance(sub_kernel, gf.kernels.Combination) else sub_name
                    name += reducer + sub_name

                    if reducer == '':
                        reducer = '+' if isinstance(kernel, gf.kernels.Sum) else '*'

                return name

            else:
                return kernel.name

        if type(kernels) is list:
            kernels = dict(zip(map(kernel_name, kernels), kernels))

        self.analyses = list()

        for name, kernel in kernels.items():
            analysis = SimpleAnalysis(kernel,
                                      data=data,
                                      intervention_point=intervention_point,
                                      likelihood=likelihood,
                                      share_params=share_params,
                                      gpm_type=gpm_type,
                                      inducing_var_ratio=inducing_var_ratio,
                                      optimizer=deepcopy(optimizer),
                                      effect_size_measure=deepcopy(effect_size_measure))

            self.analyses.append(analysis)

        self.effect_size_measure = effect_size_measure
        self.kernel_names = kernels.keys()
        super().__init__(data, intervention_point, share_params, optimizer, effect_size_measure)

    def init_models(self, labeler: Union[List[InputData], Callable[[InputData], int]] = None) -> None:
        self.analyses[0].init_models(labeler)
        for i in range(1, self.n_kernels):
            self.analyses[i].cont_data = self.analyses[0].cont_data
            self.analyses[i].disc_data = self.analyses[0].disc_data
            self.analyses[i].init_models()

    @property
    def __proxy(self):
        """
        The first analysis in the analyses dictionary.
        Merely here for convenience.
        """
        return self.analyses[0]

    @property
    def shared_params(self):
        if self.analyses is None:
            return self.__shared_params
        else:
            return self.__proxy.shared_params

    @shared_params.setter
    def shared_params(self, val):
        self.__shared_params = val

    @property
    def cont_data(self):
        return self.__proxy.cont_data

    @cont_data.setter
    def cont_data(self, data):
        for a in self.analyses:
            a.cont_data = data

    @property
    def disc_data(self):
        return self.__proxy.disc_data

    @disc_data.setter
    def disc_data(self, data):
        for a in self.analyses:
            a.disc_data = data

    @property
    def N(self):
        return self.__proxy.N

    @property
    def n_kernels(self):
        return len(self.kernel_names)

    @property
    def analysis_dict(self):
        return dict(zip(self.kernel_names, self.analyses))

    def train(self, optimizer: object = None, *args, **kwargs):
        if self.analyses[0] is None:
            self.init_models()
        return [a.train(deepcopy(optimizer), *args, **kwargs) for a in self.analyses]

    def log_bayes_factors(self, *args, **kwargs):
        return [a.log_bayes_factor(*args, **kwargs) for a in self.analyses]

    def get_effect_sizes(self, measure=None, *args, **kwargs):
        return [a.get_effect_size(deepcopy(measure), *args, **kwargs) for a in self.analyses]

    def get_reduced_effect_sizes(self):
        return [a.get_reduced_effect_size() for a in self.analyses]

    def total_log_bayes_factor(self, method=None, *args, **kwargs):
        cont_evidence = np.sum(np.exp([a.cont_m.log_posterior_density(method, *args, **kwargs) for a in self.analyses]))
        disc_evidence = np.sum(np.exp([a.disc_m.log_posterior_density(method, *args, **kwargs) for a in self.analyses]))

        return np.log(disc_evidence) - np.log(cont_evidence)

    def total_effect_size(self, measure=None, force_recalc=False) -> dict:
        # Checks if the passed parameter exists
        if measure:
            # Changes the effect size measure to the passed parameter
            self.effect_size_measure = measure

        elif self.effect_size_measure is None:
            raise ValueError("No effect size measure was specified. See bnqdflow.effect_size_measures")

        # Only recalculates the effect size if either an EffectSizeMeasure was specified on force_recalc is True
        if measure is not None or force_recalc or self.effect_size_measure.effect_size is None:
            self.effect_size_measure.calculate_effect_size(self)

        return self.effect_size_measure.effect_size

    def total_reduced_effect_size(self):
        return self.effect_size_measure.calculate_reduced_effect_size(self)

    def posterior_kernel_probabilities(self):
        if self.effect_size_measure.effect_size is None:
            self.calculate_effect_size()
        return dict(zip(self.kernel_names, self.effect_size_measure.effect_size['k_probs']))

    def plot_kernel_effect_sizes(self):
        ydim = math.ceil(np.sqrt(self.n_kernels))
        xdim = math.ceil(self.n_kernels / ydim)
        fig, axes = plt.subplots(ydim, xdim, sharex=True, sharey=True, figsize=(5*ydim, 6*xdim))
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.2, top=0.9)
        for ax, name, es, r_es in zip(axes.flatten(), self.kernel_names,
                                      self.get_effect_sizes(force_recalc=False), self.get_reduced_effect_sizes()):
            x = es['es_range']
            disc = es['es_disc']
            bma = es['es_bma']
            ax.fill_between(x, bma, label='BMA ES', alpha=0.5, color="#05668d")
            ax.fill_between(x, disc, label='Disc ES', alpha=0.5, color="#02c39a")
            ax.legend()
            ax.set_ylabel(r"$p(d=x)$")
            ax.set_xlabel(r"$d$")
            ax.set_title(f"{name}\n" fr"$\mathbb{{E}}[d|\mathcal{{M}}_{{Disc}}]={r_es['es_disc']:.2f}$"
                         fr"$\quad|\quad\mathbb{{E}}[d|BMA]={r_es['es_bma']:.2f}$")
        fig.suptitle("Kernel effect size estimates\n\n.")

    def plot_kernel_probabilities(self, include: Optional[list] = ['kernel effect sizes', 'kernel probs', 'total effect size']):
        k_probs = self.effect_size_measure.effect_size['k_probs']
        m_probs = self.effect_size_measure.effect_size['m_probs']

        c_k_probs = k_probs * [c_p for c_p, dc_p in m_probs]
        dc_k_probs = k_probs * [dc_p for c_p, dc_p in m_probs]

        fig, ax = plt.subplots()
        ax.bar(self.kernel_names, dc_k_probs, label=r"$p(k|D,\mathcal{M}_1)$", color="#05668d")
        ax.bar(self.kernel_names, c_k_probs, label=r"$p(k|D,\mathcal{M}_0)$", bottom=dc_k_probs, color="#02c39a")
        ax.set_title("posterior kernel probabilities\n" r"split between $\mathcal{M}_0$ and $\mathcal{M}_1$")
        fig.autofmt_xdate()
        ax.legend()

        print(f"log bf: {np.log(np.sum(dc_k_probs) / np.sum(c_k_probs))}")

    def map(self, fn: Callable):
        return [fn(a) for a in self.analyses]


class PlaceholderAnalysis(Analysis):
    """
    Just a blank implementation of Analysis used for testing the interaction with EffectSizeMeasure objects.
    """
    pass
