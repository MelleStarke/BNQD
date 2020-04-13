import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
import gpflow as gf
import bnqdflow as bf

from abc import ABC

from typing import Union, List, Callable, Any, Tuple

from tensorflow import Tensor

from gpflow.kernels import Kernel
from gpflow.models.model import GPModel, InputData

from bnqdflow import util
from bnqdflow.models import ContinuousModel, DiscontinuousModel
from bnqdflow import ContinuousData, DiscontinuousData


class Analysis(ABC):
    """
    Abstract analysis class
    """

    def __init__(self, data: Union[ContinuousData, DiscontinuousData], intervention_point: Tensor,
                 share_params: bool = True, marginal_likelihood_method: str = "bic", optimizer: Any = None,
                 effect_size_measure=None):

        self.continuous_data = None
        self.discontinuous_data = None

        if util.is_continuous_data(data):
            self.continuous_data = data
        else:
            self.discontinuous_data = data

        self.intervention_point = intervention_point
        self.share_params = share_params
        self.marginal_likelihood_method = marginal_likelihood_method
        self.optimizer = optimizer

        self._effect_size_measure = effect_size_measure

    def set_continuous_data(self, data: ContinuousData):
        self.continuous_data = data

    def set_discontinuous_data(self, data: DiscontinuousData):
        self.discontinuous_data = data

    def set_effect_size_measure(self, measure):
        """
        Sets the effect size measure.
        :param measure: Measure of type EffectSizeMeasure
        """
        self._effect_size_measure = measure


class SimpleAnalysis(Analysis):
    """
    Simple BNQD analysis object.
    This contains both the continuous and the discontinuous model,
    as well as some methods for manipulating and comparing these models.
    """

    def __init__(
            self,
            data: Union[ContinuousData, DiscontinuousData],
            regression_object: Union[Kernel, GPModel, Tuple[ContinuousModel, DiscontinuousModel]],
            intervention_point: Tensor,
            share_params: bool = True,
            marginal_likelihood_method: str = "bic",
            optimizer: Any = None,
            effect_size_measure=None):
        """
        :param data:
                    Data object used to train the models. Either a tuple of tensors, or a list of tuples of tensors.
        :param regression_object:
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
        super().__init__(data, intervention_point, share_params, marginal_likelihood_method, optimizer,
                         effect_size_measure)

        self.continuous_model = None
        self.discontinuous_model = None

        if isinstance(regression_object, tuple):
            assert len(regression_object) == 2, "Exactly two BNQDRegressionModels should be provided."
            cm, dm = regression_object
            assert isinstance(cm, ContinuousModel), "The first element of the tuple should be a ContinuousModel"
            assert isinstance(dm, DiscontinuousModel), "The second element of the tuple should be a DiscontinuousModel"

            self.continuous_model = cm
            self.discontinuous_model = dm

        else:
            self.__regression_object = regression_object

    def init_models(self, labeler: Union[List[InputData], Callable[[InputData], int]] = None) -> None:
        """
        Initializes the continuous and discontinuous model.
        Uses the continuous_data and discontinuous_data fields if they exist, and attempts to generate it if otherwise.

        For generation of the discontinuous data, a labeler should be specified. This can either be a list of data
        points, or a function that takes a data point and returns an int (0 or 1) representing whether or not that data
        point belongs to the control model or intervention model.
        If no labeler is specified, splitting is done on the intervention point.

        :param labeler: either a list of data points, or a function of type InputData -> int
        """
        # Ensures the function is only executed if either of the two models is None
        if self.continuous_model is not None and self.discontinuous_model is not None:
            warnings.warn("Both models have already been initialized", category=UserWarning)
            return

        # Generates the continuous data. But only if necessary and possible.
        if self.continuous_data is None:
            if self.discontinuous_data is None:
                raise RuntimeError("No data has been passed to the analysis object.")
            else:
                # Converts discontinuous data into continuous data
                self.continuous_data = util.flatten_data(self.discontinuous_data)

        # Generates the discontinuous data. But only if necessary and possible.
        elif self.discontinuous_data is None:
            # Checks of some labeler was passed with which the data can be split
            if labeler is None:
                warnings.warn("No labeler was specified. Splitting data at the intervention point",
                              category=UserWarning)
                # TODO: test this automatic splitting function
                self.discontinuous_data = util.split_data(self.continuous_data,
                                                          lambda x: int(x > self.intervention_point))
            else:
                # Splits the continuous data using the labeler
                self.discontinuous_data = util.split_data(self.continuous_data, labeler)

        # Initializes the continuous model if it's None
        if self.continuous_model is None:
            # Uses BNQDflow's custom kernel copying function if both bnqdflow.USE_CUSTOM_KERNEL_COPY_FUNCTION is true
            # and the regression object is a kernel
            if isinstance(self.__regression_object, Kernel) and bf.USE_CUSTOM_KERNEL_COPY_FUNCTION:
                regression_object = util.copy_kernel(self.__regression_object)
            else:
                regression_object = gf.utilities.deepcopy(self.__regression_object)

            self.continuous_model = ContinuousModel(regression_object, self.continuous_data)
        else:
            # If it already exists, checks if the data the model uses is the same as the data in the analysis object
            if not (self.continuous_model.data == self.continuous_data):
                warnings.warn("The continuous model isn't using the same data as the continuous data contained in the"
                              "analysis object.")

        # Initializes the discontinuous model if it's None
        if self.discontinuous_model is None:
            # Uses BNQDflow's custom kernel copying function if both bnqdflow.USE_CUSTOM_KERNEL_COPY_FUNCTION is true
            # and the regression object is a kernel
            if isinstance(self.__regression_object, Kernel) and bf.USE_CUSTOM_KERNEL_COPY_FUNCTION:
                regression_object = util.copy_kernel(self.__regression_object)
            else:
                regression_object = gf.utilities.deepcopy(self.__regression_object)

            self.discontinuous_model = DiscontinuousModel(regression_object, self.discontinuous_data,
                                                          self.intervention_point, self.share_params)
        else:
            # If it already exists, checks if the data the model uses is the same as the data in the analysis object
            if not (self.discontinuous_model.data == self.discontinuous_data):
                warnings.warn("The discontinuous model isn't using the same data as the discontinuous data contained in"
                              "the analysis object.")

    def train(self, optimizer=None, verbose=True):
        """
        Trains both the continuous and the discontinuous model
        """
        # Initializes the models if either is None.
        # Separating this step from the SimpleAnalysis.__init__() function allows for manual specification of the
        # continuous and discontinuous data via set_continuous_data() and set_discontinuous_data() before assigning it
        # to the models
        if not self.continuous_model or not self.discontinuous_model:
            self.init_models()

        # Uses the optimizer passed to the function, if not,
        # then the optimizer assigned to the BNQDAnalysis object.
        # Will only be None if both are None
        optimizer = optimizer if optimizer else self.optimizer

        # If neither the optimizer passed to the function, nor self.optimizer are None,
        # use this optimizer to train the models
        if optimizer:
            self.continuous_model.train(optimizer, verbose=verbose)
            self.discontinuous_model.train(optimizer, verbose=verbose)

        # If both are none, train the models with the default optimizer
        else:
            self.continuous_model.train(verbose=verbose)
            self.discontinuous_model.train(verbose=verbose)

    def plot_regressions(self, n_samples=100, padding: Union[float, Tuple[float]] = 0.2, num_f_samples=5,
                         plot_data=True, predict_y=False, separate=True):
        """
        Plots both the continuous and the discontinuous model
        """
        # TODO: return a pyplot object instead to allow separate plotting
        # TODO: allow for splitting into one plot per model, for better visibility
        self.continuous_model.plot_regression(n_samples, padding, num_f_samples, (plot_data if separate else False), predict_y)
        if separate:
            plt.show()
        self.discontinuous_model.plot_regression(n_samples, padding, num_f_samples, plot_data, predict_y)

    def log_bayes_factor(self, method: str = None, verbose: bool = False) -> tf.Tensor:
        """
        Computes the Bayes factor of the two models

        :param verbose: Whether or not to plint the Bayes factor.
        :param method: Method used for calculating the marginal likelihood (BIC or the native GPflow method)
        :return: Bayes factor of the discontinuous model to the continuous model: $BF_{M_D M_C}$
        """
        # Results in True if and only if both models are trained
        if not all(map(lambda m: m.is_trained, [self.continuous_model, self.discontinuous_model])):
            msg = "Not all models have been trained, so the Bayes factor will not be representative.\n" \
                  "Assuming your Analysis object is called 'a', you can check this with:\n" \
                  "\t'a.continuous_model.is_trained' and 'a.discontinuous_model.is_trained'\n" \
                  "Train both models at the same time with 'a.train()'"
            warnings.warn(msg, category=UserWarning)

        # Determines which marginal likelihood computation method to use.
        # Uses the method passed to the function, if it exists. Otherwise, uses the method assigned to the object.
        method = method if method else self.marginal_likelihood_method

        # Computes the Bayes factor by subtracting the two tensors element-wise, on the first axis.
        # Typically, these tensors will only contain one element.
        log_bf = tf.reduce_sum([self.discontinuous_model.log_posterior_density(method),
                                -self.continuous_model.log_posterior_density(method)], 0)
        if verbose:
            print("Bayes factor Md-Mc: {}".format(log_bf))
        return log_bf

    def posterior_model_probabilities(self, method: str = None):
        """
        Gives the posterior model probabilities as a tuple.

        :param method: Method used to estimate the marginal likelihood.
        :return: (continuous posterior probability, discontinuous posterior probability)
        """
        bf = tf.exp(self.log_bayes_factor(method))
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
            # Changes the effect size measure to the passed parameter
            self._effect_size_measure = measure

        elif self._effect_size_measure is None:
            raise ValueError("No effect size measure was specified. See bnqdflow.effect_size_measures")

        # Only recalculates the effect size if either an EffectSizeMeasure was specified on force_recalc is True
        if measure or force_recalc:
            self._effect_size_measure.calculate_effect_size(self)

        return self._effect_size_measure.effect_size


class PlaceholderAnalysis(Analysis):
    """
    Just a blank implementation of Analysis used for testing the interaction with EffectSizeMeasure objects.
    """
    pass
