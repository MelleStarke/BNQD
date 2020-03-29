import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

from abc import ABC

from typing import Union, List, Callable, Any

from tensorflow import Tensor

from gpflow.kernels import Kernel
from gpflow.models.model import InputData

from bnqdflow import util
from bnqdflow.models import (ContinuousModel,
                             DiscontinuousModel,
                             ContinuousData,
                             DiscontinuousData)


class Analysis(ABC):
    """
    Abstract analysis class
    """

    def __init__(self, data: Union[ContinuousData, DiscontinuousData], intervention_point: Tensor,
                 share_params: bool = False, marginal_likelihood_method: str = "bic", optimizer: Any = None,
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
        self._effect_size_measure = measure


class SimpleAnalysis(Analysis):
    """
    Simple BNQD analysis object.
    This contains both the continuous and the discontinuous model,
    as well as some methods for manipulating and comparing these models.
    """

    def __init__(self, data: Union[ContinuousData, DiscontinuousData], kernel: Kernel, intervention_point: Tensor,
                 share_params: bool = False, marginal_likelihood_method: str = "bic", optimizer: Any = None,
                 effect_size_measure=None):
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
        """

        # TODO: make the class able to handle ContinuousData, and then split it using the intervention_point
        # TODO: have two parameters for the data (continuous and discontinuous) so you won't have to keep
        #       splitting/flattening the data for multiple analyses objects
        super().__init__(data, intervention_point, share_params, marginal_likelihood_method, optimizer,
                         effect_size_measure)

        self.continuous_model = None
        self.discontinuous_model = None

        self.kernel = kernel

    def init_models(self, labeler: Union[List[InputData], Callable[[InputData], int]] = None, **kwargs) -> None:
        """
        Initializes the continuous and discontinuous model.
        Uses the continuous_data and discontinuous_data fields if they exist, and attempts to generate it if otherwise.

        For generation of the discontinuous data, a labeler should be specified. This can either be a list of data
        points, or a function that takes a data point and returns an int (0 or 1) representing whether or not that data
        point belongs to the control model or intervention model.
        If no labeler is specified, splitting is done on the intervention point.

        :param labeler: either a list of data points, or a function of type InputData -> int
        """

        if self.continuous_data is None:
            if self.discontinuous_data is None:
                raise RuntimeError("No data has been passed to the analysis object.")
            else:
                # Converts discontinuous data into continuous data
                self.continuous_data = util.flatten_data(self.discontinuous_data)

        elif self.discontinuous_data is None:
            if labeler is None:
                warnings.warn("No labeler was specified. Splitting data at the intervention point",
                              category=UserWarning)
                # TODO: test this automatic splitting function
                self.discontinuous_data = util.split_data(self.continuous_data,
                                                          lambda x: int(x > self.intervention_point))
            else:
                self.discontinuous_data = util.split_data(self.continuous_data, labeler)

        self.continuous_model = ContinuousModel(self.continuous_data, self.kernel, **kwargs)
        self.discontinuous_model = DiscontinuousModel(self.discontinuous_data, self.kernel, self.intervention_point,
                                                      self.share_params, **kwargs)

    def train(self, optimizer=None):
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

    def log_bayes_factor(self, method: str = None, verbose: bool = False) -> tf.Tensor:
        """
        Computes the Bayes factor of the two models

        :param method: Method used for calculating the marginal likelihood (BIC or the native GPflow method)
        :return: Bayes factor of the discontinuous model to the continuous model: $BF_{M_D M_C}$
        """

        # Results in True if and only if both models are trained
        if not all(map((lambda m: m.is_trained), [self.continuous_model, self.discontinuous_model])):
            msg = "Not all models have been trained, so the Bayes factor will not be representative.\n" \
                  "Assuming your BNQDAnalysis object is called 'ba', you can check this with:\n" \
                  "\t'ba.continuous_model.is_trained' and 'ba.discontinuous_model.is_trained'\n" \
                  "Train both models at the same time with 'ba.train()'"
            warnings.warn(msg, category=UserWarning)

        # Determines which marginal likelihood computation method to use.
        # Uses the method passed to the function, if it exists. Otherwise, uses the method assigned to the object.
        method = method if method else self.marginal_likelihood_method

        # Computes the Bayes factor by subtracting the two tensors element-wise, on the first axis.
        # Typically, these tensors will only contain one element.
        log_bf = tf.reduce_sum([self.discontinuous_model.log_marginal_likelihood(method),
                                -self.continuous_model.log_marginal_likelihood(method)], 0)
        if verbose:
            print("Bayes factor $M_D - M_C$: {}".format(log_bf))
        return log_bf

    def posterior_model_probabilities(self, method: str = None):
        """
        Gives the posterior model probabilities as a tuple
        :param method:
        :return: (continuous posterior probability, discontinuous posterior probability)
        """
        bf = tf.exp(self.log_bayes_factor(method))
        if tf.math.is_inf(bf):
            return 0.0, 1.0
        else:
            cont_prob = bf / (1 + bf)
            discont_prob = 1 - bf
            return cont_prob, discont_prob

    def get_effect_size(self, measure=None, force_recalc: bool = False) -> dict:
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
    pass
