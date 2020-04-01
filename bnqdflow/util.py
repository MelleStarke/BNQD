import gpflow as gf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from typing import Union, List, Callable

from gpflow.mean_functions import MeanFunction
from gpflow.models.model import InputData

from numpy import ndarray

from tensorflow import Tensor

from bnqdflow.data_types import ContinuousData, DiscontinuousData


###############################
###### Data Manipulation ######
###############################

def ensure_tf_vector_format(x: Union[Tensor, ndarray]) -> Tensor:
    """
    Ensures the input is in a shape usable by tensorflow.
    Meaning, an at least 2-dimensional tensor.
    The 2nd dimension will have a length of 1 if the input is a 1-dimensional tensor/ndarray.
    """
    if len(np.shape(x)) == 1:
        if np.shape(x)[0] == 0:
            return tf.convert_to_tensor(np.empty(2))
        return x[:, None]
    if len(np.shape(x)) == 0:
        return tf.convert_to_tensor(np.array([[x]]))
    return tf.convert_to_tensor(x)


def ensure_tensor(x: Union[Tensor, ndarray]) -> Tensor:
    """
    Converts the input to a tensor if it's not already.
    THIS FUNCTION IS PROBABLY REDUNDANT.
    """
    if not tf.is_tensor(x):
        return tf.convert_to_tensor(x)
    return x


def is_continuous_data(data: Union[ContinuousData, DiscontinuousData]):
    """
    Returns True if the input data is in the shape of continuous data
    """
    return type(data) is tuple


def flatten_data(data: DiscontinuousData) -> ContinuousData:
    """
    Turns discontinuous data into continuous data.
    """
    x_res, y_res = np.array([]), np.array([])
    for x, y in data:
        x_res = np.append(x_res, x)
        y_res = np.append(y_res, y)
    return tf.convert_to_tensor(x_res), tf.convert_to_tensor(y_res)


# TODO: write this function
def split_data(data: ContinuousData, criterion: Union[List[InputData], Callable[[InputData], int]]):
    raise NotImplementedError("Haven't implemented data splitting yet. Though you can manually format your data into "
                              "discontinuous data and use flatten_data()")


def linear_dummy_data(n=50, x_range=(0., 10.), a=3, b=0.7, noise=2, ip=5, dc=1.5):
    """
    Generates a very simple dataset (with a discontinuity) from a linear latent function.

    :param n: Number of samples
    :param x_range: x-range
    :param a: Slope of the linear latent function
    :param b: Bias of the linear latent function
    :param noise: Variance over the latent function
    :param ip: Intervention point
    :param dc: x-value of the discontinuity
    :return: Tuple of two 1-dimensional numpy arrays
    """
    xs = np.random.uniform(x_range[0], x_range[1], size=n)
    ys = np.array([np.random.normal(a * x + b + (dc if x > ip else 0), noise) for x in xs])
    return xs, ys


##############################
###### Plotting Methods ######
##############################

def plot_regression(x, mean, std, col="blue", alpha=0.2):
    """
    Plots a GP regression.

    :param x: x-values
    :param mean: Mean values
    :param std: Standard deviation values
    :param col: Color used for plotting
    :param alpha: Opacity of the bands
    :return:
    """
    plt.plot(x, mean, c=col)
    plt.fill_between(x, mean - 1.96 * std, mean + 1.96 * std, color=col, alpha=alpha)


###################################
###### Custom Mean Functions ######
###################################

class Step(MeanFunction):
    """
    MeanFunction that switches between two other MeanFunctions at x=s.
    s is not a Parameter because it causes a tensorflow error.
    Though it's not a very helpful class, cause it allows for overfitting.
    """
    def __init__(self, fst: MeanFunction, snd: MeanFunction, s=None):
        super().__init__()
        s = np.zeros(1) if s is None else s
        # self.s = gpflow.base.Parameter(tf.squeeze(s))
        self.s = s

        assert isinstance(fst, MeanFunction)
        self.fst = fst

        assert isinstance(snd, MeanFunction)
        self.snd = snd

    def __call__(self, X):
        """
        Calls the first MeanFunction for all x-values less than self.s, and the second one for the others.
        :param X:
        :return:
        """
        # Generates a tensor of bools.
        cond = tf.math.less(X, tf.reshape(self.s, (1, -1)))
        # Composes a tensor from two tensors.
        # Calls self.fst where cond=True, and self.snd where cond=False
        res = tf.where(cond, self.fst(X), self.snd(X))
        return res


#########################################
###### Methods Useless in Practice ######
#########################################

target_types = (gf.Parameter, tf.Variable)


def equalize_parameters(source: tf.Module, target: tf.Module, path=None):
    """
    ####################
    ### DOESN'T WORK ###
    ####################
    The idea was that this function made all pointers of the target types defined above point to the same object.
    But it doesn't do anything. The structure might be helpful in the future though.
    """
    if not (type(source) is type(target)):
        raise ValueError("Source and target module aren't of the same type.\n"
                         "\tPath: {}\n\tSource class: {}\n\t Target class: {}"
                         .format(path, source.__class__.__name__, target.__class__.__name__))

    if path is None:
        path = source.__class__.__name__

    res = list()

    if isinstance(source, target_types):
        target = source
        res += [path]

    elif isinstance(source, (list, tuple)):
        for i, (sub_source, sub_target) in enumerate(zip(source, target)):
            new_path = f"{path}[{i}]"
            res += equalize_parameters(sub_source, sub_target, new_path)

    elif isinstance(source, dict):
        for (source_key, sub_source), (target_key, sub_target) in zip(source.items(), target.items()):
            new_path = f"{path}['{source_key}']"
            res += equalize_parameters(sub_source, sub_target, new_path)

    elif isinstance(source, tf.Module):
        for (source_name, sub_source), (target_name, sub_target) in zip(vars(source).items(), vars(target).items()):
            if source_name in tf.Module._TF_MODULE_IGNORED_PROPERTIES:
                continue
            new_path = f"{path}.{source_name}"
            res += equalize_parameters(sub_source, sub_target, new_path)
    return res


############################################
###### Visitor Pattern Implementation ######
############################################

"""
Visitor implementation using a decorator.
Taken from https://tavianator.com/the-visitor-pattern-in-python/.
Written by Tavian Barnes, June 19, 2014.
Adapted with a better error message.
"""


def _qualname(obj):
    """Get the fully-qualified name of an object (including module)."""
    return obj.__module__ + '.' + obj.__qualname__


def _declaring_class(obj):
    """Get the name of the class that declared an object."""
    name = _qualname(obj)
    return name[:name.rfind('.')]


# Stores the actual visitor methods
_methods = {}


# Delegating visitor implementation
def _visitor_impl(self, arg):
    """Actual visitor method implementation."""
    try:
        method = _methods[(_qualname(type(self)), type(arg))]
    except KeyError as e:
        raise KeyError("{}. Likely because there does not exist an implementation of the function in question in class "
                       "{} for visitor {}".format(e, self.__class__.__name__, arg.__class__.__name__))
    return method(self, arg)


# The actual @visitor decorator
def visitor(arg_type):
    """Decorator that creates a visitor method."""

    def decorator(fn):
        declaring_class = _declaring_class(fn)
        _methods[(declaring_class, arg_type)] = fn

        # Replace all decorated methods with _visitor_impl
        return _visitor_impl

    return decorator
