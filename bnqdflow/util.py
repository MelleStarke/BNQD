import abc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gf
import warnings
import copy
import sys

from bnqdflow.data_types import ContinuousData, DiscontinuousData

from typing import Optional, Tuple, Union, List, Callable, Any

from numpy import ndarray

from tensorflow import Tensor

from gpflow import optimizers
from gpflow.kernels import Kernel, Constant, Linear, Exponential, SquaredExponential, Periodic, Cosine
from gpflow.models import GPModel, GPR
from gpflow.likelihoods import Likelihood, Gaussian
from gpflow.models.model import DataPoint, MeanAndVariance, Data
from gpflow.mean_functions import MeanFunction


def ensure_tf_vector_format(x: Union[Tensor, ndarray]):
    if len(np.shape(x)) == 1:
        return x[:, None]
    if len(np.shape(x)) == 0:
        return np.array([[x]])
    return x


def ensure_tensor(x: Union[Tensor, ndarray]) -> Tensor:
    if not tf.is_tensor(x):
        return tf.convert_to_tensor(x)
    return x


def is_continuous_data(data: Union[ContinuousData, DiscontinuousData]):
    return type(data) is tuple


def flatten_data(data: DiscontinuousData) -> ContinuousData:
    """
    Turns data fit for the discontinuous model into data fit for the continuous model.
    :param data:
    :return:
    """
    x_res, y_res = np.array([]), np.array([])
    for x, y in data:
        x_res = np.append(x_res, x)
        y_res = np.append(y_res, y)
    return tf.convert_to_tensor(x_res), tf.convert_to_tensor(y_res)


def split_data(data: ContinuousData, criterion: Union[List[DataPoint], Callable[[DataPoint], int]]):
    raise NotImplementedError("Haven't implemented data splitting yet. Though you can manually format your data into "
                              "discontinuous data and use flatten_data()")


def linear_dummy_data(N=50, range=(0., 10.), a=3, b=0.7, noise=2, ip=5, dc=1.5):
    xs = np.random.uniform(range[0], range[1], size=N)
    ys = np.array([np.random.normal(a * x + b + (ip if x > ip else 0), noise) for x in xs])
    return xs, ys


###### Visitor Pattern Stuff ######
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
