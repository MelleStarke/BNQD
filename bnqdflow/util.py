import gpflow as gf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import warnings

from typing import Union, List, Callable

from gpflow.mean_functions import MeanFunction
from gpflow.models.model import InputData
from gpflow.kernels import *
from gpflow.utilities import parameter_dict, multiple_assign

from numpy import ndarray

from tensorflow import Tensor

from bnqdflow import ContinuousData, DiscontinuousData


###############################
###### Data Manipulation ######
###############################

def ensure_tf_matrix(x: Union[Tensor, ndarray]) -> Tensor:
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
    plt.plot(x, mean, c=col, alpha=2*alpha)
    plt.fill_between(x, mean - 1.96 * std, mean + 1.96 * std, color=col, alpha=alpha)


##############################
###### Kernel Utilities ######
##############################

def copy_kernel(kernel: Kernel) -> Kernel:
    """
    Returns a copy of the input kernel with the same values but different pointers.

    Should only be used if copy.deepcopy or gpflow.utilities.deepcopy don't work.
    Also checks whether or not the tf.Module objects are actually different.
    If you do not wish to check this, use _copy_kernel() instead.
    """
    new_kernel = _copy_kernel(kernel)

    # The next bit of code checks whether the copying process was successful

    # Function applied to all objects contained in the kernels (of the target types)
    def same_objects(source: tf.Module, target: tf.Module) -> bool:
        return source is target

    # Types of the objects shat should be compared
    target_types = (tf.Module,)

    try:
        # List of strings (or paths) indicating what objects are the same
        same_object_paths = compare_modules(kernel, new_kernel, same_objects, target_types)

        if len(same_object_paths) > 0:
            msg = "One or more objects are shared between the kernels. So they're not true copies of each other:"
            for path in same_object_paths:
                msg += f"\n\t{path}"
            warnings.warn(msg)

        return new_kernel

    except ValueError as e:
        raise ValueError(f"The kernels contain objects of different types. Therefore the result of the _copy_kernel "
                         f"function cannot be validated:\n{e}")


def _copy_kernel(kernel: Kernel) -> Kernel:
    """
    Returns a copy of the input kernel with the same values but different pointers.

    Doesn't check whether or not the tf.Module objects are actually different.
    """
    # Case for when the kernel is a sum of multiple kernels
    if type(kernel) is Sum:
        res = _copy_kernel(kernel.kernels[0])
        for sub_kernel in kernel.kernels[1:]:
            res += _copy_kernel(sub_kernel)
        return res

    # Case for when the kernel is a product of multiple kernels
    if type(kernel) is Product:
        res = _copy_kernel(kernel.kernels[0])
        for sub_kernel in kernel.kernels[1:]:
            res *= _copy_kernel(sub_kernel)
        return res

    # Case for when the kernel is convolutional
    if type(kernel) is Convolutional:
        image_shape = kernel.image_shape
        patch_shape = kernel.patch_shape
        base_kernel = _copy_kernel(kernel.base_kernel)
        colour_channels = kernel.colour_channels
        res = Convolutional(_copy_kernel(base_kernel), image_shape, patch_shape, colour_channels=colour_channels)
        multiple_assign(res, parameter_dict(kernel))
        return res

    # Case for when the kernel is a change-point kernel
    if type(kernel) is ChangePoints:
        kernels = list(map(_copy_kernel, kernel.kernels))
        locations = kernel.locations
        steepness = kernel.steepness
        name = kernel.name
        res = ChangePoints(kernels, locations, steepness=steepness, name=name)
        multiple_assign(res, parameter_dict(kernel))
        return res

    # Case for when the kernel is periodic
    if type(kernel) is Periodic:
        base_kernel = _copy_kernel(kernel.base_kernel)
        period = kernel.period
        res = Periodic(base_kernel, period=period)
        multiple_assign(res, parameter_dict(kernel))
        return res

    # Case for when the kernel is (an instance of) a linear kernel
    if isinstance(kernel, Linear):
        variance = kernel.variance
        active_dims = kernel.active_dims

        options = [
            Linear,
            Polynomial
        ]

        correct_classes = [o for o in options if type(kernel) is o]
        assert len(correct_classes) == 1, f"Only one class should match. List of correct classes: {correct_classes}"

        if type(kernel) is Polynomial:
            # Calls the constructor of the (only) correct kernel class
            res = correct_classes[0](variance=variance, active_dims=active_dims, degree=kernel.degree)
        else:
            # Calls the constructor of the (only) correct kernel class
            res = correct_classes[0](variance=variance, active_dims=active_dims)
        multiple_assign(res, parameter_dict(kernel))
        return res

    # Case for when the kernel is an instance of a static kernel
    if isinstance(kernel, Static):
        active_dims = kernel.active_dims

        options = [
            White,
            Constant
        ]

        correct_classes = [o for o in options if type(kernel) is o]
        assert len(correct_classes) == 1, f"Only one class should match. List of correct classes: {correct_classes}"

        # Calls the constructor of the (only) correct kernel class
        res = correct_classes[0](active_dims=active_dims)
        multiple_assign(res, parameter_dict(kernel))
        return res

    # Case for when the kernel is an instance of a stationary kernel
    if isinstance(kernel, Stationary):
        active_dims = kernel.active_dims
        name = kernel.name
        variance = kernel.variance
        lengthscales = kernel.lengthscales

        options = [
            SquaredExponential,
            Cosine,
            Exponential,
            Matern12,
            Matern32,
            Matern52,
            RationalQuadratic
        ]

        correct_classes = [o for o in options if type(kernel) is o]
        assert len(correct_classes) == 1, \
            "Only one class should match. List of correct classes: {}".format(correct_classes)

        # Calls the constructor of the (only) correct kernel class
        res = correct_classes[0](variance=variance, lengthscales=lengthscales, active_dims=active_dims, name=name)
        multiple_assign(res, parameter_dict(kernel))
        return res

    raise ValueError(f"BNQDflow's copy_kernel function doesn't support this kernel type: {type(kernel)}")

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


##############################
###### Module Utilities ######
##############################

def compare_modules(source: tf.Module, target: tf.Module, condition: Callable[[object, object], bool],
                              target_types=(gf.Parameter, tf.Variable), path=None):
    """
    Recursively iterates over all tf.Module objects found in the source and the target
    and returns a list of paths for which the condition holds.

    The condition is a function that takes two arguments (obe object from the source and one object from the target)
    and returns a bool. If this bool is True, it appends the path to that object to the result.
    """
    if not (type(source) is type(target)):
        raise ValueError("Source and target module aren't of the same type.\n"
                         "\tPath: {}\n\tSource class: {}\n\t Target class: {}"
                         .format(path, source.__class__.__name__, target.__class__.__name__))

    if path is None:
        path = source.__class__.__name__

    res = list()

    if isinstance(source, target_types):
        if condition(source, target):
            res += [path]

    if isinstance(source, (list, tuple)):
        for i, (sub_source, sub_target) in enumerate(zip(source, target)):
            new_path = f"{path}[{i}]"
            res += compare_modules(sub_source, sub_target, condition, target_types, new_path)

    elif isinstance(source, dict):
        for (source_key, sub_source), (target_key, sub_target) in zip(source.items(), target.items()):
            new_path = f"{path}['{source_key}']"
            res += compare_modules(sub_source, sub_target, condition, target_types, new_path)

    elif isinstance(source, tf.Module):
        for (source_name, sub_source), (target_name, sub_target) in zip(vars(source).items(), vars(target).items()):
            if source_name in tf.Module._TF_MODULE_IGNORED_PROPERTIES:
                continue
            new_path = f"{path}.{source_name}"
            res += compare_modules(sub_source, sub_target, condition, target_types, new_path)
    return res


############################################
###### Visitor Pattern Implementation ######
############################################

"""
Visitor implementation using a decorator.
Taken from https://tavianator.com/the-visitor-pattern-in-python/.
Written by Tavian Barnes, June 19, 2014.
Adapted to support differently named decorated functions within the same class.
"""


def _qualname(obj):
    """Get the fully-qualified name of an object (including module)."""
    return obj.__module__ + '.' + obj.__qualname__


# Stores the actual visitor methods
_methods = {}


# Delegating visitor implementation
def _visitor_impl(fn_name, self, visitor_obj, *args, **kwargs):
    """Actual visitor method implementation."""
    try:
        method = _methods[(fn_name, type(visitor_obj))]
    except KeyError as e:
        raise KeyError("{}. Likely because there does not exist an implementation of the function "
                       "{} for visitor {}".format(e, fn_name, visitor_obj.__class__.__name__))
    return method(self, visitor_obj, *args, **kwargs)


# The actual @visitor decorator
def visitor(visitor_type):
    """Decorator that creates a visitor method."""
    def decorator(fn):
        function_name = _qualname(fn)
        _methods[(function_name, visitor_type)] = fn

        # Passes all required and optional arguments together with the full function name of the specific decorator use
        def wrapper(self, visitor_obj, *args, **kwargs):
            return _visitor_impl(function_name, self, visitor_obj, *args, **kwargs)

        return wrapper

    return decorator
