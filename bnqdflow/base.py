from gpflow.models import GPModel

from typing import List, Union, Tuple

from gpflow.models.model import RegressionData


class State:
    """
    Really ugly way to allow for setting and getting of global variables.
    """

    # If True, uses the copy_kernel function in bnqdflow.util when copying kernels.
    # If false, use the deepcopy function from gpflow.utilities
    use_custom_kernel_copy_function = False

# Data for the continuous model: tuple of tensors / ndarrays
ContinuousData = RegressionData  # Tuple[Union[Tensor, ndarray], Union[Tensor, ndarray]]

# Data for the discontinuous model: list of continuous data
DiscontinuousData = List[ContinuousData]


def IS_GPMODEL(o):
    return isinstance(o, GPModel)


def SET_USE_CUSTOM_KERNEL_COPY_FUNCTION(b: bool = True):
    State.use_custom_kernel_copy_function = b

@property
def USE_CUSTOM_KERNEL_COPY_FUNCTION():
    return State.use_custom_kernel_copy_function
