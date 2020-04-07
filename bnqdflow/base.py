from gpflow.models import GPModel

from typing import List, Union, Tuple

from gpflow.models.model import RegressionData


# Data for the continuous model: tuple of tensors / ndarrays
ContinuousData = RegressionData #Tuple[Union[Tensor, ndarray], Union[Tensor, ndarray]]

# Data for the discontinuous model: list of continuous data
DiscontinuousData = List[ContinuousData]


def IS_GPMODEL(o):
    return isinstance(o, GPModel)
