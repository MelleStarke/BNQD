from tensorflow import Tensor

from numpy import ndarray

from typing import List, Union, Tuple


# Data for the continuous model: tuple of tensors / ndarrays
ContinuousData = Tuple[Union[Tensor, ndarray], Union[Tensor, ndarray]]

# Data for the discontinuous model: list of continuous data
DiscontinuousData = List[ContinuousData]