from .base import *
from . import (
    util,
    models,
    effect_size_measures,
    analyses,
)

# GPflow update/install command: pip install -U git+https://github.com/GPflow/GPflow.git@develop#egg=gpflow
# TODO: look into tensorflow_probability
# TODO: figure out where I can add simulated annealing

# TODO: find fuzzy dataset and large dataset
# TODO: make notebooks for testing
# TODO: add support for multidimensional input data
# TODO: add analysis class with multiple kernels
# TODO: figure out how to specify the required GPflow version
# TODO: test  the SimpleAnalysis with both the GPModel objects already made and the continuous and discontinuous models already made
# TODO: add data splitting function
# TODO: add fuzzy effect size measure
# TODO: test the correctness of the likelihoods of the control and intervention model separately