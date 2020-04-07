"""
Script for messing around with the implementation, 3rd party libraries, and Python in general.
Not meant for exhaustive testing of the implementation.
"""

from __future__ import generators
# import BNQD
import gpflow as gf
import numpy as np
import tensorflow as tf
import time
import abc
import gpflow
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
import random

from typing import List, Tuple

from bnqdflow.util import visitor, Step
from bnqdflow.models import GPMContainer, ContinuousModel, DiscontinuousModel

import gpflow.mean_functions as mf
from gpflow.kernels import RBF, Linear, Constant, Static, White, SquaredExponential
from gpflow.models import GPR

import bnqdflow

import random

from bnqdflow.effect_size_measures import Sharp
from bnqdflow.analyses import Analysis


class A:
    def __init__(self):
        self.type = 'A'

    def accept(self, b):
        b.visit(self)


class A1(A):
    pass


class A2(A):
    pass


class B(abc.ABC):
    def __init__(self):
        self.type = 'B'

    @visitor(A1)
    def visit(self, a):
        print("{} visited a1".format(self.__class__.__name__))

    @visitor(A2)
    def visit(self, a):
        print("{} visited a2".format(self.__class__.__name__))


class B1(B):

    @visitor(A1)
    def visit(self, a):
        print("{} visited a1".format(self.__class__.__name__))

    @visitor(A2)
    def visit(self, a):
        print("{} visited a2".format(self.__class__.__name__))


class B2(B):
    pass




#####################################
###### Test Dataset Parameters ######
#####################################

ip = 0.  # Intervention point
dc = .0  # Discontinuity
sigma = 0.1  # Standard deviation
sigma_d = 0.  # Value added to the standard deviation after the intervention point
n = 100  # Number of data points


############################
###### Kernel Options ######
############################

#k = Constant() + Linear()  # "Linear" kernel
#k = Exponential()
k = SquaredExponential()
#k = Periodic(SquaredExponential())
#k = Cosine() + Constant()


###########################################
###### Generation of Test Dataset ######
###########################################

x = np.linspace(-3, 3, n)  # Evenly distributed x values

# Latent function options
f = 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 4) + dc * (x > ip)  # Underlying function
y = np.random.normal(f, sigma + sigma_d * (x > ip), size=n)  # y values as the underlying function + noise

# Uncomment to flip the data along the y-axis
y = np.flip(y)

# Data used by the control model (pre-intervention)
xc = x[x <= ip]
yc = y[x <= ip]

# Data used by the (post-)intervention model
xi = x[x > ip]
yi = y[x > ip]


gpc = GPMContainer(SquaredExponential(), [(x, y)])
gpc.train()