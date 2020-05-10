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
import bnqdflow as bf

from typing import List, Tuple

from bnqdflow.util import visitor, Step, copy_kernel
from bnqdflow.models import GPMContainer, ContinuousModel, DiscontinuousModel

from tensorflow_probability import distributions as dist

import gpflow.mean_functions as mf
from gpflow.kernels import *
from gpflow.models import GPR

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

ip = 0  # Intervention point
ip2 = 2.3
dc = .5  # Discontinuity
sigma = 0.2  # Standard deviation
sigma_d = 0.  # Value added to the standard deviation after the intervention point
n = 10  # Number of data points


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

np.random.seed(3)

x = np.linspace(-3, 3, n)  # Evenly distributed x values

# Latent function options
#f = 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 4) + dc * (x > ip)  # Underlying function
f = (0.8 - 1.6 * (x > ip))*x + dc * (x > ip)
y = np.random.normal(f, sigma + sigma_d * (x > ip), size=n)  # y values as the underlying function + noise

# Uncomment to flip the data along the y-axis
#y = np.flip(y)

# Data used by the control model (pre-intervention)
xc = x[x <= ip]
yc = y[x <= ip]

xd = x[x > ip]
yd = y[x > ip]

# Data used by the (post-)intervention model
xi = xd[xd <= ip2]
yi = yd[xd <= ip2]

xe = xd[xd > ip2]
ye = yd[xd > ip2]

k = Linear() + Constant()

m = GPMContainer(k, [(xc, yc), (xd, yd)], [ip])
m.train()
m.plot_regression()
plt.show()

"""ks[1].variance.prior = dist.Gamma(np.float64(20), np.float64(4.35))
m1 = None
for k in ks:
    m1 = GPMContainer(gf.utilities.deepcopy(k), [(x, y)], [])
    m2 = GPMContainer(gf.utilities.deepcopy(k), [(xc, yc), (xd, yd)], [ip])
    for name, m in zip(['c', 'd'], [m1, m2]):
        m.train()
        m.plot_regression()
        plt.show()
        print(f"{name} l: {m.log_posterior_density()}")

print(f"trainable parameters: {m1.trainable_parameters}")
print(f"log prior density: {m1.kernel.variance.log_prior_density()}")"""
